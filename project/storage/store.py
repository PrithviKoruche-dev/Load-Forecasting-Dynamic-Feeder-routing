"""Repository: every read and write of the database goes through Store.

The store is safe to use from several threads (the dashboard runs one per browser session): SQLAlchemy
hands each call its own connection, and SQLite runs in WAL mode so readers do not block the writer.
"""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, delete, event, func, insert, select, update
from sqlalchemy.dialects import postgresql, sqlite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project/ -> import core
from core import config, ids  # noqa: E402
from core.logging_setup import get_logger  # noqa: E402
from storage import schema as S  # noqa: E402

log = get_logger("storage")


def _now():
    return dt.datetime.now().isoformat(timespec="seconds")


def _chunks(rows, n):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


class SchemaVersionError(RuntimeError):
    pass


class Store:
    def __init__(self, url=None):
        self.url = url or config.database_url()
        cfg = config.storage()
        self.batch_size = cfg["batch_size"]
        self.retention = cfg["retention"]
        self.import_cfg = cfg["forecast_import"]
        kwargs = {"future": True}
        if self.url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            path = self.url[len("sqlite:///"):]
            if self.url.startswith("sqlite:///") and path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)      # Linux /..., Windows C:/... or C:\...
        self.engine = create_engine(self.url, **kwargs)
        if self.engine.dialect.name == "sqlite":
            event.listen(self.engine, "connect", _sqlite_pragmas)
        S.metadata.create_all(self.engine)
        self._check_version()
        self.sync_assets()

    @property
    def dialect(self):
        return self.engine.dialect.name

    # ------------------------------------------------------------------ helpers
    def _insert(self, table, ignore=False, upsert_keys=None, update_cols=None):
        """INSERT, INSERT ... ON CONFLICT DO NOTHING, or an upsert, in the right dialect."""
        if self.dialect in ("sqlite", "postgresql"):
            ins = (sqlite if self.dialect == "sqlite" else postgresql).insert(table)
            if upsert_keys:
                return ins.on_conflict_do_update(index_elements=upsert_keys,
                                                 set_={c: ins.excluded[c] for c in update_cols})
            return ins.on_conflict_do_nothing() if ignore else ins
        return insert(table)

    def _check_version(self):
        with self.engine.begin() as c:
            row = c.execute(select(S.meta.c.value).where(S.meta.c.key == "schema_version")).first()
            if row is None:
                c.execute(insert(S.meta), [{"key": "schema_version", "value": str(S.SCHEMA_VERSION)},
                                           {"key": "created_at", "value": _now()}])
                return
            have = int(row[0])
        if have > S.SCHEMA_VERSION:
            raise SchemaVersionError(f"database schema version {have} is newer than this code ({S.SCHEMA_VERSION}): "
                                     f"update the code, or use another database than {self.url}")
        if have < S.SCHEMA_VERSION:
            self._migrate(have)

    def _migrate(self, have):
        """Bring an older database up to date: add every column the schema has and the database lacks.
        Tables were already created by create_all. Only additive changes, so existing data is untouched."""
        from sqlalchemy import inspect
        insp = inspect(self.engine)
        added = []
        with self.engine.begin() as c:
            for table in S.metadata.sorted_tables:
                present = {col["name"] for col in insp.get_columns(table.name)}
                for col in table.columns:
                    if col.name not in present:
                        if not col.nullable or col.primary_key:
                            raise SchemaVersionError(f"cannot add required column {table.name}.{col.name} "
                                                     f"to an existing database ({self.url})")
                        ctype = col.type.compile(dialect=self.engine.dialect)
                        c.exec_driver_sql(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {ctype}')
                        added.append(f"{table.name}.{col.name}")
            c.execute(update(S.meta).where(S.meta.c.key == "schema_version").values(value=str(S.SCHEMA_VERSION)))
        log.info("database migrated from schema version %d to %d (%s): added %s", have, S.SCHEMA_VERSION,
                 "; ".join(S.MIGRATIONS[v] for v in range(have + 1, S.SCHEMA_VERSION + 1)),
                 ", ".join(added) or "no columns")

    def counts(self):
        with self.engine.connect() as c:
            return {t.name: c.execute(select(func.count()).select_from(t)).scalar_one() for t in S.metadata.sorted_tables}

    # ------------------------------------------------------------------ assets and meters
    def sync_assets(self):
        rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in ids.registry().to_dict("records")]
        for col in ("rating_kva", "switch_kind", "normal", "parent_id"):
            for r in rows:
                r.setdefault(col, None)
        stmt = self._insert(S.assets, upsert_keys=["asset_id"],
                            update_cols=["kind", "name", "parent_id", "rating_kva", "switch_kind", "normal"])
        with self.engine.begin() as c:
            c.execute(stmt, rows)

    def assets(self):
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(select(S.assets)).mappings().all())

    def import_meter_registry(self, path):
        """Load data/processed/meter_registry.csv (written by ml/prepare_data.py). Returns rows inserted."""
        path = Path(path)
        if not path.exists():
            return 0
        reg = pd.read_csv(path, dtype=str)
        rows = [{"meter_id": r["meter_id"], "transformer_asset_id": r["transformer_asset_id"],
                 "deployment_id": r["deployment_id"], "household_id": r["household_id"], "region": r["region"]}
                for _, r in reg.iterrows()]
        with self.engine.begin() as c:
            before = c.execute(select(func.count()).select_from(S.meters)).scalar_one()
            c.execute(self._insert(S.meters, ignore=True), rows)
            return c.execute(select(func.count()).select_from(S.meters)).scalar_one() - before

    def import_consumers(self, path):
        """Load a synthetic city's consumers.csv (one meter per consumer). Idempotent. Returns rows inserted."""
        path = Path(path)
        if not path.exists():
            return 0
        c_ = pd.read_csv(path, dtype=str)
        rows = [{"meter_id": r["meter_id"], "transformer_asset_id": r["transformer_asset_id"], "deployment_id": None,
                 "household_id": None, "region": None, "consumer_id": r["consumer_id"], "consumer_type": r["type"]}
                for _, r in c_.iterrows()]
        with self.engine.begin() as c:
            before = c.execute(select(func.count()).select_from(S.meters)).scalar_one()
            for chunk in _chunks(rows, self.batch_size):
                c.execute(self._insert(S.meters, ignore=True), chunk)
            return c.execute(select(func.count()).select_from(S.meters)).scalar_one() - before

    def meters(self):
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(select(S.meters)).mappings().all())

    # ------------------------------------------------------------------ forecasts and measurements
    def import_forecast_csv(self, path, model_version=None, source=None, horizon_h=24):
        """Load data/forecast.csv: predictions into `forecasts`, measured loads into `measurements`.
        Idempotent: rows already present are skipped. Returns the number of forecast rows inserted."""
        model_version = model_version or self.import_cfg["model_version"]
        source = source or self.import_cfg["measurement_source"]
        fc = pd.read_csv(path, parse_dates=["timestamp"])
        if "asset_id" not in fc:
            fc = ids.add_transformer_ids(fc)
        with self.engine.connect() as c:
            have = c.execute(select(func.count()).select_from(S.forecasts)
                             .where(S.forecasts.c.model_version == model_version)).scalar_one()
        if have == len(fc):
            return 0
        created = _now()
        f_rows = [{"model_version": model_version, "target_time": t.to_pydatetime(), "asset_id": a,
                   "horizon_h": float(horizon_h), "predicted_kw": float(p), "status": "pending", "error_kw": None,
                   "created_at": created}
                  for t, a, p in zip(fc["timestamp"], fc["asset_id"], fc["predicted_kw"])]
        m_rows = [{"time": t.to_pydatetime(), "asset_id": a, "source": source, "kw": float(k), "quality": "good"}
                  for t, a, k in zip(fc["timestamp"], fc["asset_id"], fc["actual_kw"]) if pd.notna(k)]
        with self.engine.begin() as c:
            for chunk in _chunks(f_rows, self.batch_size):
                c.execute(self._insert(S.forecasts, ignore=True), chunk)
        self.write_measurements(m_rows)
        log.info("imported %d forecasts (%s) and %d measurements from %s", len(f_rows) - have, model_version,
                 len(m_rows), Path(path).name)
        return len(f_rows) - have

    def import_intervals_csv(self, path, model_version=None, horizon_h=24, method="conformal"):
        """Load data/forecast_intervals.csv (columns pNN_kw, e.g. p10_kw and p90_kw). Idempotent.
        Returns the number of rows inserted (0 if the file does not exist)."""
        path = Path(path)
        if not path.exists():
            return 0
        model_version = model_version or self.import_cfg["model_version"]
        iv = pd.read_csv(path, parse_dates=["timestamp"])
        if "asset_id" not in iv:
            iv = ids.add_transformer_ids(iv)
        qcols = [c for c in iv.columns if c[0] == "p" and c.endswith("_kw") and c[1:-3].isdigit()]
        rows = [{"model_version": model_version, "target_time": t.to_pydatetime(), "asset_id": a,
                 "horizon_h": float(horizon_h), "quantile": int(c[1:-3]) / 100, "kw": float(v), "method": method}
                for c in qcols for t, a, v in zip(iv["timestamp"], iv["asset_id"], iv[c]) if pd.notna(v)]
        with self.engine.connect() as c:
            have = c.execute(select(func.count()).select_from(S.forecast_quantiles)
                             .where(S.forecast_quantiles.c.model_version == model_version)).scalar_one()
        if have == len(rows):
            return 0
        with self.engine.begin() as c:
            for chunk in _chunks(rows, self.batch_size):
                c.execute(self._insert(S.forecast_quantiles, ignore=True), chunk)
        return len(rows) - have

    def write_forecasts(self, rows, model_version, horizon_h=24):
        """Insert forecasts issued at run time: rows of (target_time, asset_id, predicted_kw). Idempotent."""
        created = _now()
        data = [{"model_version": model_version, "target_time": pd.Timestamp(t).to_pydatetime(), "asset_id": a,
                 "horizon_h": float(horizon_h), "predicted_kw": float(p), "status": "pending", "error_kw": None,
                 "created_at": created} for t, a, p in rows]
        if not data:
            return 0
        with self.engine.begin() as c:
            return c.execute(self._insert(S.forecasts, ignore=True), data).rowcount

    def write_quantiles(self, rows, model_version, method, horizon_h=24):
        """Insert forecast bounds issued at run time: rows of (target_time, asset_id, quantile, kw). Idempotent."""
        data = [{"model_version": model_version, "target_time": pd.Timestamp(t).to_pydatetime(), "asset_id": a,
                 "horizon_h": float(horizon_h), "quantile": float(q), "kw": float(k), "method": method}
                for t, a, q, k in rows]
        if not data:
            return 0
        with self.engine.begin() as c:
            return c.execute(self._insert(S.forecast_quantiles, ignore=True), data).rowcount

    def write_measurements(self, rows):
        """Bulk insert (in batches of storage.yaml batch_size); duplicates (time, asset, source) are skipped."""
        stmt = self._insert(S.measurements, ignore=True)
        with self.engine.begin() as c:
            for chunk in _chunks(list(rows), self.batch_size):
                c.execute(stmt, chunk)

    def upsert_measurements(self, rows):
        """Insert, or replace the kW and quality of, measurements (time, asset, source): estimates that are revised
        as late readings arrive."""
        rows = list(rows)
        if not rows:
            return 0
        stmt = self._insert(S.measurements, upsert_keys=["time", "asset_id", "source"], update_cols=["kw", "quality"])
        with self.engine.begin() as c:
            for chunk in _chunks(rows, self.batch_size):
                c.execute(stmt, chunk)
        return len(rows)

    def write_meter_readings(self, rows):
        """Smart-meter readings as received (duplicates of meter + interval are skipped). Returns the rows given."""
        rows = list(rows)
        if not rows:
            return 0
        stmt = self._insert(S.meter_readings, ignore=True)
        with self.engine.begin() as c:
            for chunk in _chunks(rows, self.batch_size):
                c.execute(stmt, chunk)
        return len(rows)

    def meter_readings(self, sim_from=None, sim_to=None, meter_id=None, flag=None, limit=None):
        r = S.meter_readings
        q = select(r).order_by(r.c.interval_time, r.c.meter_id)
        if sim_from is not None:
            q = q.where(r.c.interval_time >= pd.Timestamp(sim_from).to_pydatetime())
        if sim_to is not None:
            q = q.where(r.c.interval_time <= pd.Timestamp(sim_to).to_pydatetime())
        if meter_id is not None:
            q = q.where(r.c.meter_id == meter_id)
        if flag is not None:
            q = q.where(r.c.flag == flag)
        if limit:
            q = q.limit(limit)
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(q).mappings().all(), columns=[col.name for col in r.columns])

    def measurements(self, source, sim_from=None, sim_to=None):
        m = S.measurements
        q = select(m.c.time, m.c.asset_id, m.c.kw, m.c.quality).where(m.c.source == source).order_by(m.c.time)
        if sim_from is not None:
            q = q.where(m.c.time >= pd.Timestamp(sim_from).to_pydatetime())
        if sim_to is not None:
            q = q.where(m.c.time <= pd.Timestamp(sim_to).to_pydatetime())
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(q).mappings().all(), columns=["time", "asset_id", "kw", "quality"])

    def forecast_frame(self, model_version=None, source=None):
        """timestamp, transformer_id, asset_id, predicted_kw, actual_kw: the same shape as data/forecast.csv."""
        model_version = model_version or self.import_cfg["model_version"]
        source = source or self.import_cfg["measurement_source"]
        f, m, a = S.forecasts, S.measurements, S.assets
        q = (select(f.c.target_time.label("timestamp"), a.c.name.label("transformer_id"), f.c.asset_id,
                    f.c.predicted_kw, m.c.kw.label("actual_kw"))
             .join(a, a.c.asset_id == f.c.asset_id)
             .outerjoin(m, (m.c.time == f.c.target_time) & (m.c.asset_id == f.c.asset_id) & (m.c.source == source))
             .where(f.c.model_version == model_version)
             .order_by(a.c.name, f.c.target_time))
        fq = S.forecast_quantiles
        qq = select(fq.c.target_time, fq.c.asset_id, fq.c.quantile, fq.c.kw).where(fq.c.model_version == model_version)
        with self.engine.connect() as c:
            df = pd.DataFrame(c.execute(q).mappings().all())
            qs = pd.DataFrame(c.execute(qq).mappings().all())
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if not qs.empty:
            qs["col"] = qs["quantile"].map(lambda x: f"p{round(100 * x)}_kw")
            wide = qs.pivot_table(index=["target_time", "asset_id"], columns="col", values="kw").reset_index()
            wide = wide.rename(columns={"target_time": "timestamp"})
            wide["timestamp"] = pd.to_datetime(wide["timestamp"])
            df = df.merge(wide, on=["timestamp", "asset_id"], how="left")
            df.columns.name = None
        return df

    # ------------------------------------------------------------------ forecast evaluation (E10)
    def score_forecasts(self, now, model_version=None, source=None):
        """Score every pending forecast whose target time has passed (<= now) and whose measurement exists:
        error_kw = predicted - actual, status 'scored'. Returns the number scored."""
        model_version = model_version or self.import_cfg["model_version"]
        source = source or self.import_cfg["measurement_source"]
        f, m = S.forecasts, S.measurements
        now = pd.Timestamp(now).to_pydatetime()
        q = (select(f.c.id, (f.c.predicted_kw - m.c.kw).label("err"))
             .join(m, (m.c.time == f.c.target_time) & (m.c.asset_id == f.c.asset_id) & (m.c.source == source))
             .where(f.c.model_version == model_version, f.c.status == "pending", f.c.target_time <= now,
                    m.c.kw.is_not(None)))
        with self.engine.begin() as c:
            todo = c.execute(q).all()
            if todo:
                from sqlalchemy import bindparam
                stmt = update(f).where(f.c.id == bindparam("fid")).values(status="scored", error_kw=bindparam("err"))
                for chunk in _chunks([{"fid": i, "err": float(err)} for i, err in todo], self.batch_size):
                    c.execute(stmt, chunk)
        return len(todo)

    def forecast_performance(self, until=None, model_version=None, source=None):
        """Accuracy of the forecasts scored so far (target time <= until), per transformer and pooled:
        scored, pending, MAE, RMSE, nRMSE (% of mean actual) and bias (+ = over-forecast), in grid kW."""
        model_version = model_version or self.import_cfg["model_version"]
        source = source or self.import_cfg["measurement_source"]
        f, m, a = S.forecasts, S.measurements, S.assets
        q = (select(a.c.name.label("transformer"), f.c.status, f.c.error_kw, m.c.kw.label("actual"), f.c.target_time)
             .join(a, a.c.asset_id == f.c.asset_id)
             .outerjoin(m, (m.c.time == f.c.target_time) & (m.c.asset_id == f.c.asset_id) & (m.c.source == source))
             .where(f.c.model_version == model_version))
        if until is not None:
            q = q.where(f.c.target_time <= pd.Timestamp(until).to_pydatetime())
        with self.engine.connect() as c:
            d = pd.DataFrame(c.execute(q).mappings().all())
        cols = ["transformer", "scored", "pending", "MAE_kW", "RMSE_kW", "nRMSE_%", "bias_kW"]
        if d.empty:
            return pd.DataFrame(columns=cols)
        rows = []
        for tx, g in list(d.groupby("transformer")) + [("ALL", d)]:
            s_ = g[g["status"] == "scored"]
            e_ = s_["error_kw"]
            rmse = float((e_ ** 2).mean() ** 0.5) if len(s_) else None
            rows.append({"transformer": tx, "scored": len(s_), "pending": int((g["status"] == "pending").sum()),
                         "MAE_kW": float(e_.abs().mean()) if len(s_) else None, "RMSE_kW": rmse,
                         "nRMSE_%": 100 * rmse / float(s_["actual"].mean()) if len(s_) else None,
                         "bias_kW": float(e_.mean()) if len(s_) else None})
        return pd.DataFrame(rows, columns=cols)

    def forecast_errors(self, until=None, model_version=None, source=None):
        """Every forecast with its actual value (target time <= until): timestamp, transformer, predicted_kw,
        actual_kw, error_kw (forecast - actual). For accuracy over time (dashboard page D)."""
        model_version = model_version or self.import_cfg["model_version"]
        source = source or self.import_cfg["measurement_source"]
        f, m, a = S.forecasts, S.measurements, S.assets
        q = (select(f.c.target_time.label("timestamp"), a.c.name.label("transformer"), f.c.predicted_kw,
                    m.c.kw.label("actual_kw"))
             .join(a, a.c.asset_id == f.c.asset_id)
             .join(m, (m.c.time == f.c.target_time) & (m.c.asset_id == f.c.asset_id) & (m.c.source == source))
             .where(f.c.model_version == model_version, m.c.kw.is_not(None)))
        if until is not None:
            q = q.where(f.c.target_time <= pd.Timestamp(until).to_pydatetime())
        with self.engine.connect() as c:
            d = pd.DataFrame(c.execute(q.order_by(f.c.target_time)).mappings().all())
        if d.empty:
            return pd.DataFrame(columns=["timestamp", "transformer", "predicted_kw", "actual_kw", "error_kw"])
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d["error_kw"] = d["predicted_kw"] - d["actual_kw"]
        return d

    def forecast_count_after(self, t, model_version=None):
        """Forecasts whose target time is still to come (after t): they cannot be scored yet."""
        model_version = model_version or self.import_cfg["model_version"]
        f = S.forecasts
        with self.engine.connect() as c:
            return c.execute(select(func.count()).select_from(f).where(
                f.c.model_version == model_version, f.c.target_time > pd.Timestamp(t).to_pydatetime())).scalar_one()

    def model_versions(self):
        with self.engine.connect() as c:
            return [r[0] for r in c.execute(select(S.forecasts.c.model_version).distinct())]

    # ------------------------------------------------------------------ recommendations
    def save_recommendation(self, controller, rec):
        """Insert or replace a recommendation (dict from Recommendation.to_dict()) and its candidates."""
        row = {k: v for k, v in rec.items() if k in S.recommendations.c and k != "controller"}
        row["controller"] = controller
        row["seq"] = int("".join(ch for ch in rec["rec_id"] if ch.isdigit()) or 0)
        cands = [{"rec_id": rec["rec_id"], "cand_id": c["id"], "round": c.get("round"), "kind": c.get("kind"),
                  "actions": c.get("actions"), "label": c.get("label"), "status": c.get("status"),
                  "reason": c.get("reason"), "feasible": c.get("feasible"), "max_loading": c.get("max_loading"),
                  "max_loading_tx": c.get("max_loading_tx"), "min_voltage_pu": c.get("min_voltage_pu"),
                  "max_line_loading": c.get("max_line_loading"), "losses_kw": c.get("losses_kw")}
                 for c in rec.get("candidates", [])]
        with self.engine.begin() as c:
            c.execute(delete(S.candidates).where(S.candidates.c.rec_id == rec["rec_id"]))
            c.execute(delete(S.recommendations).where(S.recommendations.c.rec_id == rec["rec_id"]))
            c.execute(insert(S.recommendations), [row])
            if cands:
                c.execute(insert(S.candidates), cands)

    def next_rec_seq(self):
        """The next recommendation number, unique across every controller on this database (the replay sandbox
        and the simulation service both number recommendations REC-nnnn). The UPDATE comes first so that two
        processes asking at once are serialised by the database."""
        from sqlalchemy import cast, Integer as Int
        m = S.meta
        for _ in range(5):
            try:
                with self.engine.begin() as c:
                    done = c.execute(update(m).where(m.c.key == "rec_seq")
                                     .values(value=cast(cast(m.c.value, Int) + 1, S.meta.c.value.type))).rowcount
                    if not done:
                        top = c.execute(select(func.max(S.recommendations.c.seq))).scalar_one() or 0
                        c.execute(insert(m).values(key="rec_seq", value=str(top + 1)))
                    return int(c.execute(select(m.c.value).where(m.c.key == "rec_seq")).scalar_one())
            except Exception as ex:                                     # two first allocations at once: retry
                if "UNIQUE" not in str(ex).upper() and "duplicate" not in str(ex).lower():
                    raise
        raise RuntimeError("could not allocate a recommendation number")

    def update_recommendation_status(self, rec_id, status):
        with self.engine.begin() as c:
            c.execute(update(S.recommendations).where(S.recommendations.c.rec_id == rec_id).values(status=status))

    def load_recommendation(self, rec_id):
        with self.engine.connect() as c:
            row = c.execute(select(S.recommendations).where(S.recommendations.c.rec_id == rec_id)).mappings().first()
            if row is None:
                return None
            cands = c.execute(select(S.candidates).where(S.candidates.c.rec_id == rec_id)
                              .order_by(S.candidates.c.id)).mappings().all()
        out = dict(row)
        out.pop("controller")
        out.pop("seq")
        out["candidates"] = [{"id": r["cand_id"], "round": r["round"], "kind": r["kind"], "actions": r["actions"],
                              "label": r["label"], "status": r["status"], "reason": r["reason"],
                              "feasible": r["feasible"], "max_loading": r["max_loading"],
                              "max_loading_tx": r["max_loading_tx"], "min_voltage_pu": r["min_voltage_pu"],
                              "max_line_loading": r["max_line_loading"], "losses_kw": r["losses_kw"]} for r in cands]
        return out

    def _latest_rec_id(self, controller, status=None):
        r = S.recommendations
        q = select(r.c.rec_id).where(r.c.controller == controller)
        if status:
            q = q.where(r.c.status == status)
        q = q.order_by(r.c.seq.desc()).limit(1)
        with self.engine.connect() as c:
            row = c.execute(q).first()
        return row[0] if row else None

    def pending_recommendation(self, controller):
        rec_id = self._latest_rec_id(controller, "pending")
        return self.load_recommendation(rec_id) if rec_id else None

    def latest_recommendation(self, controller):
        rec_id = self._latest_rec_id(controller)
        return self.load_recommendation(rec_id) if rec_id else None

    def recommendations(self, controller=None, kinds=None, statuses=None, limit=200):
        r = S.recommendations
        q = select(r.c.rec_id, r.c.created_at, r.c.for_time, r.c.kind, r.c.status, r.c.summary,
                   r.c.losses_before_kw, r.c.losses_after_kw, r.c.compute_s)
        if controller:
            q = q.where(r.c.controller == controller)
        if kinds:
            q = q.where(r.c.kind.in_(kinds))
        if statuses:
            q = q.where(r.c.status.in_(statuses))
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(q.order_by(r.c.seq.desc()).limit(limit)).mappings().all())

    def candidates(self, rec_id):
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(select(S.candidates).where(S.candidates.c.rec_id == rec_id)
                                          .order_by(S.candidates.c.id)).mappings().all())

    # ------------------------------------------------------------------ events, actions, alerts
    def add_event(self, controller, ev, assets=(), plan=None, how=None):
        """Store one event, the assets it concerns and (if it switched anything) its switch operations.
        Returns the event's sequence number."""
        row = {"controller": controller, "wall_time": ev.get("wall_time"), "sim_time": ev.get("sim_time"),
               "type": ev["type"], "severity": ev.get("severity", "info"), "message": ev.get("message"),
               "rec_id": ev.get("rec_id"), "mode": ev.get("mode"), "configuration": ev.get("configuration"),
               "data": {k: v for k, v in ev.items() if k not in
                        ("seq", "wall_time", "sim_time", "type", "severity", "message", "rec_id", "mode",
                         "configuration", "assets")} or None}
        with self.engine.begin() as c:
            seq = c.execute(insert(S.events).values(**row)).inserted_primary_key[0]
            if assets:
                c.execute(insert(S.event_assets), [{"seq": seq, "asset_id": a} for a in sorted(set(assets))])
            if plan:
                c.execute(insert(S.actions), [{"event_seq": seq, "controller": controller, "wall_time": row["wall_time"],
                                               "sim_time": row["sim_time"], "rec_id": row["rec_id"], "how": how,
                                               "step": p["step"], "switch": p["switch"], "switch_id": p["switch_id"],
                                               "operation": p["operation"], "action_key": p["action"]} for p in plan])
        return seq

    def events(self, controller=None, sim_from=None, sim_to=None, types=None, severities=None, asset_id=None,
               limit=500, ascending=False):
        """The audit log, newest first, with optional filters. sim_from / sim_to compare the simulated time as
        text ("2020-02-29" covers that whole day when used as both bounds)."""
        e = S.events
        q = select(e)
        if controller:
            q = q.where(e.c.controller == controller)
        if sim_from:
            q = q.where(e.c.sim_time >= str(sim_from))
        if sim_to:
            q = q.where(e.c.sim_time <= str(sim_to) + "￿")
        if types:
            q = q.where(e.c.type.in_(types))
        if severities:
            q = q.where(e.c.severity.in_(severities))
        if asset_id:
            q = q.where(e.c.seq.in_(select(S.event_assets.c.seq).where(S.event_assets.c.asset_id == asset_id)))
        q = q.order_by(e.c.seq if ascending else e.c.seq.desc())
        if limit:
            q = q.limit(limit)
        with self.engine.connect() as c:
            rows = [dict(r) for r in c.execute(q).mappings()]
            if rows:
                link = c.execute(select(S.event_assets).where(S.event_assets.c.seq.in_([r["seq"] for r in rows])))
                by_seq = {}
                for s, a in link:
                    by_seq.setdefault(s, []).append(a)
                for r in rows:
                    r["assets"] = sorted(by_seq.get(r["seq"], []))
        return rows

    def actions(self, controller=None, limit=500):
        """Executed switch operations, newest first (limit=None: all of them)."""
        a = S.actions
        q = select(a)
        if controller:
            q = q.where(a.c.controller == controller)
        q = q.order_by(a.c.id.desc())
        if limit:
            q = q.limit(limit)
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(q).mappings().all())

    def record_alerts(self, sim_time, rows, context=""):
        """Store alerts observed at a simulated time. The same alert (time, asset, kind, context) is kept once."""
        if not rows:
            return
        now = _now()
        data = [{"sim_time": str(sim_time), "asset": r["asset"], "kind": r["kind"], "level": r["level"],
                 "value": r.get("value"), "text": r.get("text"), "context": context, "first_seen": now} for r in rows]
        with self.engine.begin() as c:
            c.execute(self._insert(S.alerts, ignore=True), data)

    def alerts(self, sim_from=None, sim_to=None, levels=None, asset=None, limit=500, context_prefix=None):
        """context_prefix: e.g. "service " for the simulation service's alerts only."""
        a = S.alerts
        q = select(a)
        if context_prefix:
            q = q.where(a.c.context.like(f"{context_prefix}%"))
        if sim_from:
            q = q.where(a.c.sim_time >= str(sim_from))
        if sim_to:
            q = q.where(a.c.sim_time <= str(sim_to) + "￿")
        if levels:
            q = q.where(a.c.level.in_(levels))
        if asset:
            q = q.where(a.c.asset == asset)
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(q.order_by(a.c.sim_time.desc(), a.c.id).limit(limit)).mappings().all())

    # ------------------------------------------------------------------ controller state
    def save_state(self, controller, mode, active, estopped, undo, rec_seq):
        row = {"controller": controller, "mode": mode, "active": list(active), "estopped": bool(estopped),
               "undo": [list(u) for u in undo], "rec_seq": int(rec_seq), "updated_at": _now()}
        stmt = self._insert(S.controller_state, upsert_keys=["controller"],
                            update_cols=["mode", "active", "estopped", "undo", "rec_seq", "updated_at"])
        with self.engine.begin() as c:
            c.execute(stmt, [row])

    def load_state(self, controller):
        with self.engine.connect() as c:
            row = c.execute(select(S.controller_state)
                            .where(S.controller_state.c.controller == controller)).mappings().first()
        return dict(row) if row else None

    # ------------------------------------------------------------------ simulation service (Phase 5)
    def save_sim_state(self, service, **fields):
        """Create or update the service's row (clock, scenario, status, heartbeat ...)."""
        cols = [c.name for c in S.sim_state.columns if c.name != "service"]
        unknown = set(fields) - set(cols)
        if unknown:
            raise ValueError(f"unknown sim_state fields: {sorted(unknown)}")
        current = self.load_sim_state(service) or {}
        row = {"service": service, **{c: current.get(c) for c in cols}, **fields}
        stmt = self._insert(S.sim_state, upsert_keys=["service"], update_cols=cols)
        with self.engine.begin() as c:
            c.execute(stmt, [row])

    def load_sim_state(self, service):
        with self.engine.connect() as c:
            row = c.execute(select(S.sim_state).where(S.sim_state.c.service == service)).mappings().first()
        return dict(row) if row else None

    def record_step(self, row):
        """Store one completed simulation step. Idempotent: the same (service, run, time) is kept once."""
        row = {**row, "sim_time": pd.Timestamp(row["sim_time"]).to_pydatetime()}
        with self.engine.begin() as c:
            return c.execute(self._insert(S.sim_steps, ignore=True), [row]).rowcount

    def last_step_time(self, service, run_id):
        with self.engine.connect() as c:
            return c.execute(select(func.max(S.sim_steps.c.sim_time))
                             .where(S.sim_steps.c.service == service, S.sim_steps.c.run_id == run_id)).scalar_one()

    def steps(self, service, run_id=None, sim_from=None, sim_to=None, limit=None, newest_first=False):
        s = S.sim_steps
        q = select(s).where(s.c.service == service)
        if run_id is not None:
            q = q.where(s.c.run_id == run_id)
        if sim_from is not None:
            q = q.where(s.c.sim_time >= pd.Timestamp(sim_from).to_pydatetime())
        if sim_to is not None:
            q = q.where(s.c.sim_time <= pd.Timestamp(sim_to).to_pydatetime())
        q = q.order_by(s.c.sim_time.desc() if newest_first else s.c.sim_time)
        if limit:
            q = q.limit(limit)
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(q).mappings().all())

    def send_command(self, service, command, args=None, source="dashboard"):
        """Queue a command for the service. Returns its id."""
        with self.engine.begin() as c:
            r = c.execute(insert(S.commands).values(service=service, created_at=_now(), source=source,
                                                    command=command, args=args or {}, status="pending"))
            return r.inserted_primary_key[0]

    def pending_commands(self, service):
        q = (select(S.commands).where(S.commands.c.service == service, S.commands.c.status == "pending")
             .order_by(S.commands.c.id))
        with self.engine.connect() as c:
            return [dict(r) for r in c.execute(q).mappings().all()]

    def finish_command(self, cmd_id, status, result=""):
        with self.engine.begin() as c:
            c.execute(update(S.commands).where(S.commands.c.id == cmd_id)
                      .values(status=status, result=result, done_at=_now()))

    def command(self, cmd_id):
        with self.engine.connect() as c:
            row = c.execute(select(S.commands).where(S.commands.c.id == cmd_id)).mappings().first()
        return dict(row) if row else None

    def commands(self, service, limit=50):
        q = select(S.commands).where(S.commands.c.service == service).order_by(S.commands.c.id.desc()).limit(limit)
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(q).mappings().all())

    # ------------------------------------------------------------------ model registry (used from Phase 4)
    def register_model(self, version, kind, path, features=None, params=None, metrics=None, notes=None):
        stmt = self._insert(S.model_registry, upsert_keys=["version"],
                            update_cols=["kind", "path", "features", "params", "metrics", "notes"])
        with self.engine.begin() as c:
            c.execute(stmt, [{"version": version, "created_at": _now(), "kind": kind, "path": str(path),
                              "features": features, "params": params, "metrics": metrics, "notes": notes}])

    def models(self):
        with self.engine.connect() as c:
            return pd.DataFrame(c.execute(select(S.model_registry)).mappings().all())

    # ------------------------------------------------------------------ retention
    def prune(self, now_sim):
        """Delete measurements, alerts and forecasts older than the retention window (storage.yaml) before `now_sim`.
        Returns {table: rows deleted}. A window of 0 days keeps everything."""
        now_sim = pd.Timestamp(now_sim).to_pydatetime()
        out = {}
        with self.engine.begin() as c:
            days = self.retention["measurements_days"]
            if days:
                cut = now_sim - dt.timedelta(days=days)
                out["measurements"] = c.execute(delete(S.measurements).where(S.measurements.c.time < cut)).rowcount
            days = self.retention.get("meter_readings_days", 0)
            if days:
                cut = now_sim - dt.timedelta(days=days)
                out["meter_readings"] = c.execute(delete(S.meter_readings)
                                                  .where(S.meter_readings.c.interval_time < cut)).rowcount
            days = self.retention["alerts_days"]
            if days:
                cut = (now_sim - dt.timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
                out["alerts"] = c.execute(delete(S.alerts).where(S.alerts.c.sim_time < cut)).rowcount
            days = self.retention.get("forecasts_days", 0)
            if days:
                cut = now_sim - dt.timedelta(days=days)
                out["forecast_quantiles"] = c.execute(delete(S.forecast_quantiles)
                                                      .where(S.forecast_quantiles.c.target_time < cut)).rowcount
                out["forecasts"] = c.execute(delete(S.forecasts).where(S.forecasts.c.target_time < cut)).rowcount
        return out


def _sqlite_pragmas(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()
