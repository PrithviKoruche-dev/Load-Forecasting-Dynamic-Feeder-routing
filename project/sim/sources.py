"""Where the closed loop gets its loads from.

A demand source answers "what did the meters measure at time t?" and a forecast source "what is the
24 h-ahead forecast for time t (and its upper bound)?". Phase 5 has the classic source: the real eMARC test
period (six transformers, hourly) from the database. The synthetic city of Phase 7 plugs in as another
demand source with the same interface, so the orchestrator, database and dashboard do not change.

Forecast sources:
    PrecomputedForecasts  the committed 24 h-ahead forecasts (data/forecast.csv, imported into the database)
                          and their P10/P90 band: reproducible on any machine
    LiveForecasts         ml/forecaster.py forecast_at(t) at run time, from data/processed/clusters.parquet.
                          At step t it issues the forecast for t + 24 h (stored as pending, scored when its hour
                          passes); the upper bound still comes from the committed band
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import ids  # noqa: E402

HORIZON = pd.Timedelta(hours=24)


def _qcols(df):
    return sorted((c for c in df.columns if c[0] == "p" and c.endswith("_kw") and c[1:-3].isdigit()),
                  key=lambda c: int(c[1:-3]))


class ClassicSource:
    """Measured loads of the eMARC test period (scaled to grid kW), from the database."""
    name = "classic"
    resolution = pd.Timedelta(hours=1)

    def __init__(self, store, frame=None):
        self.frame = store.forecast_frame() if frame is None else frame
        if self.frame.empty:
            raise RuntimeError("the database holds no forecasts or measurements: import data/forecast.csv first")
        self.actual = self.frame.pivot(index="timestamp", columns="transformer_id", values="actual_kw")
        self.transformers = list(self.actual.columns)

    def times(self):
        return self.actual.index

    def measured(self, t):
        """{transformer: kW} at t, or None if any reading is missing."""
        t = pd.Timestamp(t)
        if t not in self.actual.index or self.actual.loc[t].isna().any():
            return None
        return {tx: float(v) for tx, v in self.actual.loc[t].items()}


class PrecomputedForecasts:
    name = "precomputed"

    def __init__(self, store, frame=None):
        frame = store.forecast_frame() if frame is None else frame
        self.model_version = store.import_cfg["model_version"]
        self.point_ = frame.pivot(index="timestamp", columns="transformer_id", values="predicted_kw")
        q = _qcols(frame)
        self.upper_col = q[-1] if len(q) >= 2 else None
        self.upper_ = frame.pivot(index="timestamp", columns="transformer_id", values=self.upper_col) if self.upper_col else None

    @staticmethod
    def _row(table, t):
        t = pd.Timestamp(t)
        if table is None or t not in table.index or table.loc[t].isna().any():
            return None
        return {tx: float(v) for tx, v in table.loc[t].items()}

    def point(self, t):
        return self._row(self.point_, t)

    def upper(self, t):
        return self._row(self.upper_, t)

    runtime = False                    # every forecast was imported before the run: nothing to clear for a new one

    def issue(self, now):
        """Precomputed forecasts were all issued in advance: nothing to do."""
        return 0


class LiveForecasts(PrecomputedForecasts):
    """forecast_at(t) from the committed model at run time."""
    name = "live"
    runtime = True                     # issued as the run goes: a new run clears them from its start

    def __init__(self, store, frame=None, clusters=None, forecaster=None):
        super().__init__(store, frame)
        sys.path.insert(0, str(ROOT / "ml"))
        from forecaster import Forecaster
        path = ROOT / "data" / "processed" / "clusters.parquet"
        if clusters is None:
            if not path.exists():
                raise FileNotFoundError(f"live forecasting needs {path} (the eMARC clusters); "
                                        "use forecast_source: precomputed without it")
            clusters = pd.read_parquet(path)
        self.history = clusters
        self.f = forecaster or Forecaster()
        self.store = store
        self.model_version = f"{self.f.version}-live"
        self._cache = {}

    def _compute(self, t):
        t = pd.Timestamp(t)
        if t not in self._cache:
            out = self.f.forecast_at(t, self.history)
            self._cache[t] = {r.transformer_id: float(r.predicted_kw) for r in out.itertuples()}
            self.store.write_forecasts([(t, ids.transformer_id(tx), kw) for tx, kw in self._cache[t].items()],
                                       self.model_version)
        return self._cache[t]

    def point(self, t):
        """The forecast for t (issued at t - 24 h: by an earlier step, or now if the loop started later)."""
        try:
            return self._compute(t)
        except ValueError:                      # no history row for t (outside the data)
            return None

    def issue(self, now):
        """At time `now` the forecast for now + 24 h can be made; store it as pending."""
        try:
            self._compute(pd.Timestamp(now) + HORIZON)
            return 1
        except ValueError:
            return 0


class CitySource:
    """Loads of a synthetic city (Phase 7): cities/<name>/, 15-minute steps over a year. The scenario's events
    (heatwave, EV charging, ...) change the demand itself.

    Two views of the same loads (Phase 8):
        measured(t)      the true load of every transformer in interval t: what the network carries, used for the
                         power flow and to judge the outcome
        observe(t, sub)  what the operator knows at the end of interval t: the transformer meter if it reports,
                         else the smart meters' estimate of the interval before, else `sub` (the forecast), else
                         the last value seen; written to the database as it arrives
    With city.yaml `meters` disabled (or meters=False) the operator sees the true loads (perfect measurement)."""
    name = "city"
    ESTIMATE_SOURCE = "ami-estimate"

    @property
    def runtime_sources(self):
        """Measurement sources written as the run goes (cleared from a new run's start, with the readings)."""
        return [self.source, self.ESTIMATE_SOURCE]

    def __init__(self, store, city, scenario=None, meters=None):
        sys.path.insert(0, str(ROOT / "city"))
        from demand import CityDemand
        from meters import HeadEnd, MeterModel, settings
        from core import config
        self.store, self.city = store, city
        self.demand = CityDemand(city)
        self.resolution = self.demand.times[1] - self.demand.times[0]
        self.transformers = list(config.grid()["transformers"])
        missing = set(self.transformers) - set(self.demand.transformers)
        if missing:
            raise RuntimeError(f"cities/{city}/consumers.csv has no consumers for {sorted(missing)}: generate again")
        on = settings(self.demand.spec)["enabled"] if meters is None else bool(meters)
        self.meters = MeterModel(self.demand) if on else None
        self.head = HeadEnd(self.meters) if on else None
        self.store_readings = settings(self.demand.spec)["store_readings"]
        self._col = [self.demand.transformers.index(tx) for tx in self.transformers] if on else None
        self.events = ()
        self.source = store.import_cfg["measurement_source"]
        self.last_seen = {}
        self.set_events(scenario.events if scenario is not None else ())

    def set_events(self, events):
        self.events = tuple(events)
        self.table = self.demand.matrix(self.events)[self.transformers]
        self._history = None
        if self.head is not None:
            self.head.position = None                    # the readings depend on the events: rebuild on next step

    def times(self):
        from core import config
        c = config.simulation()["clock"]
        t = self.table.index
        lo = t.min() if c["start"] == "test_start" else pd.Timestamp(c["start"])
        hi = t.max() if c["end"] == "test_end" else pd.Timestamp(c["end"])
        return t[(t >= lo) & (t <= hi)]

    def measured(self, t):
        """True load of every transformer in interval t (kW)."""
        t = pd.Timestamp(t)
        if t not in self.table.index:
            return None
        return {tx: float(v) for tx, v in self.table.loc[t].items()}

    @property
    def history(self):
        """The loads the operator has recorded over the year: the transformer meters, a gap (sensor out) filled
        with the reading of the same time the day before. Without meters: the true loads."""
        if self.meters is None:
            return self.table
        if self._history is None:
            from meters import observed_history
            self._history = observed_history(self.meters, self.demand.matrix(self.events))[self.transformers]
        return self._history

    def observe(self, t, substitute=None):
        """The end of interval t: the readings that arrive now. Writes them to the database and returns
        {"loads": {tx: kW}, "quality": {tx: measured | ami-estimate | forecast | last-value}, "stats": {...}}."""
        t = pd.Timestamp(t)
        true = self.measured(t)
        if true is None:
            return None
        when = t.to_pydatetime()
        if self.meters is None:                         # perfect measurement
            self.store.write_measurements([{"time": when, "asset_id": ids.transformer_id(tx), "source": self.source,
                                            "kw": round(kw, 3), "quality": "good"} for tx, kw in true.items()])
            return {"loads": true, "quality": {tx: "measured" for tx in true}, "stats": None}
        i = self.demand.times.get_loc(t)
        head = self.head
        if head.position is None or head.position != i - 1:
            head.rebuild(i - 1, self.events)
        delivered, estimates = head.step(i, self.events)
        full = self.demand.matrix(self.events).iloc[i].to_numpy()
        dt = self.meters.transformer_readings(i, full)[self._col]
        prev = estimates.get(i - 1) or (head.estimate(i - 1) if i - 1 in head.agg else {})
        loads, quality = {}, {}
        for k, tx in enumerate(self.transformers):
            if not np.isnan(dt[k]):
                loads[tx], quality[tx] = float(dt[k]), "measured"
            elif tx in prev and prev[tx][1] in ("good", "estimated"):
                loads[tx], quality[tx] = prev[tx][0], "ami-estimate"
            elif substitute and substitute.get(tx) is not None:
                loads[tx], quality[tx] = float(substitute[tx]), "forecast"
            elif tx in self.last_seen:
                loads[tx], quality[tx] = self.last_seen[tx], "last-value"
            else:                                        # nothing known yet (only before the first reading)
                loads[tx], quality[tx] = 0.0, "missing"
        self.last_seen.update(loads)
        # database: transformer meters, smart-meter estimates (revised as late readings arrive), raw readings
        aid = {tx: ids.transformer_id(tx) for tx in self.transformers}
        self.store.write_measurements([{"time": when, "asset_id": aid[tx], "source": self.source,
                                        "kw": None if np.isnan(dt[k]) else round(float(dt[k]), 3),
                                        "quality": "missing" if np.isnan(dt[k]) else "good"}
                                       for k, tx in enumerate(self.transformers)])
        rows = []
        for j, est in estimates.items():
            tj = self.demand.times[j].to_pydatetime()
            rows += [{"time": tj, "asset_id": aid[tx], "source": self.ESTIMATE_SOURCE,
                      "kw": None if kw is None else round(kw, 3), "quality": q}
                     for tx, (kw, q, _) in est.items() if tx in aid]
        self.store.upsert_measurements(rows)
        n_rows = self.store.write_meter_readings(self._reading_rows(delivered))
        self.last_stats = stats = self._stats(i, delivered, prev, dt, n_rows)
        return {"loads": loads, "quality": quality, "stats": stats}

    def _reading_rows(self, delivered):
        """Readings to store (city.yaml meters.store_readings): all, none, or the exceptions: readings that came
        later than one interval after their own, or from a meter flagged as stuck."""
        if self.store_readings == "none":
            return []
        ids_, times = self.meters.meter_ids, self.demand.times
        step_ns = self.resolution.value
        rows = []
        for j, idx, kw, arrive, suspect in delivered:
            end = (times[j] + self.resolution).value
            tj = times[j].to_pydatetime()
            if self.store_readings == "exceptions":
                keep = (arrive - end > step_ns) | suspect
                idx, kw, arrive, suspect = idx[keep], kw[keep], arrive[keep], suspect[keep]
            for m, v, a, sus in zip(idx, kw, arrive, suspect):
                rows.append({"meter_id": ids_[m], "interval_time": tj, "kw": float(v),
                             "flag": "suspect" if sus else "ok",
                             "received_at": pd.Timestamp(int(a) // 1000 * 1000).to_pydatetime(), "delay_s": round((a - end) / 1e9, 1)})
        return rows

    def _stats(self, i, delivered, prev, dt, n_rows):
        head, m = self.head, self.meters
        now = (self.demand.times[i] + self.resolution).value
        late = sum(len(x[1]) for x in delivered if x[0] < i - 1)
        down = sum(any(a <= now < b for a, b in w) for w in m.outages)
        q = {}
        for _, qq, _ in prev.values():
            q[qq] = q.get(qq, 0) + 1
        return {"dt_missing": int(np.isnan(dt).sum()), "readings": int(sum(len(x[1]) for x in delivered)),
                "late_readings": int(late), "awaited": int(sum((a >= 0).sum() for _, a in head.pending.values())),
                "suspect_meters": int(head.suspect.sum()), "concentrators_down": int(down),
                "ami_quality": q, "stored": n_rows}


class PersistenceForecasts:
    """24 h-ahead forecast for the synthetic city: the load recorded at the same time the day before (the
    transformer meters, Phase 8; gaps filled from the day before that), with a P10-P90 band
    by rolling conformal calibration (config forecasting.yaml intervals: the ratio actual / forecast over the
    last window_days, per transformer and block of the day, using only loads known at issue time t - 24 h).
    Phase 9 adds a short-horizon model trained on the simulator's history (ShortHorizonForecasts)."""
    name = "persistence"
    runtime = True                     # issued as the run goes: a new run clears them from its start
    VERSION = "persistence-24h"

    def __init__(self, store, demand):
        from core import config
        self.store, self.demand = store, demand
        self.model_version = self.VERSION
        self.cfg = config.forecasting()["intervals"]
        self._key = None

    def _refresh(self):
        if self._key == self.demand.events:
            return
        a = getattr(self.demand, "history", self.demand.table)    # what the operator has recorded
        lag = int(HORIZON / self.demand.resolution)
        self.point_ = a.shift(lag)
        ratio = (a / self.point_).replace([float("inf")], float("nan"))
        hours = 24 // self.cfg["hour_blocks"]
        block = a.index.hour // hours
        lo_q, hi_q = self.cfg["lower_quantile"], self.cfg["upper_quantile"]
        win, min_n = f"{self.cfg['window_days']}D", 7 * (hours * int(pd.Timedelta(hours=1) / self.demand.resolution))
        lo, hi = pd.DataFrame(index=a.index, columns=a.columns, dtype=float), pd.DataFrame(index=a.index, columns=a.columns, dtype=float)
        for b in sorted(set(block)):
            r = ratio[block == b]
            ql = r.rolling(win, min_periods=min_n).quantile(lo_q)
            qh = r.rolling(win, min_periods=min_n).quantile(hi_q)
            targets = a.index[block == b]
            issued = targets - HORIZON                           # known at issue time: ratios up to t - 24 h
            pos = ql.index.searchsorted(issued, side="right") - 1
            ok = pos >= 0
            lo.loc[targets[ok]] = ql.iloc[pos[ok]].to_numpy()
            hi.loc[targets[ok]] = qh.iloc[pos[ok]].to_numpy()
        self.lower_ = self.point_ * lo
        self.upper_ = self.point_ * hi
        self._key = self.demand.events

    @staticmethod
    def _row(table, t):
        t = pd.Timestamp(t)
        if t not in table.index or table.loc[t].isna().any():
            return None
        return {tx: float(v) for tx, v in table.loc[t].items()}

    def point(self, t):
        self._refresh()
        return self._row(self.point_, t)

    def upper(self, t):
        self._refresh()
        return self._row(self.upper_, t)

    def issue(self, now):
        """At time `now` the forecast for now + 24 h is known (it is the load measured now): store it, with its band."""
        self._refresh()
        target = pd.Timestamp(now) + HORIZON
        p = self._row(self.point_, target)
        if p is None:
            return 0
        aid = {tx: ids.transformer_id(tx) for tx in p}
        self.store.write_forecasts([(target, aid[tx], kw) for tx, kw in p.items()], self.model_version)
        rows = []
        for q, table in ((self.cfg["lower_quantile"], self.lower_), (self.cfg["upper_quantile"], self.upper_)):
            b = self._row(table, target)
            if b:
                rows += [(target, aid[tx], q, kw) for tx, kw in b.items()]
        self.store.write_quantiles(rows, self.model_version, "rolling conformal")
        return 1


class ShortHorizonForecasts:
    """The short-horizon forecast of a synthetic city (Phase 9, ml/short_horizon.py): every transformer's load
    one step ahead, made at the end of each step from what the transformer meters recorded up to then, the
    weather forecast and the calendar, with a P10-P90 band. point(t) is the forecast for t made one step
    before it; the switching decision at the end of step t uses point(t + 1)."""
    name = "short"
    runtime = True                     # issued as the run goes: a new run clears them from its start

    def __init__(self, store, demand):
        sys.path.insert(0, str(ROOT / "ml"))
        from short_horizon import ShortModel
        self.store, self.demand = store, demand
        self.model = ShortModel(demand.city)
        self.model_version = self.model.version
        self.horizon = self.model.horizon * demand.resolution
        self._key = None

    def _refresh(self):
        if self._key == self.demand.events:
            return
        hist = getattr(self.demand, "history", self.demand.table)
        self.point_, self.lower_, self.upper_ = self.model.tables(self.demand.demand, hist, self.demand.events)
        self._key = self.demand.events

    _row = staticmethod(PersistenceForecasts._row)

    def point(self, t):
        self._refresh()
        return self._row(self.point_, t)

    def upper(self, t):
        self._refresh()
        return self._row(self.upper_, t)

    def lower(self, t):
        self._refresh()
        return self._row(self.lower_, t)

    def issue(self, now):
        """At the end of step `now` the forecast for `now + horizon` is made: store it, with its band."""
        self._refresh()
        target = pd.Timestamp(now) + self.horizon
        p = self._row(self.point_, target)
        if p is None:
            return 0
        h = self.horizon / pd.Timedelta(hours=1)
        aid = {tx: ids.transformer_id(tx) for tx in p}
        self.store.write_forecasts([(target, aid[tx], kw) for tx, kw in p.items()], self.model_version, horizon_h=h)
        q = self.model.meta["band_quantiles"]
        rows = []
        for qq, table in ((q[0], self.lower_), (q[1], self.upper_)):
            b = self._row(table, target)
            if b:
                rows += [(target, aid[tx], qq, kw) for tx, kw in b.items()]
        self.store.write_quantiles(rows, self.model_version, "conformal (calibration month)", horizon_h=h)
        return 1


def forecast_source(kind, store, frame=None, demand=None, **kw):
    if kind == "precomputed":
        return PrecomputedForecasts(store, frame)
    if kind == "live":
        return LiveForecasts(store, frame, **kw)
    if kind == "persistence":
        return PersistenceForecasts(store, demand)
    if kind == "short":
        return ShortHorizonForecasts(store, demand)
    raise ValueError(f"unknown forecast source {kind!r}")
