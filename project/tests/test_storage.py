"""Database layer: schema, imports, queries, batching, retention and controller persistence."""
import datetime as dt

import pandas as pd
import pytest
from sqlalchemy import inspect

import engine as e
import run_results as rr
from core import config, scenarios
from decisions import Controller, DecisionError
from storage import Store
from storage import schema as S

T = "2020-02-29 21:00"


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'test.db'}")


@pytest.fixture(scope="module")
def loads():
    pred = rr.load_predictions()
    sc = {s.name.split()[0]: s for s in scenarios.benchmark()}

    def f(name, ts):
        ts = pd.Timestamp(ts)
        return {tx: float(sc[name].adjust(tx, ts, pred.loc[ts, tx])) for tx in e.TX}
    return f


# ------------------------------------------------------------------ schema and configuration
def test_schema(store):
    tables = set(inspect(store.engine).get_table_names())
    assert tables == {t.name for t in S.metadata.sorted_tables}
    assert {"forecasts", "measurements", "recommendations", "candidates", "events", "actions", "alerts",
            "controller_state", "model_registry", "assets", "meters"} <= tables


def test_assets_are_synced_from_config(store):
    a = store.assets().set_index("asset_id")
    assert len(a) == 24 and a.loc["DT-004", "name"] == "T4" and a.loc["SW-013", "name"] == "TIE1"
    store.sync_assets()                                            # idempotent
    assert len(store.assets()) == 24


def test_database_url_from_environment(monkeypatch, tmp_path):
    assert config.database_url().endswith("data/twin.db")          # default, inside project/
    monkeypatch.setenv("LFDR_DB_URL", f"sqlite:///{tmp_path / 'x.db'}")
    assert config.database_url() == f"sqlite:///{tmp_path / 'x.db'}"
    monkeypatch.setenv("LFDR_DB_URL", "postgresql+psycopg://u:p@localhost/feeder")
    assert config.database_url().startswith("postgresql")          # passed through untouched


def test_schema_version_mismatch_is_refused(store, tmp_path):
    from sqlalchemy import update
    from storage.store import SchemaVersionError
    with store.engine.begin() as c:
        c.execute(update(S.meta).where(S.meta.c.key == "schema_version").values(value="999"))
    with pytest.raises(SchemaVersionError):
        Store(store.url)


# ------------------------------------------------------------------ forecasts, measurements, batching
def test_forecast_import_round_trip(store, project_root):
    assert store.import_forecast_csv(project_root / "data" / "forecast.csv") == 6438
    assert store.import_forecast_csv(project_root / "data" / "forecast.csv") == 0          # idempotent
    csv = pd.read_csv(project_root / "data" / "forecast.csv", parse_dates=["timestamp"])
    got = store.forecast_frame()
    expected = csv.sort_values(["transformer_id", "timestamp"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got[csv.columns].reset_index(drop=True), expected, check_dtype=False)
    assert store.counts()["measurements"] == 6438 and store.model_versions() == ["xgb-24h-v1"]


def test_batched_measurements_and_retention(store, monkeypatch):
    t0 = dt.datetime(2020, 1, 1)
    rows = [{"time": t0 + dt.timedelta(minutes=15 * i), "asset_id": "DT-001", "source": "meter", "kw": float(i),
             "quality": "good"} for i in range(5000)]
    store.batch_size = 700
    store.write_measurements(rows)
    store.write_measurements(rows[:10])                             # duplicates are skipped
    assert store.counts()["measurements"] == 5000
    assert store.prune(t0 + dt.timedelta(days=60)) == {}           # retention 0 = keep everything
    store.retention = {"measurements_days": 30, "alerts_days": 0}
    deleted = store.prune(t0 + dt.timedelta(days=40))["measurements"]
    assert deleted == 10 * 96 and store.counts()["measurements"] == 5000 - 960


def test_meter_registry_import(store, tmp_path):
    p = tmp_path / "meter_registry.csv"
    pd.DataFrame([{"meter_id": "MTR-00001", "deployment_id": "1001", "household_id": "H1", "region": "Pune city",
                   "transformer": "T1", "transformer_asset_id": "DT-001"}]).to_csv(p, index=False)
    assert store.import_meter_registry(p) == 1 and store.import_meter_registry(p) == 0
    assert store.import_meter_registry(tmp_path / "missing.csv") == 0
    assert store.meters().iloc[0]["transformer_asset_id"] == "DT-001"


# ------------------------------------------------------------------ events and alerts
def test_event_filters(store):
    def ev(t, kind, sev, assets):
        return store.add_event("c", {"sim_time": t, "type": kind, "severity": sev, "message": kind}, assets)
    ev("2020-02-28 10:00", "mode", "info", [])
    ev("2020-02-29 21:00", "recommendation", "info", ["DT-004", "SW-004"])
    ev("2020-02-29 21:00", "emergency_stop", "critical", [])
    ev("2020-03-01 06:00", "applied", "info", ["DT-001"])
    assert [r["type"] for r in store.events()] == ["applied", "emergency_stop", "recommendation", "mode"]
    assert [r["type"] for r in store.events(sim_from="2020-02-29", sim_to="2020-02-29")] == \
        ["emergency_stop", "recommendation"]
    assert [r["type"] for r in store.events(severities=["critical"])] == ["emergency_stop"]
    assert [r["type"] for r in store.events(asset_id="DT-004")] == ["recommendation"]
    assert store.events(asset_id="DT-004")[0]["assets"] == ["DT-004", "SW-004"]
    assert [r["type"] for r in store.events(types=["mode", "applied"], ascending=True)] == ["mode", "applied"]


def test_alerts_are_kept_once(store):
    a = [{"asset": "DT-004", "kind": "transformer", "level": "predicted_overload", "value": 91.5, "text": "x"}]
    store.record_alerts(T, a, context="S3")
    store.record_alerts(T, a, context="S3")
    store.record_alerts(T, a, context="S4")                          # other conditions: another alert
    assert len(store.alerts()) == 2 and len(store.alerts(asset="DT-004", levels=["predicted_overload"])) == 2


# ------------------------------------------------------------------ controller persistence
def test_everything_is_written_as_it_happens(store, loads):
    c = Controller(e._default, store=store)
    rec = c.recommend(loads("S3", T), T, "S3")
    c.approve(rec.rec_id, T, "S3")
    counts = store.counts()
    assert counts["recommendations"] == 1 and counts["candidates"] == 6 and counts["events"] == 3
    assert counts["actions"] == 2 and counts["controller_state"] == 1
    assert store.load_recommendation(rec.rec_id)["status"] == "applied"
    acts = store.actions().sort_values("step")
    assert acts[["switch_id", "operation", "how"]].values.tolist() == [["SW-004", "open", "operator"],
                                                                       ["SW-010", "close", "operator"]]
    assert {r["severity"] for r in store.events()} == {"info"}


def test_restart_restores_everything(store, loads):
    c1 = Controller(e._default, store=store)
    c1.approve(c1.recommend(loads("S3", T), T, "S3").rec_id, T, "S3")
    c1.set_mode("auto")
    c1.set_mode("recommend")
    pending = c1.recommend(loads("S1", T), T, "S1")                # a restore proposal, left pending
    c1.emergency_stop(T)

    c2 = Controller(e._default, store=Store(store.url))           # a new process on the same database
    assert c2.active == ["T4>T5"] and c2.estopped and c2.mode == "monitor"
    assert c2.pending is None                                      # the stop cancelled it, and that was stored
    assert store.load_recommendation(pending.rec_id)["status"] == "superseded"
    assert [h["type"] for h in c2.history] == [h["type"] for h in c1.history]
    assert c2.can_rollback
    c2.clear_emergency_stop()
    c2.set_mode("recommend")
    nxt = c2.recommend(loads("S1", T), T, "S1")
    assert nxt.rec_id == "REC-0003"                                # numbering continues
    c2.approve(nxt.rec_id, T, "S1")
    assert c2.active == []


def test_pending_recommendation_can_be_approved_after_restart(store, loads):
    c1 = Controller(e._default, store=store)
    rec = c1.recommend(loads("S4", T), T, "S4")                    # mitigation, pending
    c2 = Controller(e._default, store=Store(store.url))
    assert c2.pending.rec_id == rec.rec_id and c2.pending.kind == "mitigate"
    assert len(c2.pending.candidates) == len(rec.candidates)
    with pytest.raises(DecisionError, match="conditions have changed"):
        c2.approve(rec.rec_id, "2020-03-01 00:00", "S4")
    c2.approve(rec.rec_id, T, "S4")
    assert c2.active == ["T4>T5"]


def test_invalid_stored_configuration_falls_back_to_normal(store):
    store.save_state("default", "recommend", ["T9>T1"], False, [], 0)
    assert Controller(e._default, store=store).active == []
    store.save_state("default", "recommend", ["MV1", "MV2"], False, [], 0)      # valid keys, not radial
    assert Controller(e._default, store=store).active == []


def test_severity_and_assets(store, loads):
    c = Controller(e._default, store=store)
    c.recommend(loads("S5", "2020-02-17 07:00"), "2020-02-17 07:00", "S5")       # T3 at 200%: critical
    ev = store.events()[0]
    assert ev["type"] == "no_safe_option" and ev["severity"] == "critical" and ev["assets"] == ["DT-003"]
    c.recommend(loads("S4", T), T, "S4")                                          # mitigation: warning
    ev = store.events(types=["recommendation"])[0]
    assert ev["severity"] == "warning" and "SW-010" in ev["assets"]


def test_recommendations_order_past_9999(store):
    for n in (9998, 9999, 10000):
        store.save_recommendation("c", {"rec_id": f"REC-{n:04d}", "kind": "no_action", "status": "info",
                                        "candidates": []})
    assert store.latest_recommendation("c")["rec_id"] == "REC-10000"


def test_state_is_saved_the_moment_a_plan_is_applied(store, loads):
    """A crash right after approval must not lose the new configuration."""
    c1 = Controller(e._default, store=store)
    c1.approve(c1.recommend(loads("S3", T), T, "S3").rec_id, T, "S3")
    assert Controller(e._default, store=Store(store.url)).active == ["T4>T5"]
    c1.rollback(T)
    assert Controller(e._default, store=Store(store.url)).active == []
