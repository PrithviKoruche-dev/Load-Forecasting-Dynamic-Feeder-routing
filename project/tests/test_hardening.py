"""Phase 10: the switching deadband, a failed step that pauses instead of stopping the service, and settings
checks."""
import pandas as pd
import pytest

import engine as e
import run_results as rr
from core import config, scenarios
from decisions import Controller

S3_HOUR = "2020-02-29 21:00"


@pytest.fixture(scope="module")
def s3_loads():
    pred = rr.load_predictions()
    sc = {s.name.split()[0]: s for s in scenarios.benchmark()}
    ts = pd.Timestamp(S3_HOUR)
    return {tx: float(sc["S3"].adjust(tx, ts, pred.loc[ts, tx])) for tx in e.TX}


@pytest.fixture
def engine():
    g = e.GridEngine()
    yield g
    g.restore_margin = 0.0


def scaled_to(engine, loads, peak):
    """The loads scaled so that the normal configuration peaks at `peak` %."""
    lo, hi = 0.1, 3.0
    for _ in range(40):
        k = (lo + hi) / 2
        m = engine.evaluate_configuration({t: v * k for t, v in loads.items()}, [])["metrics"]["max_loading"]
        lo, hi = (k, hi) if m < peak else (lo, k)
    return {t: v * (lo + hi) / 2 for t, v in loads.items()}


def rerouted(engine, s3_loads):
    ctl = Controller(engine, now=lambda: "2026-01-01T00:00:00")
    ctl.set_mode("auto")
    rec = ctl.recommend(s3_loads, S3_HOUR, "S3")
    assert rec.kind == "reroute" and ctl.active
    return ctl


@pytest.mark.parametrize("peak, margin, expected", [(83.0, 0, "restore"), (83.0, 5, "no_change"),
                                                    (78.0, 5, "restore")])
def test_the_deadband_holds_a_reroute_until_the_normal_configuration_is_comfortably_under(engine, s3_loads, peak,
                                                                                          margin, expected):
    engine.restore_margin = margin
    ctl = rerouted(engine, s3_loads)
    before = list(ctl.active)
    rec = ctl.recommend(scaled_to(engine, s3_loads, peak), "2020-02-29 22:00", "S3")
    assert rec.kind == expected
    assert ctl.active == (before if expected == "no_change" else [])
    if expected == "no_change":
        assert "Holding the current configuration" in rec.summary and f"below {85 - margin:g}%" in rec.summary


def test_the_deadband_never_holds_a_configuration_that_is_itself_overloaded(engine, s3_loads):
    engine.restore_margin = 5
    ctl = rerouted(engine, s3_loads)
    worse = {t: v * 1.3 for t, v in s3_loads.items()}                         # the reroute no longer suffices
    cur = engine.evaluate_configuration(worse, ctl.active)["metrics"]
    assert max(cur["loading_by_tx"].values()) > 85
    rec = ctl.recommend(worse, "2020-02-29 22:00", "S3")
    assert "Holding" not in rec.summary


def test_restore_margin_is_checked(tmp_path, monkeypatch, project_root):
    import shutil
    import yaml
    for bad, msg in ((-1, "restore_margin_pct"), (85, "must be below")):
        d = tmp_path / f"c{bad}"
        shutil.copytree(project_root / "config", d)
        ops = yaml.safe_load((d / "operations.yaml").read_text())
        ops["rerouting"]["restore_margin_pct"] = bad
        (d / "operations.yaml").write_text(yaml.safe_dump(ops))
        monkeypatch.setenv("LFDR_CONFIG_DIR", str(d))
        config.reload()
        with pytest.raises(config.ConfigError, match=msg):
            config.operations()
    monkeypatch.delenv("LFDR_CONFIG_DIR")
    config.reload()
    assert config.operations()["rerouting"]["restore_margin_pct"] == 0          # the benchmark keeps no deadband


def test_the_city_uses_a_deadband(project_root):
    import yaml
    ops = yaml.safe_load((project_root / "cities" / "demo" / "config" / "operations.yaml").read_text())
    assert ops["rerouting"]["restore_margin_pct"] == 5


class FakeWall:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_a_failed_step_pauses_the_clock_and_the_service_keeps_serving(tmp_path, project_root):
    from orchestrator import Orchestrator
    from service import Service
    from storage import Store
    s = Store(f"sqlite:///{(tmp_path / 'svc.db').as_posix()}")
    s.import_forecast_csv(project_root / "data" / "forecast.csv")
    s.import_intervals_csv(project_root / "data" / "forecast_intervals.csv")
    w = FakeWall()
    orch = Orchestrator(s, engine=e.GridEngine(), service="service", forecast_source="precomputed")
    svc = Service(s, name="service", orchestrator=orch, wall=w, sleep=w.sleep)
    svc.execute("speed", {"speed": "max"})
    svc.execute("resume")
    real, calls = orch.step, []

    def flaky(t, run_id=0):
        calls.append(t)
        if len(calls) == 2:
            raise RuntimeError("power flow did not converge")
        return real(t, run_id)

    orch.step = flaky
    svc.run(max_steps=3, exit_when_idle=True)
    state = s.load_sim_state("service")
    assert state["status"] == "step_failed" and "RuntimeError: power flow did not converge" in state["note"]
    assert len(s.steps("service")) == 1 and calls[1] == svc.clock.next_time  # paused before the failed step
    ev = s.events(controller="service")
    assert any(x["severity"] == "critical" and "failed" in x["message"] for x in ev)
    s.send_command("service", "resume")
    svc.run(max_steps=2, exit_when_idle=True)
    assert len(s.steps("service")) == 3 and s.load_sim_state("service")["status"] == "running"
    s.send_command("service", "pause")
    svc.handle_commands()
    assert s.load_sim_state("service")["status"] == "paused"                 # the failure is no longer reported


# ------------------------------------------------------------------ a new timeline (new run, jump, scenario)
@pytest.fixture
def city(project_root, monkeypatch):
    monkeypatch.setenv("LFDR_CONFIG_DIR", str(project_root / "cities" / "demo" / "config"))
    config.reload()
    yield
    monkeypatch.delenv("LFDR_CONFIG_DIR")
    config.reload()


def city_service(tmp_path, prev_source=None, new_run=False):
    from orchestrator import prepare_database
    from service import Service
    from storage import Store
    s = Store(f"sqlite:///{(tmp_path / 'city.db').as_posix()}")
    prepare_database(s)
    if prev_source:                                                           # a state saved by an older run
        s.save_sim_state("service", run_id=3, scenario="C4 Heatwave and EV adoption", forecast_source=prev_source,
                         status="stopped", clock={"speed": "max", "running": False, "next_time": "2023-08-03 16:15",
                                                   "last_time": None})
    w = FakeWall()
    return s, Service(s, name="service", wall=w, sleep=w.sleep, new_run=new_run)


def test_a_new_run_uses_the_configured_forecast_not_the_one_saved_by_an_older_run(city, tmp_path):
    s, svc = city_service(tmp_path, prev_source="persistence")
    assert svc.orch.forecasts.name == "persistence"                          # resuming the old run keeps it
    svc.execute("restart", {"from": "2023-05-09 18:00"})
    assert svc.orch.forecasts.name == "short" and svc.run_id == 4
    s.engine.dispose()
    (tmp_path / "b").mkdir()
    _, svc2 = city_service(tmp_path / "b", prev_source="persistence", new_run=True)
    assert svc2.orch.forecasts.name == "short"


def test_a_new_timeline_clears_what_earlier_runs_wrote_from_its_start(city, tmp_path):
    s, svc = city_service(tmp_path)
    svc.execute("mode", {"mode": "auto"})
    svc.execute("jump", {"time": "2023-05-09 18:00"})
    for _ in range(3):
        svc.step_once()
    mv = svc.orch.forecasts.model_version
    with s.engine.connect() as c:
        n = lambda q: c.exec_driver_sql(q).scalar_one()  # noqa: E731
        before = (n("select count(*) from measurements where source = 'dt-scada'"),
                  n(f"select count(*) from forecasts where model_version = '{mv}'"))
    assert before[0] > 0 and before[1] > 0
    svc.execute("restart", {"from": "2023-05-09 18:15"})                      # a new run over the same times
    with s.engine.connect() as c:
        after = (n("select count(*) from measurements where source = 'dt-scada'"),
                 n(f"select count(*) from forecasts where model_version = '{mv}'"),
                 n("select min(time) from measurements where source = 'dt-scada'"))
    tx = len(config.grid()["transformers"])
    assert after[0] == tx and str(after[2]).startswith("2023-05-09 18:00")   # only the step before the new start
    assert after[1] == 0                                                       # forecasts for 18:15 on are gone
    svc.step_once()
    perf = s.forecast_performance(until="2023-05-09 18:15", model_version=mv, since="2023-05-09 18:15")
    assert perf.empty                                                         # nothing from the old run is scored


def test_the_classic_network_keeps_its_imported_data_on_a_new_run(tmp_path, project_root):
    from orchestrator import Orchestrator
    from storage import Store
    s = Store(f"sqlite:///{(tmp_path / 'c.db').as_posix()}")
    s.import_forecast_csv(project_root / "data" / "forecast.csv")
    o = Orchestrator(s, engine=e.GridEngine(), service="t", forecast_source="precomputed")
    n0 = s.counts()
    assert o.clear_from(o.demand.times()[0]) == {} and s.counts() == n0


def test_the_forecasting_page_opens_on_the_model_the_service_uses(city, tmp_path, monkeypatch, project_root):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("LFDR_DB_URL", f"sqlite:///{(tmp_path / 'city.db').as_posix()}")
    s, svc = city_service(tmp_path)
    s.write_forecasts([("2023-01-01 00:00", "DT-001", 1.0)], "persistence-24h")      # left by older runs
    s.write_forecasts([("2023-01-01 00:00", "DT-001", 1.0)], "short-15m-v1", horizon_h=0.25)
    s.write_measurements([{"time": pd.Timestamp("2023-01-01").to_pydatetime(), "asset_id": "DT-001",
                           "source": "dt-scada", "kw": 2.0, "quality": "good"}])
    s.score_forecasts("2023-01-02", model_version="short-15m-v1")
    svc.execute("jump", {"time": "2023-05-09 18:00"})
    for _ in range(3):
        svc.step_once()
    svc.save_state()
    at = AppTest.from_file(str(project_root / "app" / "views" / "forecasting.py"), default_timeout=90)
    at.run()
    assert not at.exception
    assert at.selectbox(key="fc_version").value == "short-15m-v1"
    m = {x.label: x.value for x in at.metric}
    run = s.forecast_performance(until="2023-05-09 18:30", model_version="short-15m-v1", since="2023-05-09 18:00")
    assert int(m["Scored"].replace(",", "")) == int(run.set_index("transformer").loc["ALL", "scored"]) > 0   # this run's


def test_the_city_confidence_uses_the_city_forecast_error(city, tmp_path):
    """A restore to a peak of 76% is safe to within a 4% forecast error: the confidence must say so (with the
    classic day-ahead error of about 33% it said 'low')."""
    s, svc = city_service(tmp_path)
    err = svc.orch.ctl.error_by_tx
    assert set(err) == set(config.grid()["transformers"]) and 0.03 < max(err.values()) < 0.05
    loads = {t: 60.0 for t in err}
    loads["T01"] = 76.0
    c = svc.orch.ctl.risk.confidence(loads, err, 85)
    assert c["label"] == "high"
    svc.orch.set_forecast_source("persistence")
    assert 0.2 < max(svc.orch.ctl.error_by_tx.values()) < 0.25
