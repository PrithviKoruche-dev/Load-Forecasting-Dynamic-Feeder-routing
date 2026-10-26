"""Phase 6: the multi-page dashboard (gap items G3 to G9), run headless against a database the simulation service
filled; service settings (threshold, MV transfers, approval policy); unique recommendation numbers; schema
migration of a Phase 5 database."""
import shutil
import sqlite3

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

S3 = "S3 EV surge (T4 +200 kW, 18-22 h)"


class FakeWall:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make_service(store, **kw):
    from service import Service
    w = FakeWall()
    return Service(store, name="service", wall=w, sleep=w.sleep, **kw)


@pytest.fixture(scope="module")
def built_db(tmp_path_factory):
    """S3 in automatic mode through 28 Feb (a reroute and a restore), then recommend mode until a proposal waits."""
    from storage import Store
    path = tmp_path_factory.mktemp("pages") / "built.db"
    store = Store(f"sqlite:///{path.as_posix()}")
    svc = make_service(store, scenario=S3)
    for c, a in (("mode", {"mode": "auto"}), ("jump", {"time": "2020-02-28 06:00"})):
        svc.execute(c, a)
    for _ in range(36):
        svc.step_once()
    svc.execute("mode", {"mode": "recommend"})
    svc.execute("resume")
    while not svc.orch.awaiting_operator:
        svc.step_once()
    svc.clock.pause()
    svc.paused_for_rec, svc.note = True, f"awaiting operator: {svc.ctl.pending.rec_id}"
    svc.save_state()
    info = {"rec_id": svc.ctl.pending.rec_id, "last": svc.clock.last_time, "target": svc.ctl.pending.target}
    with store.engine.connect() as c:
        c.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
    store.engine.dispose()
    return path, info


@pytest.fixture
def db(built_db, tmp_path, monkeypatch):
    """A private copy of the built database, used by the pages through LFDR_DB_URL."""
    from storage import Store
    src, info = built_db
    dst = tmp_path / "copy.db"
    shutil.copy(src, dst)
    url = f"sqlite:///{dst.as_posix()}"
    monkeypatch.setenv("LFDR_DB_URL", url)
    return Store(url), info


def page(project_root, name):
    at = AppTest.from_file(str(project_root / "app" / "views" / f"{name}.py"), default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    return at


def metrics(at):
    return {m.label: m.value for m in at.metric}


# ------------------------------------------------------------------ every page without any service yet
@pytest.mark.parametrize("name", ["overview", "network", "transformers", "forecasting", "decisions", "controls", "audit"])
def test_pages_work_on_an_empty_database(project_root, tmp_path, monkeypatch, name):
    monkeypatch.setenv("LFDR_DB_URL", f"sqlite:///{(tmp_path / 'empty.db').as_posix()}")
    at = page(project_root, name)
    if name not in ("controls", "audit"):
        assert "No simulation service has run" in at.info[0].value
        assert any(b.label == "Start the service" for b in at.button)


def test_entry_point_opens_the_overview(project_root, db):
    at = AppTest.from_file(str(project_root / "app" / "app.py"), default_timeout=90)
    at.run()
    assert not at.exception and at.title[0].value == "Overview"


# ------------------------------------------------------------------ A overview (G3)
def test_overview_counts_capacity_clock_and_season(project_root, db):
    store, info = db
    at = page(project_root, "overview")
    m = metrics(at)
    assert (m["Substations"], m["Feeders"], m["Transformers"], m["Switches"]) == ("1", "2", "6", "15")
    assert m["Transformer capacity"] == "3,000 kVA"
    last = store.latest_step if hasattr(store, "latest_step") else None
    row = store.steps("service", newest_first=True, limit=1).iloc[0]
    assert m["Simulated time"] == f"{info['last']:%d %b %H:%M}" and m["Season"] == "Winter"
    assert m["Total load"] == f"{sum(row['loads_kw'].values()):,.0f} kW"
    assert m["Peak loading"] == f"{row['measured_max_pct']:.1f} %"
    assert any("awaiting operator" in w.value or "awaits the operator" in w.value for w in at.warning)


# ------------------------------------------------------------------ B network (G4)
def test_network_inspects_a_transformer_and_a_feeder(project_root, db):
    store, info = db
    at = page(project_root, "network")
    row = store.steps("service", newest_first=True, limit=1).iloc[0]
    at.selectbox(key="net_pick").set_value("T4").run()
    m = metrics(at)
    assert m["Measured loading"] == f"{row['measured_pct']['T4']:.1f}%"
    assert m["Next hour forecast"] == f"{row['next_forecast_pct']['T4']:.1f}%"
    assert any("can move to **T5**" in md.value for md in at.markdown)
    feeder_b = next(o for o in at.selectbox(key="net_pick").options if o.startswith("Feeder B"))
    at.selectbox(key="net_pick").set_value(feeder_b).run()
    m = metrics(at)
    assert m["Head current"] == f"{row['feeders']['B']['current_a']:,.0f} A"
    assert any("Supplies **T4, T5, T6**" in md.value for md in at.markdown)


# ------------------------------------------------------------------ C transformers (G5)
def test_transformer_table_and_history(project_root, db):
    store, info = db
    at = page(project_root, "transformers")
    t = at.dataframe[0].value
    row = store.steps("service", newest_first=True, limit=1).iloc[0]
    assert t["transformer"].tolist() == ["T1", "T2", "T3", "T4", "T5", "T6"]
    assert {"rating kVA", "loading %", "forecast %", "P90 %", "next hour %", "risk class", "time"} <= set(t.columns)
    assert t.set_index("transformer").loc["T4", "loading %"] == row["measured_pct"]["T4"]
    at.selectbox(key="tx_pick").set_value("T4").run()
    at.radio(key="tx_span").set_value("Whole run").run()
    assert not at.exception
    rerouted = sum(1 for c in store.steps("service")["config_before"] if "T4>T5" in (c or []))
    assert rerouted > 0 and any(f"in force during {rerouted} hours" in c.value for c in at.caption)


# ------------------------------------------------------------------ D forecasting (G6)
def test_forecasting_scores_up_to_the_simulated_time(project_root, db):
    store, info = db
    at = page(project_root, "forecasting")
    perf = store.forecast_performance(until=info["last"]).set_index("transformer")
    m = metrics(at)
    assert m["Scored"] == f"{int(perf.loc['ALL', 'scored']):,}"
    assert m["Pending"] == f"{6438 - int(perf.loc['ALL', 'scored']):,}"
    assert m["nRMSE"] == f"{perf.loc['ALL', 'nRMSE_%']:.1f} %"
    at.radio(key="fc_metric").set_value("RMSE").run()
    at.radio(key="fc_freq").set_value("Weekly").run()
    assert not at.exception and any(h.value == "RMSE over time (kW)" for h in at.subheader)


# ------------------------------------------------------------------ E alerts and decisions (G7)
def test_waiting_proposal_with_candidates_and_approval(project_root, db):
    store, info = db
    at = page(project_root, "decisions")
    rec_id = info["rec_id"]
    assert any(h.value == f"Waiting for the operator: {rec_id}" for h in at.subheader)
    cand = next(d.value for d in at.dataframe if "reason" in d.value.columns)
    assert "selected" in set(cand["status"]) and cand["reason"].notna().all()
    next(b for b in at.button if b.label == "Approve").click().run()
    cmd = store.commands("service").iloc[0]
    assert (cmd["command"], cmd["args"], cmd["status"]) == ("approve", {"rec_id": rec_id}, "pending")
    svc = make_service(store)
    svc.handle_commands()
    assert store.command(int(cmd["id"]))["status"] == "done" and svc.ctl.active == info["target"]
    at.run()
    assert not any(h.value.startswith("Waiting for the operator") for h in at.subheader)
    at.selectbox(key="dec_pick").set_value(rec_id).run()
    assert not at.exception and any("status `applied`" in c.value for c in at.caption)


# ------------------------------------------------------------------ F controls (G8)
def test_controls_show_what_is_in_force_and_send_commands(project_root, db):
    store, info = db
    at = page(project_root, "controls")
    assert next(r for r in at.radio if r.label == "Operating mode").value == "recommend"
    assert next(s for s in at.selectbox if s.label == "Scenario").value == S3
    thr = next(s for s in at.slider if s.label == "Overload threshold (%)")
    assert thr.value == 85
    thr.set_value(80).run()
    next(b for b in at.button if b.label == "Set threshold").click().run()
    cmd = store.commands("service").iloc[0]
    assert (cmd["command"], cmd["args"]) == ("threshold", {"value": 80})
    svc = make_service(store)
    svc.handle_commands()
    assert svc.orch.threshold == 80 and store.load_sim_state("service")["settings"]["threshold"] == 80
    at.run()
    assert next(s for s in at.slider if s.label == "Overload threshold (%)").value == 80   # shows the new value
    assert any("Season now: **winter**" in md.value for md in at.markdown)


def test_controls_speed_and_jump_inputs(project_root, db):
    import datetime as dt
    store, info = db
    at = page(project_root, "controls")
    assert next(s for s in at.selectbox if s.label == "Speed").value == "1 simulated hour per second (3600x)"
    at.date_input(key="jump_date").set_value(dt.date(2020, 3, 1)).run()
    at.number_input(key="jump_hour").set_value(18).run()
    svc = make_service(store)
    svc.execute("reject", {"rec_id": info["rec_id"]})
    svc.step_once()                                                               # the clock moves on meanwhile
    at.run()
    assert at.date_input(key="jump_date").value == dt.date(2020, 3, 1)          # the choice survives
    assert at.number_input(key="jump_hour").value == 18
    next(b for b in at.button if b.label == "Jump to this time").click().run()
    cmd = store.commands("service").iloc[0]
    assert (cmd["command"], cmd["args"]) == ("jump", {"time": "2020-03-01 18:00"})


def test_controls_after_a_run_has_finished(project_root, db):
    """At the end of the data the next hour lies past the range: the page must still offer a new run."""
    store, info = db
    svc = make_service(store)
    svc.execute("reject", {"rec_id": info["rec_id"]})
    svc.execute("jump", {"time": "2020-03-24 22:00"})
    svc.step_once()
    svc.step_once()
    svc.save_state()
    assert store.load_sim_state("service")["status"] == "finished"
    at = page(project_root, "controls")
    assert any("reached the end of the data" in i.value for i in at.info)
    assert next(b for b in at.button if b.label == "Resume").disabled
    assert at.date_input(key="jump_date").value.isoformat() == "2020-03-24"
    at.date_input(key="jump_date").set_value(__import__("datetime").date(2020, 2, 28)).run()
    at.number_input(key="jump_hour").set_value(12).run()
    next(b for b in at.button if b.label == "Start a new run from this time").click().run()
    cmd = store.commands("service").iloc[0]
    assert (cmd["command"], cmd["args"]) == ("restart", {"from": "2020-02-28 12:00"})
    svc.handle_commands()
    assert svc.run_id == 2 and svc.clock.next_time == __import__("pandas").Timestamp("2020-02-28 12:00")


def test_banner_shows_the_newest_step_not_the_last_heartbeat(project_root, db):
    store, info = db
    st_ = store.load_sim_state("service")
    store.save_sim_state("service", clock={**st_["clock"], "last_time": "2020-02-28 06:00:00"})   # an old heartbeat
    at = page(project_root, "transformers")
    assert any(f"simulated time {info['last']:%Y-%m-%d %H:%M}" in c.value for c in at.caption)


def test_start_button_launches_a_detached_service(project_root, db, monkeypatch):
    import common
    calls = []
    monkeypatch.setattr(common.subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)))
    common.start_service()
    (cmd, kw), = calls
    assert cmd[1].endswith("service.py") and kw["cwd"] == str(project_root)
    assert kw.get("start_new_session") or kw.get("creationflags")           # survives the dashboard stopping


# ------------------------------------------------------------------ G audit log (G9)
def test_audit_log_filters_and_tabs(project_root, db):
    store, info = db
    at = page(project_root, "audit")
    at.multiselect(key="audit_types").set_value(["applied"]).run()
    ev = at.dataframe[0].value
    assert len(ev) > 0 and set(ev["event"]) == {"applied"}
    at.selectbox(key="audit_tx").set_value("T1").run()
    assert not at.exception
    sw = next(d.value for d in at.dataframe if "switch_id" in d.value.columns)
    assert set(sw["how"]) == {"auto"} and len(sw) == len(store.actions(controller="service"))
    at.radio(key="audit_src").set_value("What-if sandbox").run()
    assert not at.exception


# ------------------------------------------------------------------ service settings
def test_settings_are_validated_persisted_and_restored(db):
    store, _ = db
    from decisions import DecisionError  # noqa: F401
    svc = make_service(store)
    with pytest.raises(ValueError, match="hard limit"):
        svc.execute("threshold", {"value": 120})
    svc.execute("threshold", {"value": 82})
    svc.execute("use_mv", {"value": True})
    svc.execute("policy", {"pause_on_recommendation": False})
    svc.save_state()
    again = make_service(store)
    assert again.orch.threshold == 82 and again.orch.use_mv is True and again.pause_on_rec is False
    assert "threshold=82" in again.orch.context()


def test_without_the_pause_policy_an_undecided_proposal_expires(db):
    """Pause policy off: the clock keeps running, and a proposal whose hour begins without a decision expires."""
    store, info = db
    svc = make_service(store)
    svc.execute("policy", {"pause_on_recommendation": False})
    svc.execute("resume")
    for _ in range(3):
        svc.step_once()
    assert svc.clock.running and svc.status == "running" and svc.ctl.pending is None
    recs = store.recommendations(controller="service").set_index("rec_id")
    assert recs.loc[info["rec_id"], "status"] == "superseded"
    ev = store.events(controller="service", types=["superseded"], limit=5)
    assert any("expired: hour 2020-02-29 21:00 began without a decision" in e["message"] for e in ev)


def test_recommendation_numbers_are_unique_across_controllers(db):
    store, _ = db
    import engine
    from decisions import Controller
    before = store.recommendations(limit=10000).set_index("rec_id")
    sandbox = Controller(engine._default, store=store, name="default")
    sandbox.set_mode("recommend")
    r = sandbox.recommend({t: 450.0 for t in engine.TX}, "2020-03-01 19:00", "sandbox")
    after = store.recommendations(limit=10000).set_index("rec_id")
    assert len(after) == len(before) + 1 and r.rec_id not in before.index        # a new number, nothing overwritten
    for col in ("for_time", "kind", "summary"):                                    # the service's rows unchanged
        assert after.loc[before.index, col].equals(before[col])


# ------------------------------------------------------------------ migration of a Phase 5 database
def test_a_phase5_database_is_upgraded_in_place(tmp_path):
    from storage import Store
    from storage import schema as S
    path = tmp_path / "old.db"
    s = Store(f"sqlite:///{path.as_posix()}")
    s.record_step({"service": "x", "run_id": 1, "sim_time": "2020-02-10 10:00", "decision": "no_action"})
    s.engine.dispose()
    con = sqlite3.connect(path)
    for col in ("loads_kw", "measured_pct", "forecast_pct", "upper_pct", "next_forecast_pct", "next_upper_pct",
                "tx_detail", "feeders", "switches", "switches_after", "min_voltage_bus"):
        con.execute(f"ALTER TABLE sim_steps DROP COLUMN {col}")
    con.execute("ALTER TABLE sim_state DROP COLUMN settings")
    con.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
    con.commit()
    con.close()
    s = Store(f"sqlite:///{path.as_posix()}")
    with s.engine.connect() as c:
        cols = {r[1] for r in c.exec_driver_sql("PRAGMA table_info(sim_steps)")}
        version = c.exec_driver_sql("SELECT value FROM meta WHERE key = 'schema_version'").scalar_one()
    assert {"loads_kw", "switches_after", "min_voltage_bus"} <= cols and version == str(S.SCHEMA_VERSION)
    assert s.steps("x")["decision"].tolist() == ["no_action"]                  # the old data is still there
