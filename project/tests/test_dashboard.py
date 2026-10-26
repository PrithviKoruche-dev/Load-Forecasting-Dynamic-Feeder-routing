"""The Streamlit dashboard, run headless: start-up, controls, recommendations, approval, rollback,
emergency stop and automatic mode."""
import datetime as dt

import pytest
from streamlit.testing.v1 import AppTest

S3_WORST = dt.datetime(2020, 2, 29, 21, 0)


def start(project_root):
    at = AppTest.from_file(str(project_root / "app" / "views" / "replay.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Every test gets its own throwaway database (never the real data/twin.db)."""
    url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("LFDR_DB_URL", url)
    return url


@pytest.fixture
def app(project_root, db):
    return start(project_root)


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, at.exception


def labels(at):
    return [b.label for b in at.button]


def ctl(at):
    return at.session_state["ctl"]


def ev_worst_hour(at):
    at.selectbox(key="scenario").set_value("EV evening surge").run()
    at.sidebar.slider(key="t").set_value(S3_WORST).run()
    assert metric(at, "Peak loading") == "91.5 %"


def test_starts_at_the_peak_hour_with_no_overloads(app):
    assert app.sidebar.slider(key="t").value == dt.datetime(2020, 2, 21, 6, 0)
    assert metric(app, "Transformers overloaded") == "0 of 6"
    assert metric(app, "Peak loading") == "56.3 %"
    assert metric(app, "Forecast nRMSE (test)") == "25.3 %"
    assert metric(app, "Losses").endswith("kW")


def test_controls_come_from_config(app):
    assert [s.label for s in app.sidebar.slider] == [
        "Replay time", "Heatwave multiplier (T1 to T3)", "EV surge (kW, 18:00 to 22:00)",
        "Severe stress multiplier (all)", "Overload threshold (%)"]
    assert app.selectbox(key="scenario").options == ["Normal", "Heatwave (feeder A)", "EV evening surge",
                                                     "Severe stress"]
    assert app.sidebar.slider(key="thr").value == 85
    assert app.radio(key="mode_widget").value == "recommend"
    assert "Get recommendation" in labels(app) and "EMERGENCY STOP" in labels(app)


def test_risk_table_separates_forecast_and_measured(app):
    ev_worst_hour(app)
    table = app.dataframe[0].value.set_index("transformer")
    assert table.loc["T4", "forecast risk"] == "Predicted overload"
    assert table.loc["T4", "measured risk"] == "Approaching limit"


def test_severe_stress_has_no_safe_option(app):
    app.selectbox(key="scenario").set_value("Severe stress").run()
    assert metric(app, "Peak loading") == "143.8 %"
    assert metric(app, "Transformers overloaded") == "3 of 6"
    click(app, "Get recommendation")
    assert ctl(app).last.kind == "no_safe_option"
    assert any(m.value.startswith("- T4 (143.8%): even with its block b moved to T5") for m in app.markdown)
    assert "Approve" not in labels(app) and ctl(app).active == []


def test_approve_then_roll_back(app):
    ev_worst_hour(app)
    click(app, "Get recommendation")
    assert ctl(app).active == []                       # nothing switched until approved
    click(app, "Approve")
    assert ctl(app).active == ["T4>T5"]
    assert metric(app, "Peak loading") == "72.9 %" and metric(app, "Transformers overloaded") == "0 of 6"
    click(app, "Roll back last change")
    assert ctl(app).active == []
    assert metric(app, "Peak loading") == "91.5 %"


def test_reject(app):
    ev_worst_hour(app)
    click(app, "Get recommendation")
    click(app, "Reject")
    assert ctl(app).active == [] and ctl(app).history[-1]["type"] == "rejected"


def test_stale_approval_is_refused(app):
    ev_worst_hour(app)
    click(app, "Get recommendation")
    rec_id = ctl(app).pending.rec_id
    app.sidebar.slider(key="t").set_value(S3_WORST + dt.timedelta(hours=1)).run()
    next(b for b in app.button if b.label == "Approve").click().run()
    assert ctl(app).active == []
    assert any("conditions have changed" in x.value for x in app.error)
    assert ctl(app).pending.rec_id == rec_id


def test_emergency_stop(app):
    ev_worst_hour(app)
    click(app, "EMERGENCY STOP")
    assert ctl(app).estopped and ctl(app).mode == "monitor"
    assert "Get recommendation" not in labels(app)
    assert any("EMERGENCY STOP" in x.value for x in app.error)
    click(app, "Clear emergency stop")
    assert not ctl(app).estopped and ctl(app).mode == "monitor"


def test_automatic_mode_switches_by_itself(app):
    ev_worst_hour(app)
    app.radio(key="mode_widget").set_value("auto").run()
    assert ctl(app).active == ["T4>T5"]
    assert metric(app, "Peak loading") == "72.9 %"
    assert [h["type"] for h in ctl(app).history][-2:] == ["recommendation", "applied"]


def test_mitigation_from_the_dashboard(app):
    app.selectbox(key="scenario").set_value("Severe stress").run()
    app.sidebar.slider(key="t").set_value(S3_WORST).run()            # 29 Feb 21:00
    assert metric(app, "Peak loading") == "122.2 %"
    click(app, "Get recommendation")
    assert ctl(app).last.kind == "mitigate"
    assert any(w.value.startswith("Mitigation, not a full fix") for w in app.warning)
    click(app, "Approve")
    assert ctl(app).active == ["T4>T5"] and metric(app, "Peak loading") == "97.1 %"


# ------------------------------------------------------------------ persistence (Phase 3)
def test_state_survives_a_restart(project_root, db):
    """Approve, leave a recommendation pending, 'restart' the dashboard: everything is still there."""
    import streamlit as st
    at = start(project_root)
    ev_worst_hour(at)
    click(at, "Get recommendation")
    click(at, "Approve")                                       # T4>T5 in force
    at.selectbox(key="scenario").set_value("Severe stress").run()
    click(at, "Reset to normal configuration")
    click(at, "Get recommendation")                            # a mitigation, left pending
    pending = ctl(at).pending.rec_id
    n_events = len(ctl(at).history)

    st.cache_resource.clear()                                  # what a restart does to the process' objects
    at2 = start(project_root)
    c = ctl(at2)
    assert c is not ctl(at)                                    # really a new controller...
    assert c.active == [] and c.pending.rec_id == pending      # ...with the same state
    assert len(c.history) == n_events
    events_table = next(d for d in at2.dataframe if "event" in d.value.columns)
    assert {"recommendation", "approved", "applied", "manual"} <= set(events_table.value["event"])


def test_audit_log_filters(app):
    ev_worst_hour(app)
    click(app, "Get recommendation")
    click(app, "Approve")
    log = lambda: next(d for d in app.dataframe if "event" in d.value.columns).value  # noqa: E731
    assert len(log()) == 3
    app.selectbox(key="log_tx").set_value("T1").run()
    assert not any("event" in d.value.columns for d in app.dataframe)      # nothing concerns T1
    app.selectbox(key="log_tx").set_value("T4").run()
    assert len(log()) == 3
    app.multiselect(key="log_types").set_value(["applied"]).run()
    assert log()["event"].tolist() == ["applied"]


def test_each_event_is_logged_once(app):
    """Regression: switching to automatic mode at an hour that needs no action wrote 'Mode changed' to
    logs/dashboard.log twice (the assessment re-logged the previous event)."""
    import logging

    class Grab(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    grab = Grab()
    logger = logging.getLogger("lfdr.dashboard")
    logger.addHandler(grab)
    try:
        app.radio(key="mode_widget").set_value("auto").run()
    finally:
        logger.removeHandler(grab)
    assert sum("Mode changed" in m for m in grab.messages) == 1


def test_forecast_risk_uses_the_p90_bound(app):
    """Phase 4: the risk table shows the P90 loading and the forecast class follows it."""
    ev_worst_hour(app)
    table = app.dataframe[0].value.set_index("transformer")
    assert list(table.columns[:3]) == ["forecast %", "P90 %", "measured %"]
    assert table.loc["T4", "P90 %"] == 107.9 and table.loc["T4", "forecast %"] == 91.5
    assert (table["P90 %"] >= table["forecast %"]).all()


def test_load_chart_shows_the_scenario_adjusted_band(app):
    import json
    ev_worst_hour(app)
    load = json.loads(app.get("plotly_chart")[1].proto.spec)
    names = [tr.get("name") for tr in load["data"]]
    assert "P10-P90 band + scenario" in names and "forecast + scenario" in names
    assert next(tr for tr in load["data"] if tr.get("name") == "P10-P90 band + scenario")["fill"] == "tonexty"


def test_forecasts_are_scored_as_the_replay_advances(app):
    acc = app.dataframe[1].value.set_index("transformer")
    assert list(acc.columns) == ["scored", "pending", "MAE_kW", "RMSE_kW", "nRMSE_%", "bias_kW"]
    early = acc.loc["ALL", "scored"]
    ev_worst_hour(app)
    late = app.dataframe[1].value.set_index("transformer")
    assert late.loc["ALL", "scored"] == 2970 > early and late.loc["ALL", "pending"] == 0
