"""The Streamlit dashboard, run headless: start-up, controls, recommendations, approval, rollback,
emergency stop and automatic mode."""
import datetime as dt

import pytest
from streamlit.testing.v1 import AppTest

S3_WORST = dt.datetime(2020, 2, 29, 21, 0)


@pytest.fixture
def app(project_root):
    at = AppTest.from_file(str(project_root / "app" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


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
