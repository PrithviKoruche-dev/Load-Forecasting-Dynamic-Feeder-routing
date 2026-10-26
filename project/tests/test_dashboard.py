"""The Streamlit dashboard, run headless: it starts, shows the right numbers and the Reroute button works."""
import datetime as dt

import pytest
from streamlit.testing.v1 import AppTest


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


def test_starts_at_the_peak_hour_with_no_overloads(app):
    assert app.sidebar.slider(key="t").value == dt.datetime(2020, 2, 21, 6, 0)
    assert metric(app, "Transformers overloaded") == "0 of 6"
    assert metric(app, "Peak loading") == "56.3 %"
    assert metric(app, "Forecast nRMSE (test)") == "25.3 %"


def test_controls_come_from_config(app):
    labels = [s.label for s in app.sidebar.slider]
    assert labels == ["Replay time", "Heatwave multiplier (T1 to T3)", "EV surge (kW, 18:00 to 22:00)",
                      "Severe stress multiplier (all)", "Overload threshold (%)"]
    assert app.selectbox(key="scenario").options == ["Normal", "Heatwave (feeder A)", "EV evening surge",
                                                     "Severe stress"]
    assert app.sidebar.slider(key="thr").value == 85


def test_severe_stress_cannot_be_fixed(app):
    """Same hour as the S4 worst hour in docs/results_worst_hour.csv: 143.8%, no safe option."""
    app.selectbox(key="scenario").set_value("Severe stress").run()
    assert metric(app, "Peak loading") == "143.8 %"
    assert metric(app, "Transformers overloaded") == "3 of 6"
    click(app, "Reroute")
    assert any("CRITICAL: no safe reconfiguration" in m.value for m in app.markdown)
    assert app.session_state["moves"] == []


def test_ev_surge_is_rerouted_and_switches_move(app):
    """S3 worst hour (29 Feb 21:00): 91.5% before, 72.9% after one LV move."""
    app.selectbox(key="scenario").set_value("EV evening surge").run()
    app.sidebar.slider(key="t").set_value(dt.datetime(2020, 2, 29, 21, 0)).run()
    assert metric(app, "Peak loading") == "91.5 %"
    click(app, "Reroute")
    assert app.session_state["moves"] == ["T4"]
    assert metric(app, "Peak loading") == "72.9 %"
    assert metric(app, "Transformers overloaded") == "0 of 6"
    click(app, "Reset switches")
    assert app.session_state["moves"] == []
    assert metric(app, "Peak loading") == "91.5 %"
