"""Five risk classes, voltage/line alerts and recommendation confidence."""
import pytest

from risk import CLASSES, SEVERITY, RiskModel, forecast_errors


@pytest.fixture(scope="module")
def rm():
    return RiskModel()


@pytest.mark.parametrize("pct, band", [(69.9, "normal"), (70.0, "elevated"), (79.9, "elevated"),
                                       (80.0, "approaching"), (85.0, "approaching"), (85.1, "overload"),
                                       (None, None)])
def test_bands_at_85_percent(rm, pct, band):
    assert rm.band(pct) == band


def test_bands_follow_the_operator_threshold(rm):
    assert rm.band(76.0, threshold=75) == "overload"
    assert rm.band(71.0, threshold=75) == "approaching"


def test_forecast_and_measured_risk_are_separate(rm):
    """S3 worst hour on the dashboard: T4 forecast 91.5%, measured 81.5%."""
    r = rm.transformer(91.5, 81.5)
    assert (r["forecast_class"], r["measured_class"], r["risk_class"]) == \
        ("predicted_overload", "approaching", "predicted_overload")
    r = rm.transformer(60.0, 90.0)
    assert (r["forecast_class"], r["measured_class"], r["risk_class"]) == ("normal", "actual_overload", "actual_overload")
    assert rm.transformer(60.0)["risk_class"] == "normal"


def test_upper_forecast_bound_can_raise_risk(rm):
    assert rm.transformer(80.0, forecast_upper_pct=90.0)["forecast_class"] == "predicted_overload"


def test_class_order():
    assert CLASSES == ("normal", "elevated", "approaching", "predicted_overload", "actual_overload")
    assert SEVERITY["actual_overload"] > SEVERITY["predicted_overload"] > SEVERITY["approaching"]


def test_voltage_and_line_alerts(rm):
    v = rm.voltage_alerts({"sub": 1.0, "a1": 0.99, "l3": 0.955, "l3b": 0.94, "l6": 1.06})
    got = {(x["element"], x["type"], x["level"]) for x in v}
    assert got == {("L3", "voltage_low", "approaching"), ("L3B", "voltage_low", "violation"),
                   ("L6", "voltage_high", "violation")}
    lines = rm.line_alerts({"A1": 101.0, "B1": 92.0, "A3": 40.0})
    assert [(x["element"], x["level"]) for x in lines] == [("A1", "violation"), ("B1", "approaching")]


def test_confidence(rm):
    err = forecast_errors()
    assert err["T1"] == pytest.approx(0.1999)
    calm = rm.confidence({t: 30.0 for t in err}, err)
    assert calm["probability"] > 0.99 and calm["label"] == "high"
    s3_after = {"T1": 28.9, "T2": 28.7, "T3": 30.4, "T4": 72.9, "T5": 41.4, "T6": 24.3}
    s3 = rm.confidence(s3_after, err)
    assert 0.6 < s3["probability"] < 0.9 and s3["label"] == "medium"
    at_limit = rm.confidence({"T1": 85.0}, err)
    assert at_limit["probability"] == pytest.approx(0.5) and at_limit["label"] == "low"
    assert rm.confidence({"T1": 50.0}, {}) is None
