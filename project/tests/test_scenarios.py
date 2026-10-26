"""Scenario rules, and the guarantee that the dashboard and the benchmark use the same definitions."""
import itertools

import pandas as pd
import pytest

from core import scenarios
from core.scenarios import Rule, Scenario

TXS = ["T1", "T2", "T3", "T4", "T5", "T6"]
DAY = [pd.Timestamp("2020-02-29") + pd.Timedelta(hours=h) for h in range(24)]


def same_effect(a, b, base=300.0):
    """True if two scenarios change every transformer's load identically at every hour of a day."""
    return all(a.adjust(tx, ts, base) == b.adjust(tx, ts, base) for tx, ts in itertools.product(TXS, DAY))


# ------------------------------------------------------------------ rule semantics
def test_scale_and_add():
    assert Rule("scale", factor=1.5).apply(100.0) == 150.0
    assert Rule("add", kw=40).apply(100.0) == 140.0


def test_transformer_filter():
    r = Rule("scale", factor=2.0, transformers=("T1", "T3"))
    assert r.applies("T1") and r.applies("T3") and not r.applies("T2")


def test_hours_window_is_start_inclusive_end_exclusive():
    r = Rule("add", kw=10, hours=(18, 22))
    hits = [ts.hour for ts in DAY if r.applies("T4", ts)]
    assert hits == [18, 19, 20, 21]


def test_hour_rules_need_a_timestamp():
    with pytest.raises(ValueError, match="timestamp"):
        Rule("add", kw=10, hours=(18, 22)).applies("T4")


def test_rules_apply_in_order():
    sc = Scenario("x", (Rule("add", kw=100), Rule("scale", factor=2.0)))
    assert sc.adjust("T1", None, 100.0) == 400.0          # (100 + 100) * 2, not 100 * 2 + 100


def test_loads_returns_plain_floats():
    out = Scenario("x", (Rule("add", kw=1),)).loads({"T1": 1})
    assert out == {"T1": 2.0} and type(out["T1"]) is float


# ------------------------------------------------------------------ the configured scenarios
def test_benchmark_names_match_published_results(project_root):
    published = pd.read_csv(project_root / "docs" / "results.csv")["scenario"].tolist()
    assert [s.name for s in scenarios.benchmark()] == published


def test_dashboard_defaults_reproduce_the_benchmark():
    """The drift the gap analysis warned about: dashboard presets at their default slider values
    must give exactly the benchmark's S1-S4."""
    bench = {s.name.split()[0]: s for s in scenarios.benchmark()}
    presets = {p.label: p.build() for p in scenarios.dashboard_presets()}
    assert same_effect(presets["Normal"], bench["S1"])
    assert same_effect(presets["Heatwave (feeder A)"], bench["S2"])
    assert same_effect(presets["EV evening surge"], bench["S3"])
    assert same_effect(presets["Severe stress"], bench["S4"])


def test_dashboard_controls_change_the_scenario():
    ev = {p.label: p for p in scenarios.dashboard_presets()}["EV evening surge"]
    sc = ev.build(value=50, transformer="T5")
    at_19 = DAY[19]
    assert sc.adjust("T5", at_19, 300.0) == 350.0
    assert sc.adjust("T4", at_19, 300.0) == 300.0
    assert sc.adjust("T5", DAY[12], 300.0) == 300.0


def test_quick_scenarios():
    base_kw, quick = scenarios.quick()
    assert base_kw == 300
    by = {s.name.split()[0]: s for s in quick}
    assert by["S2"].adjust("T1", None, 300.0) == pytest.approx(420.0)
    assert by["S3"].adjust("T4", None, 300.0) == 450.0
    assert by["S4"].adjust("T6", None, 300.0) == pytest.approx(480.0)
    assert by["S5"].adjust("T3", None, 300.0) == 1200.0
