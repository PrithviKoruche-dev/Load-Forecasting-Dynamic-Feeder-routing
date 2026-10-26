"""The rerouting engine: return contract and behaviour in each of its three outcomes."""
import pytest

import engine as e

RESULT_KEYS = {"status", "action", "before", "after", "log", "compute_s"}
PACK_KEYS = {"max_loading", "min_voltage_pu", "max_line_loading", "loading_by_tx"}


def test_no_action_when_below_threshold(base_loads):
    r = e.evaluate_and_reroute(base_loads)
    assert RESULT_KEYS <= r.keys()
    assert r["status"] == "no_action"
    assert r["action"] == []
    assert PACK_KEYS <= r["before"].keys()
    assert r["before"] == r["after"]


def test_single_move_resolves_t1_overload(base_loads):
    """Same case as grid/test_day1.py: T1 at 420 kW, moving its block b to T2 fixes it."""
    r = e.evaluate_and_reroute({**base_loads, "T1": 420.0})
    assert r["status"] == "rerouted"
    assert r["action"] == ["move T1 block b -> T2"]
    assert r["overloaded"] == ["T1"]
    assert r["before"]["loading_by_tx"]["T1"] == pytest.approx(90.3, abs=0.1)
    assert r["after"]["loading_by_tx"]["T1"] == pytest.approx(71.9, abs=0.1)
    assert r["after"]["loading_by_tx"]["T2"] == pytest.approx(82.4, abs=0.1)
    assert r["after"]["max_loading"] <= e.THRESHOLD


def test_threshold_is_respected(base_loads):
    """With a 95% threshold the same 90% overload needs no action."""
    r = e.evaluate_and_reroute({**base_loads, "T1": 420.0}, threshold=95.0)
    assert r["status"] == "no_action"


def test_no_safe_option_restores_switches(base_loads):
    """T3 x4: no move is feasible, so nothing may change and the engine must say so."""
    r = e.evaluate_and_reroute({**base_loads, "T3": 1200.0})
    assert r["status"] == "no_safe_option"
    assert r["action"] == []
    assert any("CRITICAL" in line for line in r["log"])
    # the live network must be back in the original configuration
    after_call = e.metrics()["loading_by_tx"]
    for tx, pct in r["before"]["loading_by_tx"].items():
        assert after_call[tx] == pytest.approx(pct, abs=0.05)


def test_unsafe_voltage_is_never_accepted():
    """An absurd load converges at about 0.77 pu; the engine must refuse, not report success."""
    r = e.evaluate_and_reroute({t: 1e6 for t in e.TX})
    assert r["status"] == "no_safe_option"


def test_safety_limits_are_the_documented_values():
    """The limits quoted in the README and report. Changing one must be a deliberate decision."""
    assert e.THRESHOLD == 85.0
    assert e.HARD_LIMIT == 100.0
    assert (e.V_MIN, e.V_MAX) == (0.95, 1.05)
    assert e.LINE_LIMIT == 100.0
    assert e.MOVABLE_FRACTION == 0.20


OK_STATE = {"min_voltage_pu": 0.99, "max_voltage_pu": 1.0, "max_line_loading": 50.0, "max_loading": 80.0}


@pytest.mark.parametrize("field, bad_value", [
    ("min_voltage_pu", 0.949),       # under-voltage
    ("max_voltage_pu", 1.051),       # over-voltage
    ("max_line_loading", 100.1),     # line thermal limit
    ("max_loading", 100.1),          # transformer hard limit
])
def test_each_constraint_alone_rejects_a_state(field, bad_value):
    assert e.is_feasible(OK_STATE)
    assert not e.is_feasible({**OK_STATE, field: bad_value})


def test_evaluate_state_matches_manual_move(base_loads):
    m = e.evaluate_state({**base_loads, "T1": 420.0}, ["T1"])
    assert m["loading_by_tx"]["T1"] == pytest.approx(71.9, abs=0.1)
    assert m["loading_by_tx"]["T2"] == pytest.approx(82.4, abs=0.1)


def test_every_move_targets_a_different_transformer():
    for tx, (home, alt, target) in e.MOVES.items():
        assert target in e.TX and target != tx
        assert home.startswith("SW") and home.endswith("H")
        assert alt.startswith("SW") and alt.endswith("A")
