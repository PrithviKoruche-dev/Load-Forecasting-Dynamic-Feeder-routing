"""The OpenDSS network model: structure, default switch state and the normal operating point."""
import opendssdirect as dss
import pytest

import engine as e


@pytest.fixture
def solved_base(base_loads):
    e.load_network()
    e.set_loads(base_loads)
    assert e.solve(), "base case must converge"
    return e.full_metrics()


def is_open(line):
    dss.Circuit.SetActiveElement(f"Line.{line}")
    return dss.CktElement.IsOpen(1, 0)


def test_six_transformers_rated_500_kva(solved_base):
    names = [n.upper() for n in dss.Transformers.AllNames()]
    assert names == e.TX
    for n in names:
        dss.Transformers.Name(n)
        assert dss.Transformers.kVA() == pytest.approx(500.0)


def test_default_switch_state(solved_base):
    """Home switches closed, alternate switches and the MV tie open: the network starts radial."""
    for tx, (home, alt, _) in e.MOVES.items():
        assert not is_open(home), f"{home} should be closed"
        assert is_open(alt), f"{alt} should be open"
    assert is_open("TIE1")


def test_base_case_operating_point(solved_base):
    m = solved_base
    for tx, pct in m["loading_by_tx"].items():
        assert pct == pytest.approx(64.1, abs=0.1), tx
    assert e.V_MIN <= m["min_voltage_pu"] and m["max_voltage_pu"] <= e.V_MAX
    assert m["max_line_loading"] < e.LINE_LIMIT
    assert e.is_feasible(m)


def test_load_split_80_20(base_loads):
    e.load_network()
    e.set_loads({**base_loads, "T1": 400.0})
    dss.Loads.Name("LD1a")
    assert dss.Loads.kW() == pytest.approx(320.0)
    dss.Loads.Name("LD1b")
    assert dss.Loads.kW() == pytest.approx(80.0)


@pytest.mark.parametrize("mult, expected", [(1.0, 64.1), (1.3, 83.7), (1.5, 96.9), (1.6, 103.5)])
def test_loading_scales_with_load(base_loads, mult, expected):
    """Same numbers as grid/test_day1.py."""
    e.load_network()
    e.set_loads({t: kw * mult for t, kw in base_loads.items()})
    assert e.solve()
    assert e.metrics()["max_loading"] == pytest.approx(expected, abs=0.1)


def test_voltage_check_covers_every_bus_including_lv(base_loads):
    """Regression: voltages used to be read by zipping 19 bus names with 57 per-phase values, so only the
    11 kV buses were checked. The 0.415 kV customer-side buses must be included."""
    e.load_network()
    e.set_loads({**base_loads, "T3": 1200.0})
    assert e.solve()
    v = e.bus_voltages_pu()
    assert set(v) == {b.lower() for b in dss.Circuit.AllBusNames()}
    lv = {b: x for b, x in v.items() if b.startswith("l")}
    mv = {b: x for b, x in v.items() if not b.startswith("l") and b != "sub"}
    assert min(lv.values()) < min(mv.values())          # the drop across the transformer
    assert min(lv, key=lv.get) == "l3b"                 # T3's movable block is the lowest point
    assert e.metrics()["min_voltage_pu"] == pytest.approx(0.9222, abs=1e-4)
    assert not e.is_feasible(e.full_metrics())          # below 0.95 pu: must be rejected
