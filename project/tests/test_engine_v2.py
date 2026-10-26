"""Engine v2: candidate records, rejection reasons, binding constraints, losses, P/Q/S, feeder KPIs,
power factor per category, radiality and MV transfers (gap items F6, F7, F8, B4, B5, B6)."""
import math
import shutil

import opendssdirect as dss
import pandas as pd
import pytest
import yaml

import engine as e
import run_results as rr
from core import config, scenarios
from engine import GridEngine


@pytest.fixture(scope="module")
def bench_loads():
    pred = rr.load_predictions()
    sc = {s.name.split()[0]: s for s in scenarios.benchmark()}

    def loads(name, ts):
        ts = pd.Timestamp(ts)
        return {tx: float(sc[name].adjust(tx, ts, pred.loc[ts, tx])) for tx in e.TX}
    return loads


# ------------------------------------------------------------------ candidates, plan, losses
def test_rerouted_result_is_fully_described(bench_loads):
    """S3 worst hour: one LV move fixes T4. Every candidate has a status and reason; the plan is ordered."""
    r = e.evaluate_and_reroute(bench_loads("S3", "2020-02-29 21:00"))
    assert r["status"] == "rerouted" and r["target"] == ["T4>T5"]
    assert [c["status"] for c in r["candidates"]].count("selected") == 1
    for c in r["candidates"]:
        assert c["status"] in ("selected", "feasible", "rejected") and c["reason"]
    assert [(p["switch_id"], p["operation"]) for p in r["plan"]] == [("SW-004", "open"), ("SW-010", "close")]
    assert r["after"]["losses_kw"] < r["before"]["losses_kw"]
    assert r["binding"] == []


def test_no_action_repeated_across_rounds(bench_loads):
    """Regression: an action applied in round 1 must not be tried again in round 2."""
    r = e.evaluate_and_reroute(bench_loads("S4", "2020-02-14 10:00"))
    assert r["status"] == "rerouted"
    assert len(r["target"]) == len(set(r["target"]))
    round1 = next(c["actions"] for c in r["candidates"] if c["status"] == "selected" and c["round"] == 1)
    for c in r["candidates"]:
        if c["round"] == 2:
            assert not set(round1) & set(c["actions"])


def test_binding_constraint_explains_s5(bench_loads):
    r = e.evaluate_and_reroute(bench_loads("S5", "2020-02-17 07:00"))
    assert r["status"] == "no_safe_option"
    assert r["binding"][0] == ("T3 (200.6%): even with its block b moved to T6 it would be at 161.8%, above the "
                               "100% limit (only 20% of its load is movable).")
    assert any("bus L3B is at 0.9405 pu" in b for b in r["binding"])
    assert "would still be at" in r["candidates"][0]["reason"]          # T3 was over the limit before switching


def test_binding_constraint_explains_s4(bench_loads):
    r = e.evaluate_and_reroute(bench_loads("S4", "2020-02-21 06:00"))
    assert r["binding"][0].startswith("T4 (143.8%): even with its block b moved to T5 it would be at 114.2%")
    assert any(b.startswith("T1 (89.1%) could be relieved") and "while T4 stays above" in b for b in r["binding"])


def test_tie_break_prefers_lower_losses():
    g = e._default
    best = {"max_loading": 80.0, "losses_kw": 9.0}
    assert g._better({"max_loading": 80.0, "losses_kw": 8.0}, best)             # equal peak, lower losses
    assert not g._better({"max_loading": 80.0, "losses_kw": 9.5}, best)
    assert g._better({"max_loading": 79.0, "losses_kw": 50.0}, best)            # lower peak wins regardless
    assert not g._better({"max_loading": 80.5, "losses_kw": 1.0}, best)


# ------------------------------------------------------------------ P, Q, S, feeders, power factor
def test_transformer_power_is_consistent(base_loads):
    r = e.evaluate_configuration(base_loads)
    for t, x in r["metrics"]["transformers"].items():
        assert x["kva"] == pytest.approx(math.hypot(x["kw"], x["kvar"]), abs=0.15)
        assert x["loading_pct"] == pytest.approx(100 * x["kva"] / 500, abs=0.05)
        assert x["kw"] > 300 * 0.999                         # load plus the transformer's own losses


def test_feeder_kpis(base_loads):
    m = e.evaluate_configuration(base_loads)["metrics"]
    fa, fb = m["feeders"]["A"], m["feeders"]["B"]
    assert fa["id"] == "FD-001" and fa["transformers"] == ["T1", "T2", "T3"]
    tx_kw = sum(m["transformers"][t]["kw"] for t in fa["transformers"])
    assert fa["kw"] == pytest.approx(tx_kw, rel=0.01)          # head power = transformers + line losses
    assert 0 < fa["loading_pct"] < 100 and fa["min_voltage_pu"] < 1
    assert m["losses_kw"] > 0


def test_power_factor_comes_from_the_load_category(tmp_path, monkeypatch, base_loads):
    shutil.copytree(config.config_dir(), tmp_path / "config")
    path = tmp_path / "config" / "grid.yaml"
    g = yaml.safe_load(path.read_text())
    g["load_categories"]["residential"]["power_factor"] = 0.90
    path.write_text(yaml.safe_dump(g, sort_keys=False))
    monkeypatch.setenv("LFDR_CONFIG_DIR", str(tmp_path / "config"))
    config.reload()
    try:
        eng = GridEngine()
        eng.load_network()
        dss.Loads.Name("LD1a")
        assert dss.Loads.PF() == pytest.approx(0.90)
        low_pf = eng.evaluate_configuration(base_loads)["metrics"]["transformers"]["T1"]
    finally:
        monkeypatch.delenv("LFDR_CONFIG_DIR")
        config.reload()
    normal = e.evaluate_configuration(base_loads)["metrics"]["transformers"]["T1"]
    assert low_pf["kvar"] > normal["kvar"] and low_pf["loading_pct"] > normal["loading_pct"]


# ------------------------------------------------------------------ radiality and switching plans
def test_radiality():
    g = e._default
    assert g.check_radial([]) == (True, "")
    assert g.check_radial(["T1>T2", "T3>T6"])[0]
    ok, why = g.check_radial(["MV1", "MV2"])
    assert not ok and "unsupplied" in why and "LD3A" in why
    loop = {**g.normal_state, "SW1A": "closed"}                    # home AND alt closed
    ok, why = g.check_radial_state(loop)
    assert not ok and why.startswith("would close a loop")
    ok, why = g.check_radial_state({**g.normal_state, "SW1H": "open"})   # home opened, alt not closed
    assert not ok and "LD1B" in why


def test_switch_plan_order():
    back = e.switch_plan(["T4>T5"], [])
    assert [(s["switch"], s["operation"]) for s in back] == [("SW4A", "open"), ("SW4H", "close")]
    both = e.switch_plan([], ["T1>T2", "MV1"])
    assert [(s["switch_id"], s["operation"]) for s in both] == [("SW-001", "open"), ("SW-007", "close"),
                                                                 ("SW-014", "open"), ("SW-013", "close")]


def test_non_radial_configuration_is_refused(base_loads):
    r = e.evaluate_configuration(base_loads, ["MV1", "MV2"])
    assert not r["ok"] and not r["radial"] and r["metrics"] is None
    assert e.evaluate_state(base_loads, ["MV1", "MV2"]) is None


# ------------------------------------------------------------------ MV transfers and line limits
@pytest.fixture
def weak_feeder_engine(tmp_path, monkeypatch, project_root):
    """A copy of the network whose feeder-A head cable is rated 60 A instead of 220 A."""
    cfg = tmp_path / "config"
    shutil.copytree(config.config_dir(), cfg)
    dss_text = (project_root / "grid" / "network.dss").read_text().replace(
        "New Line.A1 bus1=SUB bus2=A1 linecode=LC11 length=1.0 units=km",
        "New Line.A1 bus1=SUB bus2=A1 linecode=LC11 length=1.0 units=km normamps=60 emergamps=70")
    (cfg / "weak.dss").write_text(dss_text)
    g = yaml.safe_load((cfg / "grid.yaml").read_text())
    g["network_file"] = str(cfg / "weak.dss")
    (cfg / "grid.yaml").write_text(yaml.safe_dump(g, sort_keys=False))
    monkeypatch.setenv("LFDR_CONFIG_DIR", str(cfg))
    config.reload()
    yield GridEngine()
    monkeypatch.delenv("LFDR_CONFIG_DIR")
    config.reload()


def feeder_a_heavy(kw):
    return {"T1": kw, "T2": kw, "T3": kw, "T4": 250.0, "T5": 250.0, "T6": 250.0}


def test_line_overload_triggers_rerouting(weak_feeder_engine):
    """No transformer is above 85%, but cable A1 is at 101.9%: the engine must act (it used to ignore this)."""
    r = weak_feeder_engine.evaluate_and_reroute(feeder_a_heavy(360.0))
    assert r["overloaded"] == []
    assert [v["constraint"] for v in r["violations_before"]] == ["line"]
    assert r["status"] == "rerouted" and r["target"] == ["T3>T6"]
    assert r["after"]["line_loading_pct"]["A1"] < 100


def test_only_an_mv_transfer_can_fix_it(weak_feeder_engine):
    loads = feeder_a_heavy(390.0)
    without = weak_feeder_engine.evaluate_and_reroute(loads, use_mv=False)
    assert without["status"] == "no_safe_option"
    assert without["binding"] == ["line A1 is at 110.6% of its rating (> 100%), and no switching option removes it."]
    with_mv = weak_feeder_engine.evaluate_and_reroute(loads, use_mv=True)
    assert with_mv["status"] == "rerouted" and with_mv["target"] == ["MV1"]
    assert [(p["switch"], p["operation"]) for p in with_mv["plan"]] == [("A2", "open"), ("TIE1", "close")]
    after = with_mv["after"]
    assert after["line_loading_pct"]["A1"] < 40
    assert after["feeders"]["A"]["transformers"] == ["T1"]
    assert after["feeders"]["B"]["transformers"] == ["T2", "T3", "T4", "T5", "T6"]


def test_mv_transfers_are_off_for_the_published_benchmark():
    assert config.operations()["rerouting"]["use_mv_transfers"] is False
