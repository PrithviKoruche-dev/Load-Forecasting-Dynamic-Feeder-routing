"""Phase 7: the synthetic city (gap items A1, A2, A3, A5, C1, C4) - generator, engine on a large multi-
substation network, demand model (calendar, seasons, weather, events), residential calibration, persistence
forecasts, and the service and dashboard running the city."""
import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from core import config

C1, C2, C3, C4 = ("C1 Normal year", "C2 May heatwave", "C3 EV adoption (15% of households)",
                  "C4 Heatwave and EV adoption")


@pytest.fixture
def city(project_root, monkeypatch):
    """The demo city's configuration for the duration of a test (the classic one afterwards)."""
    monkeypatch.setenv("LFDR_CONFIG_DIR", str(project_root / "cities" / "demo" / "config"))
    config.reload()
    yield project_root / "cities" / "demo"
    monkeypatch.delenv("LFDR_CONFIG_DIR")
    config.reload()


@pytest.fixture(scope="module")
def demand(project_root):
    import sys
    sys.path.insert(0, str(project_root / "city"))
    from demand import CityDemand
    return CityDemand("demo")


@pytest.fixture
def engine(city):
    import engine as e
    return e.GridEngine()


# ------------------------------------------------------------------ generator (A1, A3, A5)
def test_generator_reproduces_the_committed_city(project_root, tmp_path):
    import generate
    src = project_root / "cities" / "demo"
    dst = tmp_path / "demo"
    dst.mkdir()
    shutil.copy(src / "city.yaml", dst / "city.yaml")
    generate.generate(dst)
    assert (dst / "network.dss").read_text() == (src / "network.dss").read_text()
    assert (dst / "consumers.csv").read_bytes() == (src / "consumers.csv").read_bytes()
    for f in ("grid", "operations", "scenarios", "forecasting", "storage", "simulation"):
        a = [ln for ln in (dst / "config" / f"{f}.yaml").read_text().splitlines() if not ln.startswith("network_file")]
        b = [ln for ln in (src / "config" / f"{f}.yaml").read_text().splitlines() if not ln.startswith("network_file")]
        assert a == b, f


def test_the_city_has_the_specified_size_and_unique_ids(city):
    g = config.grid()
    subs = config.substations(g)
    assert len(subs) == 3 and 6 <= len(g["feeders"]) <= 9 and 30 <= len(g["transformers"]) <= 50
    assert {config.feeder_substation(g, f) for f in g["feeders"]} == set(subs)
    from core import ids
    reg = ids.registry()
    assert reg["asset_id"].is_unique and set(reg["kind"]) == {"substation", "feeder", "transformer", "switch"}
    c = pd.read_csv(city / "consumers.csv")
    assert c["meter_id"].is_unique and c["consumer_id"].is_unique
    assert c["meter_id"].str.fullmatch(r"MTR-\d{5}").all()
    assert set(c["transformer"]) == set(g["transformers"])                  # every transformer has consumers
    assert set(c["type"]) == {"residential", "commercial", "industrial", "mixed"}
    assert set(c["transformer_asset_id"]) == {t["id"] for t in g["transformers"].values()}


def test_transformers_are_sized_to_their_utilisation(city, demand):
    """Normal year, before noise: each transformer's peak is the drawn share of its rating (50-80%)."""
    import generate
    g = config.grid()
    spec = demand.spec
    units, _, _ = demand.units(())
    U = np.vstack([units[k] for k in ("residential", "commercial", "industrial", "mixed")])
    smooth = pd.DataFrame((demand.sizes.to_numpy() @ U).T, columns=demand.transformers)
    pf = spec["consumers"]["power_factor"]
    util = smooth.max() / pd.Series({t: x["rating_kva"] * pf[x["category"]] for t, x in g["transformers"].items()})
    assert util.between(0.40, 0.92).all()                                   # 0.5-0.8 drawn, +/- consumer rounding
    assert generate.HEADER.startswith("# Generated")


# ------------------------------------------------------------------ engine on a large network
def test_radiality_with_several_substations(engine):
    ok, why = engine.check_radial([])
    assert ok
    g = config.grid()
    feeder_of = {b: f for f, fd in g["feeders"].items() for b in fd["buses"]}
    sub = lambda bus: config.feeder_substation(g, feeder_of[bus])  # noqa: E731
    ties = {n: s for n, s in g["switches"].items() if s["kind"] == "mv_tie"}
    within = next(n for n, s in ties.items() if sub(s["buses"][0]) == sub(s["buses"][1]))
    across = next(n for n, s in ties.items() if sub(s["buses"][0]) != sub(s["buses"][1]))
    ok, why = engine.check_radial_state({**engine.switch_state([]), within: "closed"})
    assert not ok and "loop" in why                                          # two feeders of one substation
    ok, why = engine.check_radial_state({**engine.switch_state([]), across: "closed"})
    assert not ok and "in parallel" in why                                   # two substations through a tie
    for mv in config.grid()["mv_transfers"]:
        assert engine.check_radial([mv])[0], mv


def test_an_mv_transfer_moves_the_far_end_to_the_other_feeder(engine):
    g = config.grid()
    name, mv = next(iter(g["mv_transfers"].items()))
    far, other = mv["feeders"]
    loads = {t: 0.4 * x["rating_kva"] for t, x in g["transformers"].items()}
    before = engine.evaluate_configuration(loads, [])["metrics"]["feeders"]
    after = engine.evaluate_configuration(loads, [name])["metrics"]["feeders"]
    moved = set(before[far]["transformers"]) - set(after[far]["transformers"])
    assert moved and moved <= set(after[other]["transformers"])
    assert after[other]["current_a"] > before[other]["current_a"]


def test_rerouting_on_the_city_and_relevant_scope_matches_all(engine, demand):
    g = config.grid()
    m = demand.matrix(["may_heatwave", "ev_adoption"])
    pf = {t: g["load_categories"][x["category"]]["power_factor"] for t, x in g["transformers"].items()}
    pct = m / pd.Series({t: x["rating_kva"] * pf[t] for t, x in g["transformers"].items()}) * 100
    hot = pct.max(axis=1).sort_values(ascending=False).index[::40][:5]     # five busy steps
    ops = copy.deepcopy(config.operations())
    ops["rerouting"]["candidate_scope"] = "all"
    import engine as e
    full = e.GridEngine(ops_cfg=ops)
    assert engine.candidate_scope == "relevant"
    statuses = set()
    for t in hot:
        loads = m.loc[t].to_dict()
        a, b = engine.evaluate_and_reroute(loads), full.evaluate_and_reroute(loads)
        statuses.add(a["status"])
        assert (a["status"], sorted(a["target"]), (a["after"] or {}).get("max_loading")) == \
            (b["status"], sorted(b["target"]), (b["after"] or {}).get("max_loading")), t
        assert len(a["candidates"]) <= len(b["candidates"])
    assert statuses & {"rerouted", "no_safe_option"}


# ------------------------------------------------------------------ demand model (A2, C1, C4)
def test_calendar_seasons_and_holidays(demand):
    cal = demand.cal
    at = lambda s: demand.times.get_loc(pd.Timestamp(s))  # noqa: E731
    assert cal.season[at("2023-01-15 12:00")] == "winter" and cal.season[at("2023-07-15 12:00")] == "monsoon"
    assert cal.daytype[at("2023-08-15 12:00")] == "sunday"                  # Independence Day (a Tuesday)
    assert cal.daytype[at("2023-08-19 12:00")] == "saturday" and cal.daytype[at("2023-08-16 12:00")] == "weekday"
    units, _, _ = demand.units(())
    c = demand.spec["demand"]["commercial"]
    night = lambda d: units["commercial"][at(f"{d} 02:00")]                 # closed: no cooling term  # noqa: E731
    assert night("2023-08-15") == pytest.approx(night("2023-08-16") * c["sunday_factor"])


def test_events_change_demand_only_where_and_when_they_apply(demand):
    base = demand.matrix([])
    heat = demand.matrix(["may_heatwave"])
    ev = demand.matrix(["ev_adoption"])
    in_heat = (base.index >= "2023-05-08") & (base.index < "2023-05-17")
    assert (heat[~in_heat] == base[~in_heat]).all().all()                  # same noise and weather elsewhere
    assert (heat[in_heat] >= base[in_heat] - 1e-9).all().all() and (heat[in_heat] > base[in_heat]).any().any()
    morning = base.index.hour == 10
    assert np.allclose(ev[morning], base[morning], atol=1e-6)                # cars charge in the evening
    evening = base.index.hour == 20
    assert (ev[evening].sum(axis=1) > base[evening].sum(axis=1) * 1.05).all()
    from demand import ev_profile                                           # exactly share x households charge
    e_ = demand.spec["events"]["ev_adoption"]
    c = demand.consumers
    cars = c[c["type"].isin(e_["consumers"]) & (c["ev_rank"] < e_["share"])].groupby("transformer").size()
    cars = cars.reindex(demand.transformers, fill_value=0).to_numpy()
    assert 0.10 < cars.sum() / c["type"].isin(e_["consumers"]).sum() < 0.20
    i = demand.times.get_loc(pd.Timestamp("2023-06-14 20:00"))
    expected = cars * ev_profile(e_, demand.cal)[i] * (1 + demand.noise[i])
    assert np.allclose(ev.iloc[i] - base.iloc[i], expected)
    assert demand.temperature(["may_heatwave"])["2023-05-10"].mean() == pytest.approx(
        demand.temperature([])["2023-05-10"].mean() + demand.spec["events"]["may_heatwave"]["extra_c"])


def test_always_on_festival_and_determinism(demand, project_root):
    import sys
    from demand import CityDemand
    spec = copy.deepcopy(demand.spec)
    without = CityDemand("demo")
    without.spec["events"].pop("diwali_lights")
    a, _, _ = demand.units(())
    b, _, _ = without.units(())
    i = demand.times.get_loc(pd.Timestamp("2023-11-11 19:00"))
    assert a["residential"][i] == pytest.approx(1.25 * b["residential"][i])
    j = demand.times.get_loc(pd.Timestamp("2023-11-11 10:00"))
    assert a["residential"][j] == pytest.approx(b["residential"][j])
    assert CityDemand("demo").matrix([]).equals(demand.matrix([]))         # same seed, same city
    assert spec == demand.spec and sys.modules["demand"]


# ------------------------------------------------------------------ calibration
def test_calibration_recovers_known_coefficients():
    import calibrate
    ts = pd.date_range("2019-06-01", "2020-03-24 23:00", freq="1h")
    rng = np.random.default_rng(1)
    temp = 24 + 8 * np.sin(np.arange(len(ts)) / 500.0) + rng.normal(0, 1, len(ts))
    base = 0.2 + 0.05 * np.cos(2 * np.pi * ts.hour / 24)
    kw = base + 0.01 * np.clip(temp - 24, 0, None) + rng.normal(0, 0.005, len(ts))
    df = pd.DataFrame({"timestamp": ts, "transformer_id": "T1", "n_households": 10, "load_kw_raw": 10 * kw,
                       "temp_c": temp})
    table, stats = calibrate.fit(df)
    cool = table[table["kind"] == "cooling_kw_per_c"]["value"]
    assert cool.mean() == pytest.approx(0.01, abs=0.001) and stats["r2"] > 0.8


def test_committed_profile_matches_the_eMARC_fit(project_root):
    path = project_root / "data" / "processed" / "clusters.parquet"
    if not path.exists():
        pytest.skip("needs data/processed/clusters.parquet (the eMARC data)")
    import calibrate
    table, _ = calibrate.fit(pd.read_parquet(path))
    committed = pd.read_csv(project_root / "city" / "profiles" / "residential_emarc.csv").fillna({"season": "", "daytype": ""})
    pd.testing.assert_frame_equal(table.reset_index(drop=True), committed, check_dtype=False, atol=1e-5)


# ------------------------------------------------------------------ persistence forecasts
@pytest.fixture
def city_store(city, tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'city.db').as_posix()}"
    monkeypatch.setenv("LFDR_DB_URL", url)
    from orchestrator import prepare_database
    from storage import Store
    s = Store(url)
    prepare_database(s)
    return s


def test_persistence_forecast_and_its_band_use_only_known_loads(city_store):
    from sources import CitySource, PersistenceForecasts
    from orchestrator import benchmark_scenario
    src = CitySource(city_store, "demo", benchmark_scenario(C1))
    f = PersistenceForecasts(city_store, src)
    t = pd.Timestamp("2023-05-10 19:00")
    assert f.point(t) == src.measured(t - pd.Timedelta(hours=24))
    up = f.upper(t)
    assert all(up[k] >= f.point(t)[k] for k in up)
    src.table = src.table.copy()
    src.table.loc[src.table.index > t - pd.Timedelta(hours=24)] *= 3          # everything after issue time
    f._key = None
    assert f.upper(t) == up and f.point(t) == src.measured(t - pd.Timedelta(hours=24))
    assert city_store.meters()["consumer_type"].notna().all() and len(city_store.meters()) == len(src.demand.consumers)


# ------------------------------------------------------------------ the city in the loop
class FakeWall:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_the_service_runs_the_city(city_store, city):
    import engine as e
    from orchestrator import Orchestrator
    from service import Service
    w = FakeWall()
    orch = Orchestrator(city_store, engine=e.GridEngine(), scenario=C4)
    svc = Service(city_store, name="service", orchestrator=orch, wall=w, sleep=w.sleep)
    assert svc.clock.step == pd.Timedelta(minutes=15)
    for cmd, args in (("mode", {"mode": "auto"}), ("jump", {"time": "2023-05-09 18:00"})):
        svc.execute(cmd, args)
    for _ in range(8):
        svc.step_once()
    steps = city_store.steps("service")
    assert len(steps) == 8 and (pd.to_datetime(steps["sim_time"]).diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert len(steps.iloc[-1]["measured_pct"]) == len(config.grid()["transformers"])
    with city_store.engine.connect() as c:
        n_meas = c.exec_driver_sql("select count(*) from measurements where source = 'city-meters'").scalar_one()
        n_fc = c.exec_driver_sql("select count(*) from forecasts where model_version = 'persistence-24h'").scalar_one()
        first = c.exec_driver_sql("select min(target_time) from forecasts where model_version = 'persistence-24h'").scalar_one()
    assert str(first).startswith("2023-05-10 18:00")                        # issued at 18:00 on the 9th, 24 h ahead
    n = len(config.grid()["transformers"])
    assert n_meas == 8 * n and n_fc == 8 * n                                # readings arrive; forecasts issued 24 h ahead
    last = steps.iloc[-1]
    normal = {k: v["normal"] for k, v in config.grid()["switches"].items()}
    assert all(normal[k] != v for k, v in (last["switches_after"] or {}).items())   # only switches away from normal
    import common
    assert len(common.switch_states(last["switches_after"])) == len(normal)
    svc.execute("scenario", {"name": C1})
    assert orch.demand.events == () and orch.scenario.name == C1


def test_retention_is_applied_once_per_simulated_day(city_store, city, monkeypatch):
    import engine as e
    from orchestrator import Orchestrator
    from service import Service
    city_store.retention = {"measurements_days": 1, "alerts_days": 1, "forecasts_days": 1}
    w = FakeWall()
    svc = Service(city_store, name="service", orchestrator=Orchestrator(city_store, engine=e.GridEngine(), scenario=C1),
                  wall=w, sleep=w.sleep)
    calls = []
    real = city_store.prune
    monkeypatch.setattr(city_store, "prune", lambda t: calls.append(pd.Timestamp(t)) or real(t))
    svc.execute("jump", {"time": "2023-03-01 23:00"})
    for _ in range(8):                                                     # 23:00 .. 00:45: one day boundary
        svc.step_once()
    assert calls == [pd.Timestamp("2023-03-01 23:45")]
    city_store.write_forecasts([("2023-02-20 10:00", "DT-001", 1.0)], "persistence-24h")
    city_store.write_quantiles([("2023-02-20 10:00", "DT-001", 0.9, 1.2)], "persistence-24h", "test")
    out = city_store.prune("2023-03-02 00:00")
    assert out["forecasts"] >= 1 and out["forecast_quantiles"] >= 1


def test_dashboard_pages_show_the_city(city_store, city, project_root):
    from streamlit.testing.v1 import AppTest
    import engine as e
    from orchestrator import Orchestrator
    from service import Service
    w = FakeWall()
    svc = Service(city_store, name="service", orchestrator=Orchestrator(city_store, engine=e.GridEngine(), scenario=C2),
                  wall=w, sleep=w.sleep)
    svc.execute("jump", {"time": "2023-05-09 15:00"})
    for _ in range(3):
        svc.step_once()
    svc.save_state()
    at = AppTest.from_file(str(project_root / "app" / "views" / "overview.py"), default_timeout=90)
    at.run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert (m["Substations"], m["Feeders"], m["Transformers"]) == ("3", str(len(config.grid()["feeders"])),
                                                                   str(len(config.grid()["transformers"])))
    assert m["Smart meters"] == f"{len(svc.orch.demand.demand.consumers):,}"
    at = AppTest.from_file(str(project_root / "app" / "views" / "network.py"), default_timeout=90)
    at.run()
    assert not at.exception
    import json
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    tx = next(tr for tr in spec["data"] if tr.get("name") == "transformers")
    assert len(tx["customdata"]) == len(config.grid()["transformers"])
    at = AppTest.from_file(str(project_root / "app" / "views" / "controls.py"), default_timeout=90)
    at.run()
    assert not at.exception
    assert next(s for s in at.selectbox if s.label == "Scenario").options == [C1, C2, C3, C4]


def test_use_city_refuses_an_unknown_city():
    with pytest.raises(config.ConfigError, match="no generated city"):
        config.use_city("atlantis")


@pytest.mark.slow
def test_closed_loop_on_the_city_reduces_measured_overloads(city):
    import closed_loop
    row, steps = closed_loop.run(C4, start="2023-05-09 00:00", end="2023-05-09 23:45")
    assert row["steps"] == 96 and row["switch_operations"] > 0
    assert row["measured_tx_hours_over_with_control"] <= row["measured_tx_hours_over_no_control"]
    assert 0 < row["band_coverage_%"] <= 100
