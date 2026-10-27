"""Phase 8 (gap item C5): measurement streams of the synthetic city - transformer meters and smart meters with
delays, losses, outages and faults; the head-end's estimates, quality flags and revisions; the operator's view
against the truth in the closed loop."""
import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from core import config

C1, C4 = "C1 Normal year", "C4 Heatwave and EV adoption"
MAY = pd.Timestamp("2023-05-08 00:00")


@pytest.fixture(scope="module")
def demand():
    from demand import CityDemand
    return CityDemand("demo")


@pytest.fixture(scope="module")
def model(demand):
    from meters import MeterModel
    return MeterModel(demand)


@pytest.fixture
def city(project_root, monkeypatch):
    monkeypatch.setenv("LFDR_CONFIG_DIR", str(project_root / "cities" / "demo" / "config"))
    config.reload()
    yield project_root / "cities" / "demo"
    monkeypatch.delenv("LFDR_CONFIG_DIR")
    config.reload()


@pytest.fixture
def city_store(city, tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'city.db').as_posix()}"
    monkeypatch.setenv("LFDR_DB_URL", url)
    from orchestrator import prepare_database
    from storage import Store
    s = Store(url)
    prepare_database(s)
    return s


def idx(demand, t):
    return demand.times.get_loc(pd.Timestamp(t))


def after(demand, ns):
    """Index of the first interval that starts after a time given in nanoseconds."""
    return int(demand.times.as_unit("ns").asi8.searchsorted(ns, side="right"))


# ------------------------------------------------------------------ settings
def test_meter_settings_default_to_perfect_measurement_and_are_checked(demand):
    from meters import settings
    spec = copy.deepcopy(demand.spec)
    assert settings(spec)["enabled"] is True and settings(spec)["store_readings"] == "exceptions"
    spec.pop("meters")
    assert settings(spec)["enabled"] is False                              # no section: the operator sees the truth
    for path, value, msg in ((("smart", "lost"), 1.5, "smart.lost"), (("lv_losses",), -0.1, "lv_losses"),
                             (("smart", "stuck_after"), 1, "stuck_after"), (("store_readings",), "some", "store_readings"),
                             (("head_end", "estimated_share"), 0.99, "not be above good_share"),
                             (("smart", "delay_s", "median"), 0, "median")):
        bad = copy.deepcopy(demand.spec)
        node = bad["meters"]
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        with pytest.raises(ValueError, match=msg):
            settings(bad)
    bad = copy.deepcopy(demand.spec)
    bad["meters"]["smart"]["typo"] = 1
    with pytest.raises(ValueError, match="unknown settings"):
        settings(bad)


# ------------------------------------------------------------------ the meter model
def test_readings_are_the_same_on_every_run(demand, model):
    from meters import MeterModel
    i = idx(demand, MAY + pd.Timedelta(hours=19))
    a = model.readings(i, ())
    b = MeterModel(demand).readings(i, ())
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(model.readings(i + 1, ())[0], a[0])


def test_smart_meters_add_up_to_the_transformer_load_minus_losses(demand, model):
    """Over a day the readings of a transformer's consumers are its true load x (1 - lv_losses), on average."""
    i0 = idx(demand, MAY)
    true = demand.matrix(()).iloc[i0:i0 + 96].to_numpy()
    sums = np.array([np.bincount(model.tx_index, weights=model.readings(i, ())[0], minlength=true.shape[1])
                     for i in range(i0, i0 + 96)])
    ratio = sums.sum() / true.sum()
    assert ratio == pytest.approx(1 - model.cfg["lv_losses"], abs=0.01)
    per_tx = sums.sum(axis=0) / true.sum(axis=0)
    assert np.all(np.abs(per_tx - (1 - model.cfg["lv_losses"])) < 0.1)      # every transformer, not only the total


def test_ev_owners_charge_only_with_the_ev_event(demand, model):
    i = idx(demand, MAY + pd.Timedelta(hours=20))                             # evening charging
    without, with_ev = model.expected(i, ()), model.expected(i, ("ev_adoption",))
    owners = with_ev > without + 1e-9
    share = demand.spec["events"]["ev_adoption"]["share"]
    assert 0 < owners.mean() < 2 * share
    assert (model.ev_rank[owners] < share).all()


def test_delays_losses_and_concentrator_outages(demand, model):
    sm = model.cfg["smart"]
    i0 = idx(demand, MAY)
    delays, lost = [], 0
    for i in range(i0, i0 + 8):
        _, arrive = model.readings(i, ())
        end = (demand.times[i] + model.step).value
        ok = arrive >= 0
        lost += (~ok).sum()
        inside = np.zeros(model.n, dtype=bool)                                # skip readings held by an outage
        for k, wins in enumerate(model.outages):
            inside |= (model.tx_index == k) & any(a <= end + 3600e9 and b >= end for a, b in wins)
        delays += list((arrive[ok & ~inside] - end) / 1e9)
    assert np.median(delays) == pytest.approx(sm["delay_s"]["median"], rel=0.1)
    assert lost / (8 * model.n) == pytest.approx(sm["lost"], abs=0.005)
    # an outage: every reading of that transformer sent during it arrives after it ends
    k, (a, b) = next((k, w[0]) for k, w in enumerate(model.outages)
                     if w and w[0][0] > demand.times[200].value and w[0][1] - w[0][0] > 3600e9)
    i = after(demand, a)
    _, arrive = model.readings(i, ())
    sel = (model.tx_index == k) & (arrive >= 0)
    assert (demand.times[i] + model.step).value < b and (arrive[sel] >= b).all()


def test_transformer_meters_are_accurate_and_missing_while_out(demand, model):
    k, (a, b) = next((k, w[0]) for k, w in enumerate(model.dt_failures) if w and w[0][1] - w[0][0] > 3600e9)
    i = after(demand, a)
    true = demand.matrix(()).iloc[i].to_numpy()
    r = model.transformer_readings(i, true)
    assert np.isnan(r[k]) and np.isnan(r).sum() == model.dt_missing[i].sum()
    ok = ~np.isnan(r)
    assert np.all(np.abs(r[ok] / true[ok] - 1) < 5 * model.cfg["transformer"]["accuracy"])
    series = model.transformer_series(demand.matrix(()))
    assert np.allclose(series.iloc[i].to_numpy(), r, equal_nan=True)
    later = after(demand, b)
    assert np.isnan(model.transformer_readings(later, true)[k]) == model.dt_missing[later, k]
    assert 0.002 < model.dt_missing.mean() < 0.02                         # 2 failures of ~12 h a year: ~0.3%


def test_stuck_meters_repeat_one_value(demand, model):
    m = int(np.flatnonzero(model.stuck_from < len(demand.times) - 10)[0])
    j = int(model.stuck_from[m])
    vals = [model.readings(i, ())[0][m] for i in range(j, j + 4)]
    assert len(set(vals)) == 1
    other = int(np.flatnonzero(model.stuck_from > len(demand.times))[0])
    assert len({model.readings(i, ())[0][other] for i in range(j, j + 4)}) > 1


# ------------------------------------------------------------------ the head-end
def test_head_end_estimates_converge_and_have_quality_flags(demand, model):
    from meters import HeadEnd
    h = HeadEnd(model)
    i0 = idx(demand, MAY)
    h.rebuild(i0 - 1, ())
    first = None
    for i in range(i0, i0 + 12):
        _, est = h.step(i, ())
        if i == i0 + 1:
            first = est[i0]                                                 # interval i0, one interval later
    final = h.estimate(i0)                                                    # eleven intervals later
    true = demand.matrix(()).iloc[i0]
    shares_first = np.array([v[2] for v in first.values()])
    shares_final = np.array([v[2] for v in final.values()])
    assert (shares_final >= shares_first - 1e-12).all() and shares_final.mean() > 0.97
    q = [v[1] for v in final.values()]
    assert q.count("good") > len(q) / 2 and set(q) <= {"good", "estimated", "poor", "missing"}
    err = np.array([final[tx][0] / true[tx] - 1 for tx in final if final[tx][0] is not None])
    assert abs(err.mean()) < 0.03 and np.abs(err).max() < 0.35             # few consumers on some transformers


def test_quality_follows_the_share_that_has_reported(model):
    from meters import HeadEnd
    h = HeadEnd(model)
    n = len(model.tx_names)
    total = np.full(n, 100.0)
    got = np.zeros(n)
    got[:4] = [99, 80, 50, 0]
    kw = np.full(model.n, np.nan)
    w = np.zeros(model.n)
    for k in range(n):                                                        # one meter of weight 100 per transformer
        m = int(np.flatnonzero(model.tx_index == k)[0])
        w[m] = total[k]
        if k < 4 and got[k] > 0:
            other = np.flatnonzero(model.tx_index == k)[1]
            w[m], w[other] = got[k], total[k] - got[k]
            kw[m] = got[k] * 0.96
    h.agg[0] = {"kw": kw, "w": w}
    e = h.estimate(0)
    q = [e[tx][1] for tx in model.tx_names[:4]]
    assert q == ["good", "estimated", "poor", "missing"]
    assert e[model.tx_names[0]][0] == pytest.approx(100.0)                   # scaled up by share and losses


def test_rebuilding_gives_the_same_state_as_running(demand, model):
    from meters import HeadEnd
    i0 = idx(demand, MAY + pd.Timedelta(hours=6))
    a = HeadEnd(model)
    a.rebuild(i0 - 1, ())
    for i in range(i0, i0 + 5):
        a.step(i, ())
    b = HeadEnd(model)
    b.rebuild(i0 + 4, ())
    assert set(a.agg) == set(b.agg) and set(a.pending) == set(b.pending)
    for j in a.agg:
        assert np.array_equal(a.agg[j]["kw"], b.agg[j]["kw"], equal_nan=True) and a.estimate(j) == b.estimate(j)
    assert np.array_equal(a.suspect, b.suspect)
    c = HeadEnd(model)                                                        # a jump: step() rebuilds by itself
    c.rebuild(i0 - 50, ())
    c.step(i0 + 5, ())
    a.step(i0 + 5, ())
    assert np.array_equal(c.suspect, a.suspect) and set(c.agg) == set(a.agg)


def test_only_stuck_meters_are_flagged_and_left_out(demand, model):
    from meters import HeadEnd
    h = HeadEnd(model)
    i = idx(demand, pd.Timestamp("2023-08-01 00:00"))
    h.rebuild(i, ())
    flagged = np.flatnonzero(h.suspect)
    stuck = np.flatnonzero(model.stuck_from <= i)
    assert len(flagged) > 0 and set(flagged) <= set(stuck)                   # no false alarms
    m = int(flagged[0])
    j = next(j for j in sorted(h.agg) if not np.isnan(h.agg[j]["kw"][m]))
    tx = model.tx_names[model.tx_index[m]]
    share = h.estimate(j)[tx][2]
    h.suspect[m] = False
    assert h.estimate(j)[tx][2] > share                                       # its reading is left out


# ------------------------------------------------------------------ the city source and the database
def test_observe_uses_the_transformer_meter_else_the_smart_meters_else_the_forecast(city_store, monkeypatch):
    from orchestrator import benchmark_scenario
    from sources import CitySource
    src = CitySource(city_store, "demo", benchmark_scenario(C1))
    t = MAY + pd.Timedelta(hours=19)
    i = src.demand.times.get_loc(t)
    out = src.observe(t, {tx: 1.0 for tx in src.transformers})
    true = src.measured(t)
    assert set(out["quality"].values()) == {"measured"}
    assert all(abs(out["loads"][tx] / true[tx] - 1) < 0.05 for tx in true) and out["loads"] != true
    a, b, c = src.transformers[:3]
    k = [src.demand.transformers.index(x) for x in (a, b, c)]
    missing = src.meters.dt_missing.copy()
    missing[i + 1, k] = True                                                  # their transformer meters fail
    monkeypatch.setattr(src.meters, "dt_missing", missing)
    est = {a: (123.0, "estimated", 0.9), b: (50.0, "poor", 0.4), c: (None, "missing", 0.0)}
    monkeypatch.setattr(src.head, "estimate", lambda j: est if j == i else {})
    monkeypatch.setattr(src.head, "step", lambda j, ev: ([], {}))
    src.head.position = i
    out = src.observe(t + src.resolution, {tx: 7.0 for tx in src.transformers})
    assert out["quality"][a] == "ami-estimate" and out["loads"][a] == est[a][0]
    assert out["quality"][b] == "forecast" and out["loads"][b] == 7.0
    assert out["quality"][c] == "forecast"
    assert out["stats"]["dt_missing"] == 3
    m = city_store.measurements("dt-scada")
    assert len(m) == 2 * len(src.transformers) and m["kw"].isna().sum() == 3 and (m["quality"] == "missing").sum() == 3


def test_estimates_are_revised_and_exception_readings_stored(city_store):
    from orchestrator import benchmark_scenario
    from sources import CitySource
    src = CitySource(city_store, "demo", benchmark_scenario(C1))
    t = MAY + pd.Timedelta(hours=10)
    for k in range(6):
        src.observe(t + k * src.resolution)
    est = city_store.measurements("ami-estimate")
    first = est[est["time"] == t]
    assert len(first) == len(src.transformers)                               # one row per transformer, revised in place
    assert (first["quality"] == "good").mean() > 0.5
    r = city_store.meter_readings()
    end = r["interval_time"] + src.resolution
    late = (r["received_at"] - end).dt.total_seconds() > src.resolution.total_seconds()
    assert len(r) > 0 and (late | (r["flag"] == "suspect")).all()            # only the exceptions
    assert ((r["received_at"] - end).dt.total_seconds() - r["delay_s"]).abs().max() < 1
    city_store.upsert_measurements([{"time": t.to_pydatetime(), "asset_id": "DT-001", "source": "ami-estimate",
                                     "kw": 1.0, "quality": "poor"}])
    again = city_store.measurements("ami-estimate")
    row = again[(again["time"] == t) & (again["asset_id"] == "DT-001")]
    assert len(row) == 1 and row["kw"].iloc[0] == 1.0 and row["quality"].iloc[0] == "poor"


def test_perfect_measurement_writes_the_true_loads(city_store):
    from orchestrator import benchmark_scenario
    from sources import CitySource
    src = CitySource(city_store, "demo", benchmark_scenario(C1), meters=False)
    t = MAY + pd.Timedelta(hours=10)
    out = src.observe(t)
    assert out["stats"] is None and out["loads"] == src.measured(t) and src.history is src.table
    assert len(city_store.measurements("dt-scada")) == len(src.transformers)


def test_meter_readings_are_pruned(city_store):
    city_store.write_meter_readings([{"meter_id": "MTR-00001", "interval_time": pd.Timestamp("2023-03-01").to_pydatetime(),
                                      "kw": 1.0, "flag": "ok"},
                                     {"meter_id": "MTR-00001", "interval_time": pd.Timestamp("2023-03-04").to_pydatetime(),
                                      "kw": 1.0, "flag": "ok"}])
    city_store.retention = {**city_store.retention, "meter_readings_days": 2}
    assert city_store.prune("2023-03-05")["meter_readings"] == 1
    assert len(city_store.meter_readings()) == 1


def test_a_phase7_database_gets_the_meter_tables(tmp_path):
    from storage import Store
    from storage import schema as S
    path = tmp_path / "v3.db"
    s = Store(f"sqlite:///{path.as_posix()}")
    s.engine.dispose()
    con = sqlite3.connect(path)
    con.execute("DROP TABLE meter_readings")
    for col in ("observed_pct", "data_quality", "meter_stats"):
        con.execute(f"ALTER TABLE sim_steps DROP COLUMN {col}")
    con.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")
    con.commit()
    con.close()
    s = Store(f"sqlite:///{path.as_posix()}")
    with s.engine.connect() as c:
        cols = {r[1] for r in c.exec_driver_sql("PRAGMA table_info(sim_steps)")}
        tables = {r[0] for r in c.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")}
        version = c.exec_driver_sql("SELECT value FROM meta WHERE key = 'schema_version'").scalar_one()
    assert {"observed_pct", "data_quality", "meter_stats"} <= cols and "meter_readings" in tables
    assert version == str(S.SCHEMA_VERSION) == "4"


# ------------------------------------------------------------------ the operator's view in the loop
def test_detection_uses_what_the_operator_saw_and_the_outcome_the_truth(city_store, monkeypatch):
    """The meters show twice the load: the risk classes and alerts follow the observed loads, while the
    recorded outcome (measured_*) is the power flow of the true loads."""
    import engine as e
    from orchestrator import Orchestrator
    o = Orchestrator(city_store, engine=e.GridEngine(), scenario=C1)
    t = MAY + pd.Timedelta(hours=19)
    real = o.demand.observe

    def doubled(tt, sub=None):
        out = real(tt, sub)
        return {**out, "loads": {k: 2 * v for k, v in out["loads"].items()}}

    monkeypatch.setattr(o.demand, "observe", doubled)
    r = o.step(t, run_id=1)
    truth = o.engine.evaluate_configuration(o.demand.measured(t), [])["metrics"]
    assert r["measured_max_pct"] == round(truth["max_loading"], 2) and r["measured_over"] == 0
    assert max(r["observed_pct"].values()) > 1.5 * r["measured_max_pct"]
    seen_over = {tx for tx, v in r["observed_pct"].items() if v > o.threshold}
    assert seen_over and {tx for tx, c in r["risk"].items() if c == "actual_overload"} == seen_over
    seen = o.engine.evaluate_configuration({k: 2 * v for k, v in o.demand.last_seen.items()}, [])["metrics"]
    net = city_store.alerts(context_prefix="service ")
    net = net[net["kind"] != "transformer"]
    assert len(net) == len(o.risk.network_alerts(seen)) != len(o.risk.network_alerts(truth))
    assert set(r["data_quality"].values()) == {"measured"} and r["meter_stats"]["readings"] > 4000


def test_the_classic_network_has_no_observed_view(project_root, tmp_path):
    import engine as e
    from orchestrator import Orchestrator
    from storage import Store
    s = Store(f"sqlite:///{(tmp_path / 'c.db').as_posix()}")
    s.import_forecast_csv(project_root / "data" / "forecast.csv")
    o = Orchestrator(s, engine=e.GridEngine(), service="t", forecast_source="precomputed")
    r = o.step(o.demand.times()[30], run_id=1)
    assert r["observed_pct"] is None and r["data_quality"] is None and r["meter_stats"] is None


def test_closed_loop_reports_what_the_operator_missed(city):
    import closed_loop
    row, steps = closed_loop.run(C4, start="2023-05-09 17:00", end="2023-05-09 20:45", meters=True)
    assert row["observed_error_pp_mean"] > 0 and {"overloads_missed", "overloads_false"} <= set(row)
    perfect, _ = closed_loop.run(C4, start="2023-05-09 17:00", end="2023-05-09 20:45", meters=False)
    assert perfect["observed_error_pp_mean"] == 0 and perfect["overloads_missed"] == 0
    assert perfect["tx_steps_not_metered"] == 0


# ------------------------------------------------------------------ dashboard
class FakeWall:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_data_quality_and_transformer_pages_show_the_meter_streams(city_store, project_root):
    from streamlit.testing.v1 import AppTest
    import engine as e
    from orchestrator import Orchestrator
    from service import Service
    w = FakeWall()
    svc = Service(city_store, name="service", orchestrator=Orchestrator(city_store, engine=e.GridEngine(), scenario=C4),
                  wall=w, sleep=w.sleep)
    svc.execute("jump", {"time": "2023-05-09 18:00"})
    for _ in range(4):
        svc.step_once()
    svc.save_state()
    at = AppTest.from_file(str(project_root / "app" / "views" / "data_quality.py"), default_timeout=90)
    at.run()
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    n = len(config.grid()["transformers"])
    assert m["Transformer meters reporting"].endswith(f"/ {n}") and int(m["Smart-meter readings this step"].replace(",", "")) > 4000
    table = at.dataframe[0].value
    assert len(table) == n and {"value from", "seen %", "true %"} <= set(table.columns)
    assert table["error pts"].abs().max() < 10
    at = AppTest.from_file(str(project_root / "app" / "views" / "transformers.py"), default_timeout=90)
    at.run()
    assert not at.exception
    t = at.dataframe[0].value
    assert {"data", "true %"} <= set(t.columns) and (t["loading %"] != t["true %"]).any()


def test_the_data_quality_page_explains_perfect_measurement(tmp_path, monkeypatch, project_root):
    from streamlit.testing.v1 import AppTest
    from storage import Store
    url = f"sqlite:///{(tmp_path / 'c.db').as_posix()}"
    monkeypatch.setenv("LFDR_DB_URL", url)
    s = Store(url)
    name = config.simulation()["orchestrator"]["controller"]
    s.save_sim_state(name, status="paused", run_id=1, scenario="x", clock={"speed": 1, "last_time": "2020-02-10 10:00"})
    s.record_step({"service": name, "run_id": 1, "sim_time": "2020-02-10 10:00", "decision": "no_action"})
    at = AppTest.from_file(str(project_root / "app" / "views" / "data_quality.py"), default_timeout=90)
    at.run()
    assert not at.exception and any("perfect measurement" in x.value for x in at.info)


def test_flagging_a_meter_revises_the_estimates_it_was_in(demand, model):
    from meters import HeadEnd
    h = HeadEnd(model)
    m = int(np.flatnonzero(model.stuck_from < len(demand.times) - 400)[0])
    j = int(model.stuck_from[m])
    h.rebuild(j, ())
    revised = None
    for i in range(j + 1, j + 40):
        before = h.suspect[m]
        _, est = h.step(i, ())
        if h.suspect[m] and not before:
            revised = est
            break
    assert revised is not None
    held = {x for x in h.agg if not np.isnan(h.agg[x]["kw"][m])}
    assert held and held <= set(revised)                                     # every interval it was in is revised


def test_zero_use_is_not_taken_for_a_stuck_meter(model):
    from meters import HeadEnd
    h = HeadEnd(model)
    out = []
    for j in range(20):
        h.agg[j] = {"kw": np.full(model.n, np.nan), "w": np.ones(model.n)}
        idx = np.array([0, 1])
        h._deliver(j, idx, np.array([0.0, 1.5]), np.array([0, 0]), out)        # meter 0 uses nothing, meter 1 is stuck
    assert not h.suspect[0] and h.suspect[1]


def test_readings_later_than_the_window_are_lost(demand, model):
    from meters import HeadEnd
    h = HeadEnd(model)
    i = idx(demand, MAY)
    h.rebuild(i - 1, ())
    far = np.full(model.n, np.iinfo(np.int64).max // 2)
    h.pending[i - h.window - 5] = (np.ones(model.n), far)
    h.step(i, ())
    assert i - h.window - 5 not in h.pending and min(h.pending) > i - h.window
