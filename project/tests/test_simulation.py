"""Phase 5: simulation clock (C2, C3), closed-loop orchestrator (G10), background service with commands and
restart recovery (G11), live dashboard, performance measurement (H7)."""
import datetime as dt
import os
import subprocess
import sys
import time

import pandas as pd
import pytest

from core.clock import ClockError, SimulationClock

S3 = "S3 EV surge (T4 +200 kW, 18-22 h)"
S2 = "S2 Heatwave (T1-T3 x1.6)"
H = pd.Timedelta(hours=1)


class FakeWall:
    """A wall clock the test moves by hand (sleep advances it)."""
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def make_clock(wall, **kw):
    return SimulationClock("2020-02-09 07:00", "2020-03-24 23:00", wall=wall, **kw)


# ------------------------------------------------------------------ clock (C2, C3)
def test_accelerated_mode_is_one_year_per_day():
    w = FakeWall()
    c15 = SimulationClock("2020-01-01", "2020-12-31", step_minutes=15, speed="accelerated", wall=w)
    assert c15.factor == 365 and round(c15.interval_s, 3) == 2.466          # "a 15-minute step every 2.466 s"
    c60 = SimulationClock("2020-01-01", "2020-12-31", step_minutes=60, speed="accelerated", wall=w)
    assert round(c60.interval_s, 3) == 9.863
    assert c15.interval_s * 365 * 24 * 4 == pytest.approx(86400)            # a simulated year takes 24 h
    assert make_clock(w, speed=3600).interval_s == 1.0 and make_clock(w, speed="max").interval_s == 0.0


def test_pause_resume_step_and_pacing():
    w = FakeWall()
    c = make_clock(w, speed=3600)                                          # one simulated hour per second
    assert not c.due() and c.wait_s() is None                              # paused at creation
    c.resume()
    assert c.due()                                                         # first step at once
    c.advance()
    assert c.last_time == pd.Timestamp("2020-02-09 07:00") and c.next_time == pd.Timestamp("2020-02-09 08:00")
    assert not c.due() and c.wait_s() == pytest.approx(1.0)
    w.sleep(0.6)
    assert not c.due() and c.wait_s() == pytest.approx(0.4)
    w.sleep(0.4)
    assert c.due()
    c.advance()
    c.pause()
    assert not c.due() and c.wait_s() is None
    c.advance()                                                            # a manual step while paused
    assert c.next_time == pd.Timestamp("2020-02-09 10:00") and not c.running


def test_achieved_speed_is_measured():
    w = FakeWall()
    c = make_clock(w, speed=7200)                                          # 2 h per second: 0.5 s per step
    c.resume()
    for _ in range(10):
        while not c.due():
            w.sleep(0.05)
        w.sleep(0.01)                                                      # the step's own work
        c.advance()
    assert c.achieved_factor == pytest.approx(7200, rel=0.03)


@pytest.mark.parametrize("catch_up", [False, True])
def test_a_slow_step_never_causes_a_burst_unless_asked(catch_up):
    """Step 2 takes 3.5 s at 1 s per step. Without catch-up the clock carries on at the same pace from now
    (lag reported); with catch-up it runs the missed slots back to back."""
    w = FakeWall()
    c = make_clock(w, speed=3600, catch_up=catch_up)
    c.resume()
    c.advance()                                                            # step 1 at t=0, next due t=1
    w.sleep(1.0)
    assert c.due()
    w.sleep(3.5)                                                           # step 2 runs from t=1 to t=4.5
    c.advance()
    assert c.lag_s == pytest.approx(2.5)                                   # its slot ended at t=2
    back_to_back = 0
    while c.due():
        c.advance()
        back_to_back += 1
    assert back_to_back == (3 if catch_up else 1)                          # catch-up: slots t=2, 3, 4 were missed
    assert c.wait_s() == pytest.approx(0.5 if catch_up else 1.0)


def test_jump_restart_and_end():
    w = FakeWall()
    c = make_clock(w, speed="max")
    with pytest.raises(ClockError, match="not on the"):
        c.jump("2020-02-10 07:30")
    with pytest.raises(ClockError, match="outside the simulation range"):
        c.jump("2020-04-01 00:00")
    c.jump("2020-03-24 22:00")
    c.resume()
    w.sleep(0.3)
    c.advance()
    assert c.lag_s == 0 and c.due()                     # max speed: no schedule to be late for, next step at once
    c.advance()
    assert c.finished and not c.running
    with pytest.raises(ClockError, match="reached the end"):
        c.resume()
    c.restart()
    assert c.next_time == c.start and c.last_time is None and not c.finished
    with pytest.raises(ClockError, match="speed must be"):
        c.set_speed("warp")


def test_clock_state_round_trip():
    w = FakeWall()
    c = make_clock(w, speed="accelerated")
    c.jump("2020-02-29 18:00")
    c.resume()
    c.advance()
    d = make_clock(w, speed=3600)
    d.restore(c.state())
    assert (d.next_time, d.last_time, d.speed, d.running) == (c.next_time, c.last_time, "accelerated", True)
    d.restore(c.state(), resume=False)
    assert not d.running


# ------------------------------------------------------------------ orchestrator (G10)
@pytest.fixture
def store(tmp_path, monkeypatch, project_root):
    url = f"sqlite:///{(tmp_path / 'sim.db').as_posix()}"
    monkeypatch.setenv("LFDR_DB_URL", url)
    from orchestrator import prepare_database
    from storage import Store
    s = Store(url)
    prepare_database(s)
    return s


@pytest.fixture
def orch(store):
    from orchestrator import Orchestrator
    def make(scenario=S3, mode="auto", name="t"):
        o = Orchestrator(store, scenario=scenario, service=name)
        o.ctl.set_mode(mode)
        return o
    return make


def test_a_step_runs_every_stage_and_reroutes_before_the_overload(orch, store):
    """S3, automatic: at the 20:00 boundary the 21:00 forecast for T4 is 91.5%, so T4's block moves to T5
    before 21:00, and the 21:00 step measures the effect."""
    o = orch()
    r = o.step("2020-02-29 20:00", run_id=1)
    assert set(r["timings_ms"]) == {"measure", "power_flow", "forecast", "detect", "decide", "record"}
    assert r["decision"] == "reroute" and r["config_before"] == [] and r["config_after"] == ["T4>T5"]
    assert r["forecast_max_pct"] == pytest.approx(91.5, abs=0.05) and r["forecast_over"] == 1
    assert o.ctl.last.for_time == "2020-02-29 21:00"
    nxt = o.step("2020-02-29 21:00", run_id=1)
    assert nxt["config_before"] == ["T4>T5"]
    assert nxt["measured_max_normal_pct"] == pytest.approx(81.5, abs=0.05)   # what T4 would have carried
    assert nxt["measured_max_pct"] < nxt["measured_max_normal_pct"]
    steps = store.steps("t", run_id=1)
    assert steps["sim_time"].tolist() == [pd.Timestamp("2020-02-29 20:00"), pd.Timestamp("2020-02-29 21:00")]
    assert store.record_step({**{k: v for k, v in r.items() if k != "rec"}}) == 0   # idempotent


def test_monitor_never_switches_and_recommend_waits_for_the_operator(orch):
    m = orch(mode="monitor", name="m")
    r = m.step("2020-02-29 20:00")
    assert r["decision"] == "monitor" and r["config_after"] == [] and r["forecast_over"] == 1
    rc = orch(mode="recommend", name="r")
    r = rc.step("2020-02-29 20:00")
    assert r["decision"] == "reroute" and r["config_after"] == [] and rc.awaiting_operator
    assert rc.ctl.pending.for_time == "2020-02-29 21:00"


def test_quiet_hours_create_no_recommendation_and_the_last_step_decides_nothing(orch, store):
    o = orch(name="q")
    assert o.step("2020-02-29 03:00")["decision"] == "no_action" and o.ctl.last is None
    assert o.step("2020-03-24 23:00")["decision"] == "end"
    o.ctl.emergency_stop("2020-02-29 20:00")
    assert o.step("2020-02-29 20:00")["decision"] == "estop"


@pytest.mark.slow
def test_monitor_loop_sees_exactly_the_benchmark_overloads(project_root):
    """Monitor mode over the whole test period: the forecast transformer-hours over 85% equal the published
    benchmark (docs/results.csv), so the loop uses the same forecasts, scenario rules and power flow."""
    import closed_loop
    published = pd.read_csv(project_root / "docs" / "results.csv").set_index("scenario")
    row, steps = closed_loop.run(S3, mode="monitor")
    assert row["forecast_tx_hours_over"] == published.loc[S3, "transformer_hours_over"] == 22
    assert row["measured_tx_hours_over_with_control"] == row["measured_tx_hours_over_no_control"]
    assert row["switch_operations"] == 0 and set(steps["decision"]) == {"monitor", "end"}


@pytest.mark.slow
def test_closed_loop_results_match_docs(project_root):
    import closed_loop
    want = pd.read_csv(project_root / "docs" / "closed_loop.csv").set_index("scenario")
    for name in (S2, S3):
        row, _ = closed_loop.run(name)
        for k, v in row.items():
            if k in closed_loop.TIMING or k == "scenario":
                continue
            w = want.loc[name, k]
            assert (pd.isna(w) and v is None) or v == w, (name, k, v, w)


def test_live_forecasts_equal_the_committed_ones(store, project_root):
    if not (project_root / "data" / "processed" / "clusters.parquet").exists():
        pytest.skip("needs data/processed/clusters.parquet (the eMARC data)")
    from sources import LiveForecasts, PrecomputedForecasts
    live, pre = LiveForecasts(store), PrecomputedForecasts(store)
    for t in ("2020-02-21 06:00", "2020-02-29 21:00", "2020-03-20 13:00"):
        a, b = live.point(t), pre.point(t)
        assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-3 for k in a)
    assert live.issue("2020-03-01 10:00") == 1                            # issues the forecast for t + 24 h
    with store.engine.connect() as c:
        n = c.exec_driver_sql("select count(*) from forecasts where model_version = ? and target_time = ?",
                              (live.model_version, "2020-03-02 10:00:00.000000")).scalar_one()
    assert n == 6 and live.model_version == "xgb-24h-v1-live"


# ------------------------------------------------------------------ service (G11)
@pytest.fixture
def service(store):
    from service import Service
    made = []

    def make(name="svc", **kw):
        w = FakeWall()
        s = Service(store, name=name, wall=w, sleep=w.sleep, **kw)
        made.append(s)
        return s
    return make


def test_recommend_mode_pauses_for_the_operator_then_resumes_on_approval(service, store):
    svc = service(scenario=S3)
    for cmd, args in (("mode", {"mode": "recommend"}), ("jump", {"time": "2020-02-29 15:00"}),
                      ("speed", {"speed": "max"}), ("resume", {})):
        svc.execute(cmd, args)
    svc.run(exit_when_idle=True)
    assert svc.status == "awaiting_operator" and not svc.clock.running
    rec = svc.ctl.pending
    assert rec.for_time == f"{svc.clock.next_time:%Y-%m-%d %H:%M}"               # the hour about to start
    assert store.load_sim_state("svc")["status"] == "awaiting_operator"
    cid = store.send_command("svc", "approve", {"rec_id": rec.rec_id}, source="test")
    svc.handle_commands()
    assert store.command(cid)["status"] == "done" and svc.clock.running and svc.ctl.active == rec.target
    row = svc.step_once()
    assert row["config_before"] == rec.target and row["measured_max_pct"] < row["measured_max_normal_pct"]


def test_conditions_that_change_cancel_a_waiting_proposal(service, store):
    svc = service(scenario=S3)
    for cmd, args in (("mode", {"mode": "recommend"}), ("jump", {"time": "2020-02-29 20:00"}), ("step", {})):
        svc.execute(cmd, args)
    rec = svc.ctl.pending
    assert rec is not None
    svc.execute("scenario", {"name": S2})
    assert svc.ctl.pending is None
    cid = store.send_command("svc", "approve", {"rec_id": rec.rec_id}, source="test")
    svc.handle_commands()
    assert store.command(cid)["status"] == "refused" and "not the pending" in store.command(cid)["result"]


def test_commands_are_executed_in_order_and_refusals_recorded(service, store):
    svc = service()
    ids_ = [store.send_command("svc", c, a, source="test") for c, a in (
        ("speed", {"speed": "accelerated"}), ("jump", {"time": "2020-02-29 18:30"}), ("step", {}),
        ("resume", {}), ("step", {}), ("warp", {}), ("pause", {}), ("restart", {}))]
    svc.handle_commands()
    got = [(store.command(i)["status"], store.command(i)["result"]) for i in ids_]
    assert got[0] == ("done", "speed accelerated")
    assert got[1][0] == "refused" and "not on the" in got[1][1]
    assert got[2][0] == "done" and got[2][1].startswith("stepped 2020-02-09 07:00")
    assert got[4][0] == "refused" and "pause it before stepping" in got[4][1]
    assert got[5][0] == "refused" and "unknown command" in got[5][1]
    assert got[7] == ("done", "run 2 from 2020-02-09 07:00") and svc.run_id == 2
    assert svc.clock.running                                                # a new run starts running
    svc.execute("restart", {"paused": True})
    assert svc.run_id == 3 and not svc.clock.running
    assert store.load_sim_state("svc")["run_id"] == 2


def test_restart_recovery_continues_after_the_last_recorded_step(service, store):
    """The process 'dies' after some steps without saving its state: the next service resumes from sim_steps
    and the controller's configuration."""
    svc = service(scenario=S3)
    for cmd, args in (("mode", {"mode": "auto"}), ("jump", {"time": "2020-02-29 17:00"})):
        svc.execute(cmd, args)
    for _ in range(4):                                                      # 17:00 .. 20:00, reroute at 20:00
        svc.step_once()
    active = list(svc.ctl.active)
    assert active == ["T4>T5"]
    again = service(scenario=None)                                          # same database, same name
    assert again.run_id == svc.run_id and again.clock.next_time == pd.Timestamp("2020-02-29 21:00")
    assert again.ctl.active == active and again.orch.scenario.name == S3 and again.ctl.mode == "auto"
    row = again.step_once()                                                 # 21:00 runs in the restored configuration
    assert row["config_before"] == ["T4>T5"] and row["decision"] == "restore"   # the surge ends at 22:00
    steps = store.steps("svc", run_id=again.run_id)
    assert steps["sim_time"].is_unique and len(steps) == 5


def test_a_second_service_is_refused_while_the_first_is_alive(service, store):
    from service import Service, ServiceRunning
    service()
    store.save_sim_state("svc", pid=os.getppid(), heartbeat_at=dt.datetime.now().isoformat(timespec="seconds"),
                         status="running")
    with pytest.raises(ServiceRunning, match="already running"):
        Service(store, name="svc")
    store.save_sim_state("svc", heartbeat_at="2020-01-01T00:00:00")          # stale heartbeat: take over
    Service(store, name="svc")


def test_a_stop_sent_while_no_service_ran_is_void(service, store):
    cid = store.send_command("svc", "stop", {}, source="test")
    svc = service()
    assert store.command(cid)["status"] == "refused" and not svc._stop


@pytest.mark.slow
def test_killed_service_recovers_and_gives_the_same_result(tmp_path, project_root):
    """Run the real service process, kill it hard part way, start it again: every hour is recorded once and
    the outcome equals the uninterrupted closed-loop run (docs/closed_loop.csv)."""
    env = {**os.environ, "LFDR_DB_URL": f"sqlite:///{(tmp_path / 'kill.db').as_posix()}"}
    cmd = [sys.executable, str(project_root / "sim" / "service.py"), "--speed", "max", "--mode", "auto",
           "--scenario", S3]
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from storage import Store
    s = Store(env["LFDR_DB_URL"])
    t_end = time.monotonic() + 120
    while time.monotonic() < t_end:
        st = s.load_sim_state("service")
        if st and (st.get("steps_done") or 0) >= 150:
            break
        time.sleep(0.2)
    p.kill()
    p.wait()
    done = len(s.steps("service"))
    assert 150 <= done < 1073
    out = subprocess.run(cmd[:2] + ["--exit-when-idle"], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    steps = s.steps("service")
    assert len(steps) == 1073 and steps["sim_time"].is_unique and (steps["sim_time"].diff().dropna() == H).all()
    assert s.load_sim_state("service")["status"] == "finished"
    want = pd.read_csv(project_root / "docs" / "closed_loop.csv").set_index("scenario").loc[S3]
    assert int(steps["measured_over"].sum()) == want["measured_tx_hours_over_with_control"]
    assert int(steps["measured_over_normal"].sum()) == want["measured_tx_hours_over_no_control"]
    assert len(s.actions(controller="service", limit=None)) == want["switch_operations"]
    assert int((steps["config_before"].map(len) > 0).sum()) == want["hours_rerouted"]


def test_control_cli(store, capsys):
    import control
    assert control.main(["status"]) == 0 and "no service has run" in capsys.readouterr().out
    row = control.send(store, "service", "pause", wait_s=0.3)
    assert row["status"] == "pending"
    assert control.main(["jump"]) == 2 and "needs a value" in capsys.readouterr().out
    for words, args in ((["threshold", "80"], {"value": 80.0}), (["use-mv", "on"], {"value": True}),
                        (["policy", "run"], {"pause_on_recommendation": False})):
        control.send(store, "service", *words, wait_s=0)
        assert store.commands("service").iloc[0]["args"] == args
    with pytest.raises(ValueError, match="on/off"):
        control.send(store, "service", "use-mv", "maybe", wait_s=0)


# ------------------------------------------------------------------ performance (H7)
def test_benchmark_measures_every_stage(tmp_path):
    import benchmark
    m = {k: v for k, v, _ in benchmark.loop_benchmark(steps=12, db_dir=tmp_path)}
    for stage in ("measure", "power_flow", "forecast", "detect", "decide", "record", "total"):
        assert m[f"step_{stage}_mean"] >= 0
    assert m["steps"] == 12 and m["throughput"] > 0 and m["memory_rss_peak"] > 0
    assert m["accelerated_budget_per_15min_step"] == pytest.approx(2465.753, abs=0.01)


def test_real_clock_holds_its_speed(tmp_path):
    import benchmark
    m = {k: v for k, v, _ in benchmark.clock_benchmark(target_factor=36000, steps=8, db_dir=tmp_path)}
    assert abs(m["clock_speed_error"]) < 10
