"""Background simulation service (gap item G11): runs the closed loop on the simulation clock, independent
of any dashboard, with its whole state in the database.

    python sim/service.py                         # start or resume (settings from config/simulation.yaml)
    python sim/service.py --speed accelerated     # one simulated year per 24 h
    python sim/service.py --speed max --mode auto --scenario "S3 EV surge (T4 +200 kW, 18-22 h)"
    python sim/service.py --new-run               # start again from the beginning as a new run

Control it while it runs from another terminal (python sim/control.py ...) or from the live dashboard
(streamlit run app/live.py). Both only write commands to the database; the service is the only process
that changes the simulation, so there is one authoritative state.

Restart recovery: the clock, scenario, run number and controller (mode, switch configuration, pending
proposal, emergency stop) are restored from the database, and the simulation continues with the step after
the last one recorded in sim_steps. A step interrupted half way is run again; its alerts and forecast scores
are idempotent, and the controller saved any switching it did, so the re-run sees the configuration in
force. Only one service per name runs at a time: a second one refuses to start while the first one's
heartbeat is fresh.
"""
import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "grid", ROOT / "sim"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from core import config  # noqa: E402
from core.clock import ClockError, SimulationClock, resolve_range  # noqa: E402
from core.logging_setup import get_logger  # noqa: E402

log = get_logger("service")

COMMANDS = ("resume", "pause", "step", "speed", "jump", "restart", "mode", "scenario", "threshold", "use_mv",
            "policy", "approve", "reject", "estop", "clear_estop", "rollback", "reset", "stop")


class ServiceRunning(RuntimeError):
    """Another service with the same name is alive."""


def _iso_now():
    return dt.datetime.now().isoformat(timespec="seconds")


def _pid_alive(pid):
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except ImportError:                                   # without psutil, trust the heartbeat alone
        return True


class Service:
    def __init__(self, store=None, name=None, orchestrator=None, wall=time.monotonic, sleep=time.sleep,
                 new_run=False, paused=False, scenario=None, forecast_source=None):
        from storage import Store
        cfg = config.simulation()
        self.cfg = cfg
        self.store = store or Store()
        self.name = name or cfg["orchestrator"]["controller"]
        self._wall, self._sleep = wall, sleep
        prev = self.store.load_sim_state(self.name)
        self._refuse_if_running(prev)
        if orchestrator is None:
            from orchestrator import Orchestrator, prepare_database
            prepare_database(self.store)
            orchestrator = Orchestrator(self.store, service=self.name,
                                        scenario=scenario or (prev or {}).get("scenario"),
                                        forecast_source=forecast_source or (prev or {}).get("forecast_source"))
        self.orch = orchestrator
        if scenario:
            self.orch.set_scenario(scenario)
        self.ctl = self.orch.ctl
        c = cfg["clock"]
        start, end = resolve_range(c, self.orch.demand.times())
        self.clock = SimulationClock(start, end, c["step_minutes"], c["speed"], c["accelerated_factor"],
                                     c["catch_up"], wall=wall)
        if self.clock.step != self.orch.demand.resolution:
            raise ClockError(f"clock.step_minutes ({self.clock.step}) must equal the data resolution of the "
                             f"{self.orch.demand.name} source ({self.orch.demand.resolution})")
        self.note, self.paused_for_rec, self._stop, self._behind = "", False, False, False
        self.failed = False                                 # the last step raised (cleared by any clock command)
        self.pause_on_rec = cfg["orchestrator"]["pause_on_recommendation"]     # the approval policy
        saved = (prev or {}).get("settings") or {}
        if saved:
            self.orch.apply_settings(saved)
            self.pause_on_rec = saved.get("pause_on_recommendation", self.pause_on_rec)
        self._last_beat = None
        if prev is None or new_run:
            self.run_id = (prev or {}).get("run_id", 0) + 1
            if prev is None and self.store.load_state(self.name) is None:
                self.ctl.set_mode(cfg["orchestrator"]["mode"])          # first start ever
            if new_run and prev is not None:
                self.ctl.cancel_pending("new run")
                if self.ctl.active:
                    self.ctl.reset_to_normal(None)
            how = "started"
        else:
            self.run_id = prev["run_id"]
            self.clock.restore(prev["clock"], resume=cfg["service"]["resume_on_start"] and not paused)
            last = self.store.last_step_time(self.name, self.run_id)
            if last is not None:                                         # sim_steps is the authority
                self.clock.last_time = pd.Timestamp(last)
                self.clock.next_time = self.clock.last_time + self.clock.step
                if self.clock.finished:
                    self.clock.pause()
            self.note = prev.get("note") or ""
            self.paused_for_rec = self.note.startswith("awaiting operator")
            how = "resumed"
        for cmd in self.store.pending_commands(self.name):          # a stop sent while no service ran is void
            if cmd["command"] == "stop":
                self.store.finish_command(cmd["id"], "refused", "the service was not running when this was sent")
        self.ctl.note("service", f"service '{self.name}' {how}: run {self.run_id}, next step "
                      f"{self.clock.next_time:%Y-%m-%d %H:%M}, {self.orch.scenario.name}, "
                      f"forecasts {self.orch.forecasts.name}, mode {self.ctl.mode}")
        self.save_state(started=True)

    # ------------------------------------------------------------------ state
    def _refuse_if_running(self, prev):
        if not prev or prev.get("status") == "stopped" or prev.get("pid") == os.getpid():
            return
        beat = prev.get("heartbeat_at")
        fresh = beat and (dt.datetime.now() - dt.datetime.fromisoformat(beat)).total_seconds() < \
            self.cfg["service"]["stale_after_s"]
        if fresh and _pid_alive(prev.get("pid")):
            raise ServiceRunning(f"service '{self.name}' is already running (pid {prev['pid']}, heartbeat {beat}); "
                                 "stop it first: python sim/control.py stop")

    @property
    def status(self):
        if self.clock.finished:
            return "finished"
        if self._stop:
            return "stopped"
        if self.paused_for_rec:
            return "awaiting_operator"
        if self.failed and not self.clock.running:
            return "step_failed"
        return "running" if self.clock.running else "paused"

    def save_state(self, started=False):
        fields = dict(run_id=self.run_id, clock=self.clock.state(), scenario=self.orch.scenario.name,
                      forecast_source=self.orch.forecasts.name, status=self.status, note=self.note, pid=os.getpid(),
                      heartbeat_at=_iso_now(), achieved_factor=self.clock.achieved_factor, lag_s=self.clock.lag_s,
                      steps_done=self.clock.steps_done, settings=self.settings())
        if started:
            fields["started_at"] = _iso_now()
        self.store.save_sim_state(self.name, **fields)
        self._last_beat = self._wall()

    def settings(self):
        return {**self.orch.settings(), "pause_on_recommendation": self.pause_on_rec}

    def _label(self, t=None):
        t = self.clock.last_time if t is None else t
        return None if t is None else f"{pd.Timestamp(t):%Y-%m-%d %H:%M}"

    # ------------------------------------------------------------------ one step
    def step_once(self):
        t = self.clock.next_time
        row = self.orch.step(t, self.run_id)
        self.clock.advance()
        if pd.Timestamp(t).normalize() != pd.Timestamp(self.clock.next_time).normalize():   # once per simulated day
            pruned = self.store.prune(t)
            if any(pruned.values()):
                log.debug("retention: %s", pruned)
        if self.clock.factor is not None:
            behind = self.clock.lag_s > self.cfg["clock"]["behind_warning_s"]
            if behind and not self._behind:
                self.ctl.note("clock", f"falling behind: step {self._label(t)} finished {self.clock.lag_s:.1f} s "
                              f"late; target {self.clock.factor:g}x, achieved "
                              f"{(self.clock.achieved_factor or 0):.0f}x", self._label(t), severity="warning")
            self._behind = behind
        if self.orch.awaiting_operator and self.pause_on_rec and self.clock.running:
            self.clock.pause()
            self.paused_for_rec = True
            self.note = f"awaiting operator: {self.ctl.pending.rec_id} for {self.ctl.pending.for_time}"
        if self.clock.finished:
            self.ctl.note("service", f"run {self.run_id} reached the end of the period ({self._label(t)})", self._label(t))
        return row

    def _step_failed(self, ex):
        """A step raised (a power flow that cannot be solved, a database error, ...): log it, pause before that
        step and say so on every page; the operator can resume (retry), jump past it or start a new run."""
        t = self._label(self.clock.next_time)
        log.exception("step %s failed", t)
        if self.clock.running:
            self.clock.pause()
        self.paused_for_rec, self.failed = False, True
        self.note = f"step {t} failed ({type(ex).__name__}: {ex}); paused, see logs/service.log"
        try:
            self.ctl.note("service", self.note, t, severity="critical")
        except Exception:                                   # the database itself may be the problem
            log.exception("could not record the failure")
        self.save_state()

    # ------------------------------------------------------------------ commands
    def _after_decision(self):
        if self.paused_for_rec:
            self.paused_for_rec, self.note = False, ""
            if not self.clock.finished:
                self.clock.resume()

    def execute(self, command, args=None):
        """Run one command; returns a short result text. Raises DecisionError/ClockError/ValueError if refused."""
        a, ctl, clk = args or {}, self.ctl, self.clock
        if command in ("resume", "step", "jump", "restart"):
            self.failed = False
        if command == "resume":
            clk.resume()
            self.paused_for_rec, self.note = False, ""
            return f"running at {clk.speed}"
        if command == "pause":
            clk.pause()
            self.paused_for_rec, self.note = False, "paused by the operator"
            return f"paused before {self._label(clk.next_time)}"
        if command == "step":
            if clk.running:
                raise ClockError("the clock is running: pause it before stepping")
            if clk.finished:
                raise ClockError("the run has reached the end of the period")
            row = self.step_once()
            return f"stepped {self._label(row['sim_time'])}: {row['decision']}"
        if command == "speed":
            return f"speed {clk.set_speed(a.get('speed'))}"
        if command == "jump":
            clk.jump(a.get("time"))
            ctl.cancel_pending("time jump")
            if self.paused_for_rec:
                self.paused_for_rec, self.note = False, ""
            return f"next step {self._label(clk.next_time)}"
        if command == "restart":
            start_at = a.get("from")
            if start_at:
                clk.jump(start_at)                                # validates the time before anything changes
            self.run_id += 1
            clk.restart()
            if start_at:
                clk.jump(start_at)
            ctl.cancel_pending("restart")
            if ctl.active:
                ctl.reset_to_normal(None)
            self.paused_for_rec, self.note = False, ""
            if a.get("paused"):                                   # a new run starts running, unless asked otherwise
                clk.pause()
            else:
                clk.resume()
            ctl.note("service", f"restarted: run {self.run_id} from {self._label(clk.next_time)}")
            return f"run {self.run_id} from {self._label(clk.next_time)}" + (", paused" if a.get("paused") else "")
        if command == "mode":
            ctl.set_mode(a.get("mode"), self._label())
            if a.get("mode") != "recommend":
                self._after_decision()
            return f"mode {ctl.mode}"
        if command == "scenario":
            self.orch.set_scenario(a.get("name"))
            ctl.cancel_pending("scenario changed")
            self._after_decision()
            ctl.note("service", f"scenario changed to {self.orch.scenario.name}", self._label())
            return self.orch.scenario.name
        if command == "threshold":
            self.orch.apply_settings({"threshold": a.get("value")})
            ctl.cancel_pending("threshold changed")
            self._after_decision()
            ctl.note("service", f"overload threshold set to {self.orch.threshold:g}%", self._label())
            return f"threshold {self.orch.threshold:g}%"
        if command == "use_mv":
            self.orch.apply_settings({"use_mv": bool(a.get("value"))})
            ctl.cancel_pending("MV transfer setting changed")
            self._after_decision()
            return f"MV transfers {'allowed' if self.orch.use_mv else 'not used'}"
        if command == "policy":
            self.pause_on_rec = bool(a.get("pause_on_recommendation"))
            if not self.pause_on_rec:
                self._after_decision()
            return ("recommend mode pauses the clock for each proposal" if self.pause_on_rec
                    else "recommend mode keeps running; proposals not decided in time are superseded")
        if command == "approve":
            rec = ctl.approve(a.get("rec_id"), self._label(clk.next_time), self.orch.context())
            self._after_decision()
            return f"{rec.rec_id} applied"
        if command == "reject":
            ctl.reject(a.get("rec_id"), a.get("reason", "rejected by the operator"))
            self._after_decision()
            return f"{a.get('rec_id')} rejected"
        if command == "estop":
            ctl.emergency_stop(self._label())
            self._after_decision()
            return "emergency stop active"
        if command == "clear_estop":
            ctl.clear_emergency_stop(self._label())
            return "emergency stop cleared"
        if command == "rollback":
            ctl.rollback(self._label())
            return "rolled back"
        if command == "reset":
            ctl.reset_to_normal(self._label())
            return "normal configuration"
        if command == "stop":
            self._stop = True
            return "stopping"
        raise ValueError(f"unknown command {command!r}; known: {', '.join(COMMANDS)}")

    def handle_commands(self):
        from decisions import DecisionError
        done = 0
        for cmd in self.store.pending_commands(self.name):
            try:
                result = self.execute(cmd["command"], cmd["args"])
                self.store.finish_command(cmd["id"], "done", result)
            except (DecisionError, ClockError, ValueError) as ex:
                self.store.finish_command(cmd["id"], "refused", str(ex))
            except Exception as ex:                                   # report, keep the service alive
                log.exception("command %s failed", cmd)
                self.store.finish_command(cmd["id"], "failed", f"{type(ex).__name__}: {ex}")
            done += 1
        if done:
            self.save_state()
        return done

    # ------------------------------------------------------------------ main loop
    def run(self, max_steps=None, exit_when_idle=False, max_wall_s=None):
        """Serve until a stop command (or max_steps steps, or max_wall_s seconds, or, with exit_when_idle, until
        the clock is paused or finished and no command is waiting). Returns the number of steps run."""
        poll, beat = self.cfg["service"]["poll_interval_s"], self.cfg["service"]["heartbeat_s"]
        steps, t_end = 0, None if max_wall_s is None else self._wall() + max_wall_s
        log.info("service '%s' serving: %s", self.name, self.status)
        try:
            while not self._stop:
                self.handle_commands()
                if self._stop or (max_steps is not None and steps >= max_steps):
                    break
                if t_end is not None and self._wall() >= t_end:
                    break
                if self.clock.due():
                    try:
                        self.step_once()
                    except Exception as ex:                       # a failed step pauses the clock, never the service
                        self._step_failed(ex)
                        continue
                    steps += 1
                    if self._wall() - self._last_beat >= beat or not self.clock.running:
                        self.save_state()
                    continue
                if exit_when_idle and not self.clock.running:
                    break
                if self._wall() - self._last_beat >= beat:
                    self.save_state()
                wait = self.clock.wait_s()
                self._sleep(poll if wait is None else min(poll, wait))
        finally:
            self.save_state()
            if self._stop:
                self.ctl.note("service", f"service '{self.name}' stopped at {self._label() or 'start'}", self._label())
        return steps


def main(argv=None):
    from core.logging_setup import setup
    from core.paths import LOGS_DIR
    ap = argparse.ArgumentParser(description="Run the closed-loop simulation service.")
    ap.add_argument("--city", help="run a synthetic city (cities/<name>, made by city/generate.py)")
    ap.add_argument("--speed", help="simulated seconds per wall second, 'accelerated' (1 year per 24 h) or 'max'")
    ap.add_argument("--mode", choices=["monitor", "recommend", "auto"])
    ap.add_argument("--scenario", help="a benchmark scenario name from config/scenarios.yaml")
    ap.add_argument("--forecast", choices=["precomputed", "live", "persistence", "short"],
                    help="forecast source (default from config; a city: persistence or short)")
    ap.add_argument("--paused", action="store_true", help="start paused (resume with: python sim/control.py resume)")
    ap.add_argument("--new-run", action="store_true", help="start again from the beginning as a new run")
    ap.add_argument("--steps", type=int, help="stop after this many steps")
    ap.add_argument("--exit-when-idle", action="store_true", help="exit when the clock is paused or finished")
    args = ap.parse_args(argv)
    config.use_city(args.city)
    setup(log_file=LOGS_DIR / ("service.log" if not args.city else f"service_{args.city}.log"))
    try:
        svc = Service(new_run=args.new_run, paused=args.paused, scenario=args.scenario, forecast_source=args.forecast)
    except ServiceRunning as ex:
        print(ex)
        return 2
    if args.mode:
        svc.execute("mode", {"mode": args.mode})
    if args.speed:
        svc.execute("speed", {"speed": args.speed})
    if not args.paused and not svc.clock.running and not svc.clock.finished and not svc.paused_for_rec:
        svc.execute("resume")
    svc.save_state()
    print(f"service '{svc.name}': run {svc.run_id}, {svc.status}, next step {svc.clock.next_time:%Y-%m-%d %H:%M}, "
          f"speed {svc.clock.speed}, mode {svc.ctl.mode}, {svc.orch.scenario.name}. Ctrl+C to stop.")
    try:
        n = svc.run(max_steps=args.steps, exit_when_idle=args.exit_when_idle)
    except KeyboardInterrupt:                 # Ctrl+C: a clean stop; the clock's running flag is kept for the restart
        svc._stop = True
        svc.save_state()
        svc.ctl.note("service", f"service '{svc.name}' stopped (Ctrl+C) at {svc._label() or 'start'}", svc._label())
        n = svc.clock.steps_done
        print("stopped")
    else:
        if not svc._stop:                       # --steps or --exit-when-idle: the process ends, the run does not
            svc._stop = True
            svc.save_state()
            svc.ctl.note("service", f"service '{svc.name}' exited after {n} steps at {svc._label() or 'start'}",
                         svc._label())
    print(f"{n} steps; last simulated time {svc._label() or '-'}; status {svc.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
