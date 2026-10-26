"""Send a command to the running simulation service and print the result (or show its status).

    python sim/control.py status
    python sim/control.py pause | resume | step | restart | stop
    python sim/control.py speed 3600 | speed accelerated | speed max
    python sim/control.py jump "2020-02-29 18:00"
    python sim/control.py mode monitor | recommend | auto
    python sim/control.py scenario "S3 EV surge (T4 +200 kW, 18-22 h)"
    python sim/control.py approve REC-0003 | reject REC-0003
    python sim/control.py estop | clear-estop | rollback | reset
    python sim/control.py threshold 80
    python sim/control.py use-mv on | use-mv off          # allow MV feeder transfers
    python sim/control.py policy pause | policy run        # recommend mode: pause for each proposal, or keep running
    python sim/control.py steps 10          # the last 10 steps

Commands go through the database, so this works from any terminal while the service runs; the service
executes them in order and records the result.
"""
import argparse
import datetime as dt
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import config  # noqa: E402

ARGS = {"speed": "speed", "jump": "time", "mode": "mode", "scenario": "name", "approve": "rec_id", "reject": "rec_id",
        "threshold": "value", "use_mv": "value", "policy": "pause_on_recommendation"}
_YES = {"on": True, "yes": True, "true": True, "1": True, "pause": True,
        "off": False, "no": False, "false": False, "0": False, "run": False}


def _value(command, value):
    """Command-line text -> the argument the service expects."""
    if command == "threshold":
        return float(value)
    if command in ("use_mv", "policy"):
        v = str(value).lower()
        if v not in _YES:
            raise ValueError(f"{command} takes on/off (policy: pause/run), got {value!r}")
        return _YES[v]
    return value


def service_alive(state, stale_after_s=None):
    """True if the service's heartbeat is recent."""
    if not state or state.get("status") == "stopped" or not state.get("heartbeat_at"):
        return False
    stale = stale_after_s or config.simulation()["service"]["stale_after_s"]
    if (dt.datetime.now() - dt.datetime.fromisoformat(state["heartbeat_at"])).total_seconds() >= stale:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(state["pid"])) if state.get("pid") else True
    except ImportError:
        return True


def send(store, service, command, value=None, wait_s=10.0, source="cli"):
    """Queue a command; wait for the service to execute it. Returns the command row (status may still be
    pending if the service is not running)."""
    command = command.replace("-", "_")
    args = {ARGS[command]: _value(command, value)} if command in ARGS else {}
    cid = store.send_command(service, command, args, source=source)
    t_end = time.monotonic() + wait_s
    while time.monotonic() < t_end:
        row = store.command(cid)
        if row["status"] != "pending":
            return row
        time.sleep(0.1)
    return store.command(cid)


def describe(state):
    if not state:
        return "no service has run on this database yet (start one: python sim/service.py)"
    clk = state["clock"]
    alive = service_alive(state)
    af = state.get("achieved_factor")
    return (f"service '{state['service']}' ({'alive' if alive else 'not running'}): {state['status']}"
            + (f" ({state['note']})" if state.get("note") else "")
            + f"\n  run {state['run_id']}, last step {clk.get('last_time') or '-'}, next {clk['next_time']}, "
              f"range {clk['start']} .. {clk['end']}"
            + f"\n  speed {clk['speed']}" + (f", achieved {af:,.0f}x" if af else "")
            + f", lag {state.get('lag_s') or 0:.2f} s, {state['scenario']}, forecasts {state.get('forecast_source')}"
            + f"\n  heartbeat {state.get('heartbeat_at')}, pid {state.get('pid')}")


def main(argv=None):
    from storage import Store
    ap = argparse.ArgumentParser(description="Control the simulation service.")
    ap.add_argument("command")
    ap.add_argument("value", nargs="?")
    ap.add_argument("--service", default=config.simulation()["orchestrator"]["controller"])
    a = ap.parse_args(argv)
    store = Store()
    if a.command == "status":
        print(describe(store.load_sim_state(a.service)))
        return 0
    if a.command == "steps":
        d = store.steps(a.service, limit=int(a.value or 10), newest_first=True)
        cols = ["sim_time", "decision", "rec_id", "config_after", "forecast_max_pct", "measured_max_pct", "duration_ms"]
        print("no steps yet" if d.empty else d[cols].iloc[::-1].to_string(index=False))
        return 0
    if a.command.replace("-", "_") in ARGS and a.value is None:
        print(f"{a.command} needs a value")
        return 2
    if not service_alive(store.load_sim_state(a.service)):
        print("note: the service is not running; the command will run when it starts")
    row = send(store, a.service, a.command, a.value)
    print(f"{row['command']}: {row['status']}" + (f" - {row['result']}" if row.get("result") else ""))
    return 0 if row["status"] in ("done", "pending") else 1


if __name__ == "__main__":
    sys.exit(main())
