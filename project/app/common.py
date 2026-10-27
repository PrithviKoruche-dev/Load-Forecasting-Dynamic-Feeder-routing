"""Shared helpers for the dashboard pages: the database, the simulation service's state, commands, labels.

Every live page reads what the simulation service (sim/service.py) wrote to the database and changes nothing
itself: buttons only queue commands for the service. So all pages and all browsers show one state.
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "grid", ROOT / "sim", ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from core import config  # noqa: E402
from core.clock import season  # noqa: E402
from core.paths import LOGS_DIR  # noqa: E402
from control import service_alive  # noqa: E402
from storage import Store  # noqa: E402

# The configuration in force (the classic network or a synthetic city, chosen with --city before the pages load).
# These names are read from the configuration whenever they are used, so nothing here goes stale:
#   SIM, SERVICE, OPS, GRID, THR_DEFAULT, HARD_LIMIT, TX
_LIVE = {
    "SIM": lambda: config.simulation(),
    "SERVICE": lambda: config.simulation()["orchestrator"]["controller"],
    "OPS": lambda: config.operations(),
    "GRID": lambda: config.grid(),
    "THR_DEFAULT": lambda: config.operations()["limits"]["overload_threshold_pct"],
    "HARD_LIMIT": lambda: config.operations()["limits"]["transformer_hard_limit_pct"],
    "TX": lambda: list(config.grid()["transformers"]),
}


def _service():
    return config.simulation()["orchestrator"]["controller"]


def __getattr__(name):                     # module attributes C.GRID, C.TX, ... (PEP 562)
    if name in _LIVE:
        return _LIVE[name]()
    raise AttributeError(name)


MODES = {"monitor": "Monitor only", "recommend": "Recommend (operator approves)", "auto": "Automatic"}
RISK_LABELS = {"normal": "Normal", "elevated": "Elevated", "approaching": "Approaching limit",
               "predicted_overload": "Predicted overload", "actual_overload": "Actual overload"}
DECISIONS = {"no_action": "nothing to do", "monitor": "monitor only", "reroute": "reroute", "restore": "restore normal",
             "mitigate": "mitigation", "no_change": "already in the best configuration",
             "no_safe_option": "no safe option", "estop": "emergency stop", "end": "end of data"}
REFRESH_S = 2


@st.cache_resource
def _store(url):
    return Store(url)


def store():
    return _store(config.database_url())


def sim_state():
    return store().load_sim_state(_service())


def controller_state():
    return store().load_state(_service()) or {}


def settings(state=None):
    """The service's operator settings (threshold, MV transfers, approval policy), with config defaults."""
    s = (state or {}).get("settings") or {}
    return {"threshold": s.get("threshold") or config.operations()["limits"]["overload_threshold_pct"], "use_mv": s.get("use_mv"),
            "pause_on_recommendation": s.get("pause_on_recommendation",
                                             config.simulation()["orchestrator"]["pause_on_recommendation"])}


def last_step(state):
    """The most recent sim_steps row of the current run, as a dict (None before the first step)."""
    if not state:
        return None
    d = store().steps(_service(), run_id=state["run_id"], limit=1, newest_first=True)
    return None if d.empty else d.iloc[0].to_dict()


def run_steps(state, limit=None):
    if not state:
        return pd.DataFrame()
    d = store().steps(_service(), run_id=state["run_id"], limit=limit, newest_first=limit is not None)
    return d.sort_values("sim_time").reset_index(drop=True) if not d.empty else d


def season_of(t):
    return season(t, config.simulation()["seasons"])


def switch_states(stored):
    """Every switch's position from a step's record, which keeps only those away from normal (Phase 7)."""
    return {**{s: sw["normal"] for s, sw in config.grid()["switches"].items()}, **(stored or {})}


def tx_table():
    """Static facts per transformer from config/grid.yaml."""
    return pd.DataFrame([{"transformer": t, "asset_id": x["id"], "feeder": x["feeder"],
                          "feeder_id": config.grid()["feeders"][x["feeder"]]["id"], "mv_bus": x["mv_bus"],
                          "rating_kva": x["rating_kva"], "category": x["category"]}
                         for t, x in config.grid()["transformers"].items()])


def moves_of(tx):
    """The LV moves available for a transformer's movable block: [(action key, target transformer)]."""
    return [(f"{t}>{m['to']}", m["to"]) for t, m in config.move_options(config.grid()) if t == tx]


# ------------------------------------------------------------------ commands
def send(command, wait_s=1.5, **args):
    """Queue a command for the service and remember its id to show the result. If the service is running, wait
    briefly for it to be executed, so the page redraws with the new state."""
    import time
    cid = store().send_command(_service(), command, args, source="dashboard")
    st.session_state["last_cmd"] = cid
    if service_alive(sim_state()):
        t_end = time.monotonic() + wait_s
        while time.monotonic() < t_end and store().command(cid)["status"] == "pending":
            time.sleep(0.1)


def command_feedback():
    cid = st.session_state.get("last_cmd")
    c = store().command(cid) if cid else None
    if not c:
        return
    text = f"Command `{c['command']}`: {c['status']}" + (f" - {c['result']}" if c.get("result") else "")
    if c["status"] == "pending":
        st.info(text + (" (it runs when the service starts)" if not service_alive(sim_state()) else ""))
    else:
        (st.success if c["status"] == "done" else st.error)(text)


def start_service():
    """Launch sim/service.py in the background (it keeps running if the dashboard stops)."""
    LOGS_DIR.mkdir(exist_ok=True)
    out = open(LOGS_DIR / "service.out", "a")
    kw = {"cwd": str(ROOT), "stdout": out, "stderr": subprocess.STDOUT, "env": {**os.environ}}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
    else:
        kw["start_new_session"] = True
    subprocess.Popen([sys.executable, str(ROOT / "sim" / "service.py")], **kw)
    st.session_state["started_service"] = True


# ------------------------------------------------------------------ layout pieces
def live(fn):
    """Run a page body as a fragment that refreshes itself every REFRESH_S seconds (Streamlit 1.37+)."""
    if hasattr(st, "fragment"):
        return st.fragment(run_every=REFRESH_S)(fn)()
    st.button("Refresh")
    return fn()


def service_banner(state):
    """One line on every live page: is the service running, and where is it?"""
    if not state:
        st.info("No simulation service has run on this database yet.")
        c1, c2 = st.columns([1, 3])
        c1.button("Start the service", type="primary", on_click=start_service, key="start_banner")
        c2.caption("or in a terminal (in project/): `python sim/service.py`")
        return False
    ctl = controller_state()
    clk = state["clock"]
    alive = service_alive(state)
    if not alive:
        c1, c2 = st.columns([1, 3])
        c1.button("Start the service", type="primary", on_click=start_service, key="start_banner")
        c2.warning(f"The simulation service is not running (last heartbeat {state.get('heartbeat_at')}). Starting it "
                   "continues where it stopped.")
    if ctl.get("estopped"):
        st.error("EMERGENCY STOP is active: no switching is proposed or applied.")
    if state["status"] == "awaiting_operator":
        st.warning(f"The clock is paused: {state.get('note') or 'a switching proposal awaits the operator'}. "
                   "Decide on the Alerts and decisions page.")
    newest = store().last_step_time(_service(), state["run_id"])        # the newest step, not the last heartbeat
    last = str(pd.Timestamp(newest)) if newest is not None else clk.get("last_time")
    st.caption(f"{'Running' if alive and state['status'] == 'running' else state['status'].replace('_', ' ').capitalize()}"
               f" | simulated time {last[:16] if last else '(not started)'}"
               + (f" ({season_of(last)})" if last else "")
               + f" | {MODES.get(ctl.get('mode'), '-')} | {state['scenario']} | run {state['run_id']}")
    return True


def page_header(title, caption=None):
    st.title(title)
    if caption:
        st.caption(caption)
