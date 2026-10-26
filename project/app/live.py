"""Live view of the background simulation service (Phase 5, gap item G11).

    python sim/service.py            # terminal 1: the service runs the closed loop
    streamlit run app/live.py        # terminal 2: this page

Everything shown here is read from the database, and every button only queues a command for the service
(the commands table). The service is the one authoritative state, so several browsers, the command line
(sim/control.py) and this page all see and steer the same simulation. Nothing is kept per browser session
except which command was sent last. The replay dashboard (app/app.py) is separate: its own controller,
its own time slider. Phase 6 merges both into one multi-page dashboard.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "grid", ROOT / "sim"):
    sys.path.insert(0, str(p))
from core import config, scenarios  # noqa: E402
from control import service_alive  # noqa: E402
from storage import Store  # noqa: E402

st.set_page_config(page_title="Live simulation", layout="wide")
SIM = config.simulation()
SERVICE = SIM["orchestrator"]["controller"]
THR = config.operations()["limits"]["overload_threshold_pct"]
MODES = {"monitor": "Monitor only", "recommend": "Recommend (operator approves)", "auto": "Automatic"}
SPEEDS = {"1 h per second (3600x)": 3600, "1 day per second (86400x)": 86400,
          "Accelerated: 1 year per 24 h (365x)": "accelerated", "As fast as possible": "max"}
REFRESH_S = 2


@st.cache_resource
def store(url):
    return Store(url)


STORE = store(config.database_url())


def send(command, **args):
    st.session_state["last_cmd"] = STORE.send_command(SERVICE, command, args, source="live dashboard")


def last_command_note():
    cid = st.session_state.get("last_cmd")
    if not cid:
        return
    c = STORE.command(cid)
    if c is None:
        return
    text = f"`{c['command']}` {c['status']}" + (f": {c['result']}" if c.get("result") else "")
    {"done": st.success, "pending": st.info}.get(c["status"], st.error)(text)


def body():
    state = STORE.load_sim_state(SERVICE)
    ctl = STORE.load_state(SERVICE) or {}
    st.title("Live simulation service")
    if not state:
        st.info("No simulation service has run on this database yet. Start one in a terminal:\n\n"
                "`python sim/service.py`")
        return
    alive = service_alive(state)
    clk = state["clock"]
    if not alive:
        st.warning(f"The service is not running (last heartbeat {state.get('heartbeat_at')}). Start it with "
                   "`python sim/service.py`; it continues where it stopped. Commands sent now wait for it.")
    if ctl.get("estopped"):
        st.error("EMERGENCY STOP is active: no switching is proposed or applied.")
    last_command_note()

    af = state.get("achieved_factor")
    target = clk["speed"]
    k = st.columns(6)
    k[0].metric("Status", state["status"].replace("_", " "))
    k[1].metric("Simulated time (last step)", (clk.get("last_time") or "-")[:16])
    k[2].metric("Speed", "max" if target == "max" else
                f"{SIM['clock']['accelerated_factor']:g}x" if target == "accelerated" else f"{float(target):g}x")
    k[3].metric("Achieved", f"{af:,.0f}x" if af else "-", help="measured simulated seconds per wall-clock second")
    k[4].metric("Mode", MODES.get(ctl.get("mode"), "-"))
    k[5].metric("Configuration", ", ".join(ctl.get("active") or []) or "normal")
    st.caption(f"Run {state['run_id']} | {state['scenario']} | forecasts: {state.get('forecast_source')} | "
               f"range {clk['start'][:16]} to {clk['end'][:16]} | lag {state.get('lag_s') or 0:.2f} s | "
               f"heartbeat {state.get('heartbeat_at')}" + (f" | {state['note']}" if state.get("note") else ""))

    # ------------------------------------------------------------------ pending proposal
    pend = STORE.pending_recommendation(SERVICE)
    if pend:
        st.subheader(f"Awaiting the operator: {pend['rec_id']} for {pend['for_time']}")
        {"mitigate": st.warning}.get(pend["kind"], st.info)(pend["summary"])
        if pend.get("plan"):
            st.dataframe(pd.DataFrame(pend["plan"])[["step", "switch", "switch_id", "operation", "action"]],
                         hide_index=True)
        c1, c2 = st.columns(2)
        c1.button("Approve", type="primary", on_click=send, args=("approve",), kwargs={"rec_id": pend["rec_id"]})
        c2.button("Reject", on_click=send, args=("reject",), kwargs={"rec_id": pend["rec_id"]})

    # ------------------------------------------------------------------ steps
    steps = STORE.steps(SERVICE, run_id=state["run_id"], limit=168, newest_first=True)
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Last 7 simulated days")
        if steps.empty:
            st.caption("No steps yet in this run.")
        else:
            d = steps.iloc[::-1]
            fig = go.Figure()
            fig.add_scatter(x=d["sim_time"], y=d["measured_max_normal_pct"], name="measured peak, no control",
                            line={"dash": "dot", "color": "#888"})
            fig.add_scatter(x=d["sim_time"], y=d["measured_max_pct"], name="measured peak, with control",
                            line={"color": "#1f77b4"})
            fig.add_scatter(x=d["sim_time"] + pd.Timedelta(hours=1), y=d["forecast_max_pct"],
                            name="forecast peak (next hour)", line={"color": "#ff7f0e", "width": 1})
            fig.add_hline(y=THR, line_dash="dash", line_color="red", annotation_text=f"{THR:g}% threshold")
            fig.update_layout(height=320, margin={"l": 10, "r": 10, "t": 10, "b": 10}, yaxis_title="peak loading %",
                              legend={"orientation": "h", "y": -0.2})
            st.plotly_chart(fig)
    with right:
        st.subheader("This run so far")
        allsteps = STORE.steps(SERVICE, run_id=state["run_id"])
        if not allsteps.empty:
            dec = allsteps["decision"].value_counts()
            st.dataframe(pd.DataFrame([
                {"": "steps", "value": f"{len(allsteps)}"},
                {"": "measured transformer-hours over threshold, no control",
                 "value": f"{int(allsteps['measured_over_normal'].fillna(0).sum())}"},
                {"": "measured transformer-hours over threshold, with control",
                 "value": f"{int(allsteps['measured_over'].fillna(0).sum())}"},
                {"": "reroutes / restores / mitigations",
                 "value": f"{dec.get('reroute', 0)} / {dec.get('restore', 0)} / {dec.get('mitigate', 0)}"},
                {"": "mean step time (ms)", "value": f"{allsteps['duration_ms'].mean():.1f}"}]),
                hide_index=True)
    if not steps.empty:
        st.dataframe(steps.head(12)[["sim_time", "decision", "rec_id", "config_after", "forecast_max_pct",
                                     "measured_max_pct", "measured_max_normal_pct", "alerts", "duration_ms"]],
                     hide_index=True)

    # ------------------------------------------------------------------ events and commands
    a, b = st.columns(2)
    with a:
        st.subheader("Events")
        ev = STORE.events(controller=SERVICE, limit=15)
        if ev:
            st.dataframe(pd.DataFrame([{"sim time": r["sim_time"] or "", "severity": r["severity"], "event": r["type"],
                                        "details": r["message"]} for r in ev]), hide_index=True)
    with b:
        st.subheader("Commands")
        cm = STORE.commands(SERVICE, limit=10)
        if not cm.empty:
            st.dataframe(cm[["id", "created_at", "source", "command", "status", "result"]], hide_index=True)


with st.sidebar:
    st.header("Clock")
    c1, c2, c3 = st.columns(3)
    c1.button("Resume", on_click=send, args=("resume",))
    c2.button("Pause", on_click=send, args=("pause",))
    c3.button("Step", on_click=send, args=("step",), help="one step while paused")
    st.selectbox("Speed", list(SPEEDS), key="speed_choice")
    st.button("Set speed", on_click=lambda: send("speed", speed=SPEEDS[st.session_state["speed_choice"]]))
    st.text_input("Jump to (YYYY-MM-DD HH:MM)", key="jump_to", placeholder="2020-02-29 18:00")
    st.button("Jump", on_click=lambda: send("jump", time=st.session_state["jump_to"]))
    st.button("Restart (new run from the start)", on_click=send, args=("restart",))
    st.header("Operation")
    st.selectbox("Operating mode", list(MODES), format_func=MODES.get, key="mode_choice")
    st.button("Set mode", on_click=lambda: send("mode", mode=st.session_state["mode_choice"]))
    st.selectbox("Scenario", [s.name for s in scenarios.benchmark()], key="scenario_choice")
    st.button("Set scenario", on_click=lambda: send("scenario", name=st.session_state["scenario_choice"]))
    r1, r2 = st.columns(2)
    r1.button("Roll back", on_click=send, args=("rollback",))
    r2.button("Reset switches", on_click=send, args=("reset",))
    st.button("EMERGENCY STOP", type="primary", on_click=send, args=("estop",))
    st.button("Clear emergency stop", on_click=send, args=("clear_estop",))
    st.divider()
    st.button("Stop the service", on_click=send, args=("stop",))

if hasattr(st, "fragment"):
    st.fragment(run_every=REFRESH_S)(body)()
else:                                               # Streamlit < 1.37: refresh by hand
    st.button("Refresh")
    body()
