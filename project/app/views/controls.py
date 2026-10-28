"""Page F: simulation controls (gap item G8). Every control starts from the value in force in the service, and
every button sends a command to the service, which records whether it was done or refused and why."""
import sys
from datetime import datetime, time as dtime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from control import service_alive  # noqa: E402
from core import scenarios  # noqa: E402

C.page_header("Controls", "Steer the simulation service. Settings below show what is in force now.")
SPEEDS = {"1 simulated hour per second (3600x)": 3600, "1 simulated day per second (86400x)": 86400,
          "Accelerated: 1 simulated year per 24 h": "accelerated", "As fast as possible": "max"}


def status():
    state = C.sim_state()
    C.command_feedback()
    C.service_banner(state)


C.live(status)
state = C.sim_state()
if state:
    ctl = C.controller_state()
    sets = C.settings(state)
    clk = state["clock"]
    alive = service_alive(state)

    a, b, c = st.columns(3)
    with a:
        st.subheader("Service")
        if alive:
            st.success(f"Running (pid {state.get('pid')})")
            st.button("Stop the service", on_click=C.send, args=("stop",))
        else:
            st.warning("Not running")
            st.button("Start the service", type="primary", on_click=C.start_service)
        st.subheader("Clock")
        st.caption(f"Status **{state['status'].replace('_', ' ')}** · last step {(clk.get('last_time') or '-')[:16]} · "
                   f"next {clk['next_time'][:16]}")
        b1, b2, b3 = st.columns(3)
        finished = state["status"] == "finished"
        b1.button("Resume", on_click=C.send, args=("resume",), disabled=clk.get("running", False) or finished)
        b2.button("Pause", on_click=C.send, args=("pause",), disabled=not clk.get("running", False))
        b3.button("Step", on_click=C.send, args=("step",), disabled=clk.get("running", False) or finished,
                  help="run one step while paused")
        def same(v, w):
            try:
                return float(v) == float(w)
            except (TypeError, ValueError):
                return str(v) == str(w)
        current = next((k for k, v in SPEEDS.items() if same(v, clk["speed"])), None)
        options = list(SPEEDS) + ([f"Current: {float(clk['speed']):,.0f}x"] if current is None else [])
        pick = st.selectbox("Speed", options, index=options.index(current or options[-1]), key=f"speed_{clk['speed']}")
        custom = st.number_input("or simulated seconds per wall second", min_value=1.0, value=None,
                                 placeholder="e.g. 36000", key="speed_custom")
        st.button("Set speed", on_click=lambda: C.send("speed", speed=custom if custom else SPEEDS.get(pick, clk["speed"])))
        af = state.get("achieved_factor")
        st.caption(f"Achieved {af:,.0f}x" if af else "Achieved speed: measured once the clock runs")
    with b:
        st.subheader("Time")
        start, end = pd.Timestamp(clk["start"]), pd.Timestamp(clk["end"])
        nxt = min(max(pd.Timestamp(clk["next_time"]), start), end)   # after the last step of a run, next is past the end
        if state["status"] == "finished":
            st.info("This run has reached the end of the data. Pick a date and hour, then start a new run.")
        st.session_state.setdefault("jump_date", nxt.date())          # kept while the clock runs on
        st.session_state.setdefault("jump_hour", nxt.hour)
        if not start.date() <= st.session_state["jump_date"] <= end.date():
            st.session_state["jump_date"] = nxt.date()
        d = st.date_input("Date", min_value=start.date(), max_value=end.date(), key="jump_date")
        h = st.number_input("Hour", 0, 23, key="jump_hour")
        when = f"{datetime.combine(d, dtime(int(h))):%Y-%m-%d %H:%M}"
        st.button("Jump to this time", on_click=lambda: C.send("jump", time=when),
                  help="continue the current run from this hour")
        st.button("Start a new run from this time", on_click=lambda: C.send("restart", **{"from": when}),
                  help="a new run that starts running at once (earlier runs stay in the database); switches return to normal")
        st.caption(f"Simulation range {clk['start'][:16]} to {clk['end'][:16]} "
                   + ("(the synthetic city's year)." if C.SIM.get("city") else "(the eMARC test period)."))
        st.subheader("Scenario and season")
        names = [s_.name for s_ in scenarios.benchmark()]
        sc = st.selectbox("Scenario", names, index=names.index(state["scenario"]) if state["scenario"] in names else 0,
                          key=f"scen_{state['scenario']}")
        st.button("Set scenario", on_click=lambda: C.send("scenario", name=sc))
        last = clk.get("last_time") or clk["next_time"]
        st.markdown(f"Season now: **{C.season_of(last)}**")
        st.caption("Season calendar (config/simulation.yaml): "
                   + "; ".join(f"{k} {', '.join(datetime(2000, m, 1).strftime('%b') for m in v)}"
                               for k, v in C.SIM["seasons"].items())
                   + (". The synthetic city uses it to shape demand (profiles by season)." if C.SIM.get("city") else
                      ". With the real eMARC data the season is a label; a synthetic city uses it to shape demand."))
    with c:
        st.subheader("Operation")
        modes = list(C.MODES)
        mode = st.radio("Operating mode", modes, index=modes.index(ctl.get("mode", "recommend")),
                        format_func=C.MODES.get, key=f"mode_{ctl.get('mode')}", disabled=ctl.get("estopped", False))
        st.button("Set mode", on_click=lambda: C.send("mode", mode=mode), disabled=ctl.get("estopped", False))
        pol = st.toggle("Pause the clock for each proposal (recommend mode)", value=sets["pause_on_recommendation"],
                        key=f"policy_{sets['pause_on_recommendation']}",
                        help="Off: the clock keeps running and a proposal not decided before the next hour is "
                             "superseded.")
        st.button("Set approval policy", on_click=lambda: C.send("policy", pause_on_recommendation=pol))
        _thr = C.OPS["dashboard"]["threshold_slider"]
        thr = st.slider("Overload threshold (%)", _thr["min"], _thr["max"], value=int(sets["threshold"]),
                        key=f"thr_{sets['threshold']}")
        st.button("Set threshold", on_click=lambda: C.send("threshold", value=thr))
        mv = st.checkbox("Allow MV feeder transfers", value=bool(sets["use_mv"]), key=f"mv_{sets['use_mv']}")
        st.button("Set MV transfers", on_click=lambda: C.send("use_mv", value=mv))
        st.subheader("Switching")
        st.caption(f"Configuration now: **{', '.join(ctl.get('active') or []) or 'normal'}**")
        r1, r2 = st.columns(2)
        r1.button("Roll back", on_click=C.send, args=("rollback",))
        r2.button("Reset to normal", on_click=C.send, args=("reset",))
        if ctl.get("estopped"):
            st.button("Clear emergency stop", on_click=C.send, args=("clear_estop",))
        else:
            st.button("EMERGENCY STOP", type="primary", on_click=C.send, args=("estop",))

if state:
    st.subheader("Recent commands")
    cm = C.store().commands(C.SERVICE, limit=10)
    if not cm.empty:
        st.dataframe(cm[["id", "created_at", "source", "command", "args", "status", "result"]], hide_index=True)
