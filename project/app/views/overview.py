"""Page A: overview of the live network (gap item G3)."""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import plots  # noqa: E402

C.page_header("Overview", "The live network as the simulation service runs it.")


def body():
    state = C.sim_state()
    C.command_feedback()
    if not C.service_banner(state):
        return
    s = C.store()
    assets = s.assets()
    n = assets["kind"].value_counts() if not assets.empty else pd.Series(dtype=int)
    meters = len(s.meters())
    capacity = float(C.tx_table()["rating_kva"].sum())
    st.subheader("The network")
    k = st.columns(6)
    k[0].metric("Substations", int(n.get("substation", 0)))
    k[1].metric("Feeders", int(n.get("feeder", 0)))
    k[2].metric("Transformers", int(n.get("transformer", 0)))
    k[3].metric("Switches", int(n.get("switch", 0)))
    k[4].metric("Smart meters", f"{meters:,}" if meters else "-",
                help="from data/processed/meter_registry.csv (written by ml/prepare_data.py)"
                     + ("" if meters else "; not imported on this computer"))
    k[5].metric("Transformer capacity", f"{capacity:,.0f} kVA")

    last = C.last_step(state)
    sets = C.settings(state)
    thr = sets["threshold"]
    st.subheader("Now")
    if last is None:
        st.info("The service has not run a step in this run yet.")
        return
    loads = last.get("loads_kw") or {}
    total = sum(loads.values()) if loads else None
    alerts = s.alerts(sim_from=f"{last['sim_time']:%Y-%m-%d %H:%M}", sim_to=f"{last['sim_time']:%Y-%m-%d %H:%M}",
                      context_prefix="service ")
    active = 0 if alerts.empty else int((~alerts["level"].isin(["normal"])).sum())
    k = st.columns(4)
    k[0].metric("Total load", f"{total:,.0f} kW" if total is not None else "-",
                help=f"measured, all transformers, in the {C.STEP} just completed")
    k[1].metric("Peak loading", f"{last['measured_max_pct']:.1f} %" if last.get("measured_max_pct") is not None else "-",
                delta=None if last.get("measured_max_normal_pct") is None or last["measured_max_pct"] is None
                or abs(last["measured_max_pct"] - last["measured_max_normal_pct"]) < 0.05 else
                f"{last['measured_max_pct'] - last['measured_max_normal_pct']:+.1f} pts by control",
                delta_color="inverse", help="measured; the change shows what the switching in force made")
    k[2].metric("Active alerts", active, help=f"risk and voltage/line alerts in the {C.STEP} just completed")
    k[3].metric("Configuration now", ", ".join(last.get("config_after") or []) or "normal")

    clk = state["clock"]
    af = state.get("achieved_factor")
    speed = clk["speed"]
    target = "as fast as possible" if speed == "max" else (
        f"{C.SIM['clock']['accelerated_factor']:g}x (1 year per 24 h)" if speed == "accelerated" else f"{float(speed):g}x")
    k = st.columns(4)
    k[0].metric("Simulated time", f"{last['sim_time']:%d %b %H:%M}", help=f"{last['sim_time']:%A %d %B %Y, %H:%M}")
    k[1].metric("Season", C.season_of(last["sim_time"]).capitalize())
    k[2].metric("Speed", target, delta=f"achieved {af:,.0f}x" if af else None, delta_color="off")
    k[3].metric("Last power flow", f"{last['wall_time'][11:19]}", help=f"step took {last['duration_ms']:.0f} ms")
    st.caption(f"Wall-clock time {datetime.now():%Y-%m-%d %H:%M:%S} | threshold {thr:g}% | next decision: "
               f"{C.DECISIONS.get(last['decision'], last['decision'])}"
               + (f" ({last['rec_id']})" if last.get("rec_id") else ""))

    hist = C.run_steps(state, limit=48 * C.PER_HOUR)
    if not hist.empty:
        st.subheader("Last 48 simulated hours")
        st.plotly_chart(plots.lines_chart(
            hist["sim_time"], [("measured peak, no control", hist["measured_max_normal_pct"], "#8a8f98", "dot"),
                               ("measured peak, with control", hist["measured_max_pct"], "#1f77b4", None),
                               (f"forecast peak (next {C.STEP})", hist["forecast_max_pct"], "#ff7f0e", None)],
            threshold=thr, y_title="peak transformer loading %"), key="overview_chart")


C.live(body)
