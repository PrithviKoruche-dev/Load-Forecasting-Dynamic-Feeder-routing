"""Page B: network view with selection (gap item G4).

The diagram shows either the hour just measured (its loads, switch states and risk classes) or the next hour
(its forecast, in the configuration now in force). Select a transformer or a feeder, from the list or by
clicking a transformer, to inspect its rating, load, forecast and constraints.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import plots  # noqa: E402

C.page_header("Network", "Select a transformer or feeder to inspect it.")
FEEDERS = {f"Feeder {k} ({v['id']})": k for k, v in C.GRID["feeders"].items()}
OPTIONS = C.TX + list(FEEDERS)
st.session_state.setdefault("net_pick", C.TX[0])


def _fmt(v, unit="%", nd=1):
    return "-" if v is None else f"{v:.{nd}f}{unit}"


def transformer_panel(tx, last, thr):
    info = C.tx_table().set_index("transformer").loc[tx]
    det = (last.get("tx_detail") or {}).get(tx, {})
    g = lambda col: (last.get(col) or {}).get(tx)  # noqa: E731
    risk = (last.get("risk") or {}).get(tx)
    st.markdown(f"### {tx} ({info['asset_id']})")
    st.caption(f"Feeder {info['feeder']} ({info['feeder_id']}), MV bus {info['mv_bus']}, {info['category']}, "
               f"rating **{info['rating_kva']:g} kVA**")
    k = st.columns(3)
    k[0].metric("Measured loading", _fmt(g("measured_pct")), help=f"hour {last['sim_time']:%d %b %H:%M}")
    k[1].metric(f"Forecast for that {C.STEP}", _fmt(g("forecast_pct")), help=f"{C.FC_AHEAD} forecast, same configuration")
    k[2].metric("Upper bound (P90)", _fmt(g("upper_pct")))
    k = st.columns(3)
    k[0].metric("Load", f"{det.get('kw', 0):,.0f} kW" if det else "-", help=f"{det.get('kvar', 0):,.0f} kvar" if det else None)
    k[1].metric("Apparent power", f"{det.get('kva', 0):,.0f} kVA" if det else "-")
    k[2].metric(f"Next {C.STEP} forecast", _fmt(g("next_forecast_pct")), help="in the configuration now in force")
    st.markdown(f"Risk class: **{C.RISK_LABELS.get(risk, '-')}** · threshold {thr:g}% · hard limit {C.HARD_LIMIT:g}%")
    moves = C.moves_of(tx)
    active = set(last.get("config_after") or [])
    received = [k for k in active if k.split(">")[1] == tx]
    lines = [f"- block b can move to **{to}** (`{key}`)" + (" — **in force now**" if key in active else "")
             for key, to in moves]
    lines += [f"- receives the movable block of **{k.split('>')[0]}** (`{k}`)" for k in received]
    st.markdown("**Switching options**\n" + "\n".join(lines))
    hist = C.run_steps(C.sim_state(), limit=72)
    if not hist.empty:
        pick = lambda col: [((r or {}).get(tx)) for r in hist[col]]  # noqa: E731
        st.plotly_chart(plots.lines_chart(hist["sim_time"], [("measured", pick("measured_pct"), "#1f77b4", None),
                                                             ("forecast", pick("forecast_pct"), "#ff7f0e", None)],
                                          threshold=thr, height=240), key="net_tx_hist")


def feeder_panel(label, last):
    f = FEEDERS[label]
    d = (last.get("feeders") or {}).get(f, {})
    st.markdown(f"### {label}")
    if not d:
        st.caption("No feeder data recorded for this step.")
        return
    k = st.columns(3)
    k[0].metric("Load", f"{d['kw']:,.0f} kW", help=f"{d['kvar']:,.0f} kvar, {d['kva']:,.0f} kVA")
    k[1].metric("Head current", f"{d['current_a']:,.0f} A", help=f"{d['loading_pct']:.1f}% of the head line's rating")
    k[2].metric("Lowest voltage", f"{d['min_voltage_pu']:.4f} pu" if d.get("min_voltage_pu") else "-")
    lim = C.OPS["limits"]
    st.markdown(f"Supplies **{', '.join(d['transformers'])}** in this hour · head line loading "
                f"{d['loading_pct']:.1f}% (limit {lim['line_limit_pct']:g}%) · voltage band "
                f"{lim['voltage_min_pu']:g}–{lim['voltage_max_pu']:g} pu")


def body():
    state = C.sim_state()
    if not C.service_banner(state):
        return
    last = C.last_step(state)
    if last is None or not last.get("measured_pct"):
        st.info("No step with network data yet in this run.")
        return
    thr = C.settings(state)["threshold"]
    just, ahead = f"{C.STEP.capitalize()} just measured", f"Next {C.STEP} (forecast)"
    view = st.radio("Show", [just, ahead], horizontal=True, key="net_view")
    if view == just:
        loading, moves, sw = last["measured_pct"], last["config_before"], C.switch_states(last.get("switches"))
        risk = last.get("risk")
        cap = f"Measured loading in the {C.STEP} {last['sim_time']:%d %b %H:%M}, configuration then in force."
    else:
        loading = last.get("next_forecast_pct") or last["measured_pct"]
        moves, sw, risk = last["config_after"], C.switch_states(last.get("switches_after")), None
        cap = f"Forecast loading for the next {C.STEP}, in the configuration now in force (coloured by loading)."
    left, right = st.columns([1.15, 1])
    with left:
        fig = (plots.network_figure(C.GRID, loading, moves, thr, risk, sw) if "layout" in C.GRID   # a synthetic city
               else plots.topology_figure(loading, moves, thr, risk, sw))
        ev = st.plotly_chart(fig, key="net_fig", on_select="rerun", selection_mode="points")
        st.caption(cap + " Click a transformer to inspect it.")
        pts = (ev or {}).get("selection", {}).get("points", []) if isinstance(ev, dict) else \
            getattr(getattr(ev, "selection", None), "points", []) or []
        clicked = next((p.get("customdata") for p in pts if p.get("customdata") in C.TX), None)
        if clicked and clicked != st.session_state.get("net_last_click"):
            st.session_state["net_pick"] = clicked
            st.session_state["net_last_click"] = clicked
    with right:
        pick = st.selectbox("Inspect", OPTIONS, key="net_pick")
        if pick in C.TX:
            transformer_panel(pick, last, thr)
        else:
            feeder_panel(pick, last)
    sw_now = pd.DataFrame([{"switch": k, "state": v, "normal": C.GRID["switches"][k]["normal"]}
                           for k, v in C.switch_states(last.get("switches_after")).items()])
    with st.expander("Switch states now in force"):
        st.dataframe(sw_now, hide_index=True)


C.live(body)
