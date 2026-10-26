"""Page E: alerts and decision support (gap item G7): current alerts, the proposal waiting for the operator,
and every recommendation of the live network with its switch plan, before/after, checks and the candidates
that were evaluated (accepted or rejected, with the reason)."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import plots  # noqa: E402

C.page_header("Alerts and decisions")
LEVELS = ["actual_overload", "predicted_overload", "approaching", "elevated", "violation"]


def recommendation_detail(rec, cands, where):
    """where: a prefix that keeps element keys unique (the same recommendation can be shown twice)."""
    {"no_safe_option": st.error, "mitigate": st.warning, "no_change": st.success}.get(rec["kind"], st.info)(rec["summary"])
    st.caption(f"{rec['rec_id']} for {rec['for_time']} · kind `{rec['kind']}` · status `{rec['status']}` · "
               f"computed in {rec['compute_s']} s · {rec['context']}")
    for line in rec.get("binding") or []:
        st.write("- " + line)
    if rec.get("plan"):
        st.markdown("**Switching plan**")
        st.dataframe(pd.DataFrame(rec["plan"])[["step", "switch", "switch_id", "operation", "action"]], hide_index=True)
    if rec.get("before") and rec.get("after") and rec["kind"] != "no_safe_option":
        st.plotly_chart(plots.before_after_chart(rec["before"]["loading_by_tx"], rec["after"]["loading_by_tx"],
                                                 rec["threshold"]), key=f"{where}_ba_{rec['rec_id']}")
        st.dataframe(pd.DataFrame([
            {"": "peak loading %", "before": rec["before"]["max_loading"], "after": rec["after"]["max_loading"]},
            {"": "lowest voltage pu", "before": rec["before"]["min_voltage_pu"], "after": rec["after"]["min_voltage_pu"]},
            {"": "losses kW", "before": rec["losses_before_kw"], "after": rec["losses_after_kw"]}]), hide_index=True)
    if rec.get("confidence"):
        st.markdown(f"Confidence: **{rec['confidence']['label']}** ({rec['confidence']['probability']:.0%})",
                    help=rec["confidence"]["basis"])
    if rec.get("checks"):
        with st.expander("Constraint checks"):
            st.dataframe(pd.DataFrame(rec["checks"]), hide_index=True)
    if not cands.empty:
        st.markdown(f"**Candidates evaluated ({len(cands)})**")
        st.dataframe(cands[["cand_id", "round", "status", "label", "reason", "max_loading", "min_voltage_pu",
                            "losses_kw"]].rename(columns={"cand_id": "id", "label": "option", "max_loading": "peak %",
                                                          "min_voltage_pu": "lowest V pu", "losses_kw": "losses kW"}),
                     hide_index=True)


def body():
    state = C.sim_state()
    C.command_feedback()
    if not C.service_banner(state):
        return
    s = C.store()
    ctl = C.controller_state()
    pend = s.pending_recommendation(C.SERVICE)
    if pend:
        st.subheader(f"Waiting for the operator: {pend['rec_id']}")
        recommendation_detail(pend, s.candidates(pend["rec_id"]), "pending")
        if ctl.get("mode") == "recommend" and not ctl.get("estopped"):
            c1, c2, _ = st.columns([1, 1, 3])
            c1.button("Approve", type="primary", on_click=C.send, args=("approve",), kwargs={"rec_id": pend["rec_id"]},
                      key=f"approve_{pend['rec_id']}")
            c2.button("Reject", on_click=C.send, args=("reject",), kwargs={"rec_id": pend["rec_id"]},
                      key=f"reject_{pend['rec_id']}")
        st.divider()

    last = C.last_step(state)
    st.subheader("Alerts in the hour just completed")
    if last is None:
        st.caption("No step yet.")
    else:
        label = f"{last['sim_time']:%Y-%m-%d %H:%M}"
        al = s.alerts(sim_from=label, sim_to=label, context_prefix="service ")
        if al.empty:
            st.success(f"No alerts at {label}.")
        else:
            st.dataframe(al[["sim_time", "asset", "kind", "level", "value", "text"]], hide_index=True)
    with st.expander("Alert history (this database, live network)"):
        lv = st.multiselect("Levels", LEVELS, default=["actual_overload", "predicted_overload", "violation"],
                            key="dec_levels")
        hist = s.alerts(levels=lv or None, context_prefix="service ", limit=300)
        st.dataframe(hist[["sim_time", "asset", "kind", "level", "value", "text"]] if not hist.empty else hist,
                     hide_index=True)

    st.subheader("Recommendations of the live network")
    recs = s.recommendations(controller=C.SERVICE)
    if recs.empty:
        st.caption("No recommendation yet. In monitor mode none are made; in recommend and automatic mode one is made "
                   "whenever the next hour's forecast breaks a limit or the network is not in its normal configuration.")
        return
    status = st.multiselect("Status", sorted(recs["status"].unique()), key="dec_status")
    shown = recs[recs["status"].isin(status)] if status else recs
    st.dataframe(shown[["rec_id", "for_time", "kind", "status", "summary"]], hide_index=True)
    pick = st.selectbox("Details of", shown["rec_id"].tolist(), key="dec_pick")
    if pick:
        recommendation_detail(s.load_recommendation(pick), s.candidates(pick), "detail")


C.live(body)
