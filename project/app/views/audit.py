"""Page G: event and audit log with filters (gap item G9). Everything is read from the database, so it survives
restarts. Choose the live network (the simulation service) or the what-if sandbox."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from core import ids  # noqa: E402

C.page_header("Audit log", "Newest first. Stored in the database; survives restarts.")
SOURCES = {"Live network (simulation service)": C.SERVICE, "What-if sandbox": "default"}
TYPES = ["mode", "recommendation", "approved", "applied", "rejected", "superseded", "manual", "rollback",
         "no_safe_option", "emergency_stop", "emergency_stop_cleared", "service", "clock"]


def body():
    s = C.store()
    src = st.radio("Log of", list(SOURCES), horizontal=True, key="audit_src")
    who = SOURCES[src]
    tab_ev, tab_rec, tab_sw, tab_al, tab_cmd = st.tabs(["Events", "Recommendations", "Switching operations", "Alerts",
                                                        "Commands"])
    with tab_ev:
        days = sorted({(ev["sim_time"] or "")[:10] for ev in s.events(controller=who, limit=20000)} - {""})
        f1, f2, f3, f4 = st.columns(4)
        day = f1.selectbox("Day (simulated)", ["All days"] + days, key="audit_day")
        tx = f2.selectbox("Transformer", ["All"] + C.TX, key="audit_tx")
        types = f3.multiselect("Event type", TYPES, key="audit_types")
        sev = f4.multiselect("Severity", ["info", "warning", "critical"], key="audit_sev")
        rows = s.events(controller=who, sim_from=None if day == "All days" else day,
                        sim_to=None if day == "All days" else day, types=types or None, severities=sev or None,
                        asset_id=None if tx == "All" else ids.transformer_id(tx), limit=1000)
        st.caption(f"{len(rows)} events" + (" (showing the newest 1,000)" if len(rows) == 1000 else ""))
        if rows:
            st.dataframe(pd.DataFrame([{"#": r["seq"], "sim time": r["sim_time"] or "", "wall time": r["wall_time"],
                                        "severity": r["severity"], "event": r["type"], "assets": ", ".join(r["assets"]),
                                        "mode": r["mode"], "details": r["message"]} for r in rows]), hide_index=True)
        else:
            st.caption("No events match these filters.")
    with tab_rec:
        recs = s.recommendations(controller=who, limit=1000)
        if recs.empty:
            st.caption("No recommendations yet.")
        else:
            st.dataframe(recs.rename(columns={"rec_id": "id", "for_time": "for", "losses_before_kw": "losses before kW",
                                              "losses_after_kw": "losses after kW", "compute_s": "s"}), hide_index=True)
            pick = st.selectbox("Candidates of", recs["rec_id"].tolist(), key="audit_rec")
            cands = s.candidates(pick)
            if not cands.empty:
                st.dataframe(cands[["cand_id", "round", "status", "label", "reason", "max_loading", "losses_kw"]],
                             hide_index=True)
    with tab_sw:
        acts = s.actions(controller=who, limit=2000)
        if acts.empty:
            st.caption("No switch has been operated yet.")
        else:
            st.dataframe(acts[["sim_time", "rec_id", "how", "step", "switch", "switch_id", "operation", "action_key"]],
                         hide_index=True)
    with tab_al:
        prefix = "service " if who == C.SERVICE else None
        al = s.alerts(limit=1000, context_prefix=prefix)
        if who != C.SERVICE and not al.empty:
            al = al[~al["context"].fillna("").str.startswith("service ")]
        if al.empty:
            st.caption("No alerts recorded yet.")
        else:
            st.dataframe(al[["sim_time", "asset", "kind", "level", "value", "text"]], hide_index=True)
    with tab_cmd:
        if who != C.SERVICE:
            st.caption("The sandbox takes no commands: it is operated directly on its page.")
        else:
            cm = s.commands(C.SERVICE, limit=500)
            if cm.empty:
                st.caption("No commands yet.")
            else:
                st.dataframe(cm[["id", "created_at", "source", "command", "args", "status", "result", "done_at"]],
                             hide_index=True)


body()
