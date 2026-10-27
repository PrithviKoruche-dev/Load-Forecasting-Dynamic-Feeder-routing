"""Page H: data quality of the measurement streams (Phase 8, gap item C5): transformer meters, smart meters
and the head-end's estimates, and how far the operator's view was from the true loads (the simulator knows both).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import plots  # noqa: E402

C.page_header("Data quality", "Where each transformer's load came from in the step just completed, how the smart "
              "meters are reporting, and how far what the operator saw was from the true load.")
SOURCES = {"measured": "transformer meter", "ami-estimate": "smart meters (interval before)",
           "forecast": "forecast (no reading)", "last-value": "last value seen", "missing": "nothing"}
ESTIMATE = "ami-estimate"


@st.cache_data
def head_end_settings(city):
    sys.path.insert(0, str(C.ROOT / "city"))
    from demand import load_spec
    from meters import settings
    return settings(load_spec(city))["head_end"]


def tx_rows(last, est):
    q = last.get("data_quality") or {}
    seen, true = last.get("observed_pct") or {}, last.get("measured_pct") or {}
    base = C.tx_table().set_index("transformer")
    rows = []
    for tx in C.TX:
        aid = base.loc[tx, "asset_id"]
        e = est[est["asset_id"] == aid].sort_values("time")
        recent = " ".join({"good": "G", "estimated": "E", "poor": "P", "missing": "-"}.get(x, "?")
                          for x in e["quality"].tail(8))
        rows.append({"transformer": tx, "asset ID": aid, "value from": SOURCES.get(q.get(tx), q.get(tx) or "-"),
                     "seen %": seen.get(tx), "true %": true.get(tx),
                     "error pts": None if seen.get(tx) is None or true.get(tx) is None else round(seen[tx] - true[tx], 2),
                     "smart-meter estimates, last 2 h": recent or "-"})
    return pd.DataFrame(rows)


def body():
    state = C.sim_state()
    if not C.service_banner(state):
        return
    last = C.last_step(state)
    if last is None:
        st.info("No step yet in this run.")
        return
    m = last.get("meter_stats")
    if not m:
        st.info("The operator sees the true loads in this configuration (perfect measurement): there are no meter "
                "streams to show. A synthetic city with `meters: enabled: true` in its city.yaml has them.")
        return
    s = C.store()
    n_tx = len(C.TX)
    q = last.get("data_quality") or {}
    k = st.columns(6)
    k[0].metric("Transformer meters reporting", f"{n_tx - m['dt_missing']} / {n_tx}")
    k[1].metric("Not metered now", sum(v != "measured" for v in q.values()),
                help="transformers whose value came from the smart meters, the forecast or the last value")
    k[2].metric("Smart-meter readings this step", f"{m['readings']:,}",
                help=f"{m['late_readings']:,} of them for intervals before the last one")
    k[3].metric("Readings in transit", f"{m['awaited']:,}", help="sent but not yet received by the head-end")
    k[4].metric("Suspect meters", m["suspect_meters"], help="flagged as stuck: left out of the estimates")
    k[5].metric("Concentrators down", m["concentrators_down"])
    aq = m.get("ami_quality") or {}
    he = head_end_settings(C.SIM["city"])
    st.caption("Smart-meter estimate of the interval before, per transformer: "
               + (", ".join(f"{v} {k_}" for k_, v in sorted(aq.items())) or "none yet") + ". Good: at least "
               f"{100 * he['good_share']:g}% of the expected load has reported; estimated: at least "
               f"{100 * he['estimated_share']:g}%; poor: less.")

    t = pd.Timestamp(last["sim_time"])
    est = s.measurements(ESTIMATE, sim_from=t - pd.Timedelta(hours=2), sim_to=t)
    st.subheader("Transformers")
    st.dataframe(tx_rows(last, est), hide_index=True, key="dq_table")
    st.caption("Seen % is the loading the operator saw; true % is what the network carried (the simulator's truth, "
               "used to judge the outcome). Smart-meter estimates, oldest to newest: G good, E estimated, P poor, "
               "- missing; they are revised as late readings arrive.")

    span = st.radio("Period", ["Last 24 hours", "Last 7 days"], horizontal=True, key="dq_span")
    hist = C.run_steps(state, limit={"Last 24 hours": 96, "Last 7 days": 672}[span])
    hist = hist[hist["meter_stats"].notna()] if "meter_stats" in hist else hist.iloc[0:0]
    if not hist.empty:
        st.subheader("Over time")
        ms = pd.DataFrame(list(hist["meter_stats"]))
        not_metered = [sum(v != "measured" for v in (d or {}).values()) for d in hist["data_quality"]]
        err = [None if not a or not b else max(abs(a[x] - b[x]) for x in a if x in b)
               for a, b in zip(hist["observed_pct"], hist["measured_pct"])]
        st.plotly_chart(plots.lines_chart(hist["sim_time"], [("transformers not metered", not_metered, "#d62728", None),
                                                             ("suspect meters", ms["suspect_meters"], "#9467bd", None),
                                                             ("largest seen - true error, pts", err, "#1f77b4", "dot")],
                                          y_title="count / points"), key="dq_counts")
        st.plotly_chart(plots.lines_chart(hist["sim_time"], [("late readings", ms["late_readings"], "#ff7f0e", None),
                                                             ("in transit", ms["awaited"], "#8a8f98", "dot")],
                                          y_title="readings"), key="dq_readings")

    st.subheader("Exceptions")
    r = s.meter_readings(sim_from=t - pd.Timedelta(hours=6), sim_to=t)
    if r.empty:
        st.caption("No exception readings stored in the last 6 hours (late or from a suspect meter).")
        return
    reg = s.meters()
    if not reg.empty:
        r = r.merge(reg[["meter_id", "transformer_asset_id", "consumer_type"]].rename(
            columns={"transformer_asset_id": "transformer_id"}), on="meter_id", how="left")
    sus = r[r["flag"] == "suspect"].drop_duplicates("meter_id")
    c1, c2 = st.columns(2)
    c1.metric("Late readings, last 6 h", f"{int((r['flag'] == 'ok').sum()):,}",
              help="received more than one interval after their own")
    c2.metric("Suspect meters reporting, last 6 h", len(sus))
    if not sus.empty:
        st.dataframe(sus[[c for c in ("meter_id", "transformer_id", "consumer_type", "kw", "interval_time")
                          if c in sus]].rename(columns={"kw": "repeated kW", "interval_time": "last interval"}),
                     hide_index=True, key="dq_suspect")
    late = r[r["flag"] == "ok"]
    if not late.empty and "transformer_id" in late:
        by = (late.groupby("transformer_id")["delay_s"].agg(["count", "median", "max"])
              .rename(columns={"count": "late readings", "median": "median delay s", "max": "longest delay s"})
              .sort_values("late readings", ascending=False).head(10).round(0))
        st.caption("Transformers with the most late readings in the last 6 hours (a data concentrator outage holds "
                   "back every reading behind it):")
        st.dataframe(by, key="dq_late")


C.live(body)
