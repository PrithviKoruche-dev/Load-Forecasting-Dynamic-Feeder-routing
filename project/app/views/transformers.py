"""Page C: transformer monitoring table (gap item G5). Sort by any column (click its header); select a row
(or pick a transformer below) to see its history in this run."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import plots  # noqa: E402

C.page_header("Transformers", "Every transformer in the hour just completed. Click a column header to sort; "
              "select a row to see its history.")
st.session_state.setdefault("tx_pick", C.TX[0])


def table(last):
    base = C.tx_table()
    rows = []
    for _, r in base.iterrows():
        t = r["transformer"]
        g = lambda col: (last.get(col) or {}).get(t)  # noqa: E731
        det = (last.get("tx_detail") or {}).get(t, {})
        seen = last.get("observed_pct")
        row = {"transformer": t, "asset ID": r["asset_id"], "feeder": r["feeder"], "rating kVA": r["rating_kva"],
               "load kW": det.get("kw"), "load kVA": det.get("kva"),
               "loading %": seen.get(t) if seen else g("measured_pct"),
               "forecast %": g("forecast_pct"), "P90 %": g("upper_pct"), "next hour %": g("next_forecast_pct"),
               "risk class": C.RISK_LABELS.get((last.get("risk") or {}).get(t), "-")}
        if seen:                                                   # meters (Phase 8): the operator's view and the truth
            row["data"] = DATA.get((last.get("data_quality") or {}).get(t), "-")
            row["true %"] = g("measured_pct")
        rows.append({**row, "time": f"{last['sim_time']:%Y-%m-%d %H:%M}"})
    return pd.DataFrame(rows)


DATA = {"measured": "meter", "ami-estimate": "smart meters", "forecast": "forecast", "last-value": "last value",
        "missing": "none"}


def body():
    state = C.sim_state()
    if not C.service_banner(state):
        return
    last = C.last_step(state)
    if last is None or not last.get("measured_pct"):
        st.info("No step with transformer data yet in this run.")
        return
    thr = C.settings(state)["threshold"]
    df = table(last)
    ev = st.dataframe(df, hide_index=True, key="tx_table", on_select="rerun", selection_mode="single-row",
                      column_config={"loading %": st.column_config.ProgressColumn(
                          "loading %", min_value=0, max_value=120, format="%.1f")})
    rows = getattr(getattr(ev, "selection", None), "rows", None) or []
    if rows and df.iloc[rows[0]]["transformer"] != st.session_state.get("tx_last_row"):
        st.session_state["tx_pick"] = st.session_state["tx_last_row"] = df.iloc[rows[0]]["transformer"]
    if last.get("observed_pct"):
        st.caption("With meters, loading % is what the operator saw (data: where it came from, see the Data quality "
                   "page) and true % what the network carried; load kW is the true load.")
    st.caption(f"Loading % is measured, in the configuration in force during the hour; forecast % is the 24 h-ahead "
               f"forecast for the same hour; next hour % is the forecast for the coming hour in the configuration now "
               f"in force. Threshold {thr:g}%, hard limit {C.HARD_LIMIT:g}%.")

    tx = st.selectbox("History of", C.TX, key="tx_pick")
    span = st.radio("Period", ["Last 48 hours", "Last 7 days", "Whole run"], horizontal=True, key="tx_span", index=1)
    limit = {"Last 48 hours": 48, "Last 7 days": 168, "Whole run": None}[span]
    hist = C.run_steps(state, limit=limit)
    if hist.empty:
        return
    pick = lambda col: [((r or {}).get(tx)) for r in hist[col]]  # noqa: E731
    seen = ([("seen by the operator", pick("observed_pct"), "#2ca02c", "dot")]
            if "observed_pct" in hist and hist["observed_pct"].notna().any() else [])
    st.plotly_chart(plots.lines_chart(
        hist["sim_time"], [("measured loading", pick("measured_pct"), "#1f77b4", None),
                           ("forecast", pick("forecast_pct"), "#ff7f0e", None)] + seen,
        threshold=thr, band=("P90 upper bound", pick("forecast_pct"), pick("upper_pct"))), key="tx_hist")
    over = sum(1 for v in pick("measured_pct") if v is not None and v > thr)
    st.caption(f"{tx}: {over} hours measured above {thr:g}% in this period; moved block in force during "
               f"{sum(1 for c in hist['config_before'] if any(k.startswith(tx + '>') for k in (c or [])))} hours.")


C.live(body)
