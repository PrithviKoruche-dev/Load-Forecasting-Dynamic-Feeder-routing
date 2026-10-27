"""Page D: forecasting performance (gap item G6): accuracy so far, MAE and RMSE over time by transformer and
model version, pending forecasts, and forecast against actual with the P10-P90 band."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
import plots  # noqa: E402

C.page_header("Forecasting", f"How good the {C.FC_AHEAD} forecasts have been, up to the simulated time.")
COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#222222"]


def over_time(err, freq, metric):
    """MAE or RMSE per period, per transformer and pooled."""
    err = err.assign(period=err["timestamp"].dt.to_period(freq).dt.start_time)   # calendar days or weeks
    f = (lambda e: e.abs().mean()) if metric == "MAE" else (lambda e: float(np.sqrt((e ** 2).mean())))
    per = err.groupby(["period", "transformer"])["error_kw"].apply(f).unstack()
    per["ALL"] = err.groupby("period")["error_kw"].apply(f)
    return per


def body():
    state = C.sim_state()
    if not C.service_banner(state):
        return
    s = C.store()
    until = state["clock"].get("last_time")
    if not until:
        st.info("No step yet in this run: nothing has been scored.")
        return
    versions = s.model_versions()
    default = state.get("forecast_source") == "live" and f"{s.import_cfg['model_version']}-live" in versions
    mv = st.selectbox("Model version", versions, key="fc_version",
                      index=versions.index(f"{s.import_cfg['model_version']}-live") if default else
                      versions.index(s.import_cfg["model_version"]) if s.import_cfg["model_version"] in versions else 0)
    perf = s.forecast_performance(until=until, model_version=mv)
    pending = s.forecast_count_after(until, model_version=mv)
    allrow = perf.set_index("transformer").loc["ALL"] if not perf.empty and "ALL" in set(perf["transformer"]) else None
    k = st.columns(5)
    k[0].metric("Scored", f"{int(allrow['scored']):,}" if allrow is not None else 0,
                help=f"forecasts whose hour has passed (up to {until[:16]})")
    k[1].metric("Pending", f"{pending:,}", help="forecasts for hours still to come")
    k[2].metric("MAE", f"{allrow['MAE_kW']:.1f} kW" if allrow is not None and allrow["scored"] else "-")
    k[3].metric("nRMSE", f"{allrow['nRMSE_%']:.1f} %" if allrow is not None and allrow["scored"] else "-",
                help="RMSE as a % of the mean actual load")
    k[4].metric("Bias", f"{allrow['bias_kW']:+.1f} kW" if allrow is not None and allrow["scored"] else "-",
                help="positive = over-forecasting")
    if allrow is None or not allrow["scored"]:
        st.info("No forecast of this model version has been scored yet.")
        return
    st.dataframe(perf.round(2), hide_index=True)

    err = s.forecast_errors(until=until, model_version=mv)
    c1, c2 = st.columns(2)
    metric = c1.radio("Metric", ["MAE", "RMSE"], horizontal=True, key="fc_metric")
    freq = {"Daily": "D", "Weekly": "W"}[c2.radio("Over", ["Daily", "Weekly"], horizontal=True, key="fc_freq")]
    per = over_time(err, freq, metric)
    st.subheader(f"{metric} over time (kW)")
    st.plotly_chart(plots.lines_chart(per.index, [(c, per[c], COLOURS[i % len(COLOURS)], "dash" if c == "ALL" else None)
                                                  for i, c in enumerate(per.columns)], y_title=f"{metric} kW"),
                    key="fc_over_time")

    st.subheader("Forecast against actual")
    tx = st.selectbox("Transformer", C.TX, key="fc_tx")
    days = st.slider("Days shown", 1, 14, 3, key="fc_days")
    end = pd.Timestamp(until)
    d = err[(err["transformer"] == tx) & (err["timestamp"] > end - pd.Timedelta(days=days))]
    band = None
    frame = s.forecast_frame(model_version=s.import_cfg["model_version"])
    if not frame.empty and {"p10_kw", "p90_kw"} <= set(frame.columns) and mv == s.import_cfg["model_version"]:
        b = frame[(frame["transformer_id"] == tx) & frame["timestamp"].isin(d["timestamp"])].set_index("timestamp")
        b = b.reindex(d["timestamp"])
        band = ("P10-P90 band", b["p10_kw"].to_numpy(), b["p90_kw"].to_numpy())
    st.plotly_chart(plots.lines_chart(d["timestamp"], [("actual", d["actual_kw"], "#1f77b4", None),
                                                       ("forecast", d["predicted_kw"], "#ff7f0e", None)],
                                      y_title="kW", band=band), key="fc_vs_actual")
    st.caption("Loads before any scenario is applied (the forecast is scored against what the meters measured).")


C.live(body)
