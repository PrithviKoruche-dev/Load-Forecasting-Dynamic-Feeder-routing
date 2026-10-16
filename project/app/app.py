"""Streamlit operator dashboard: forecast -> overload alert -> OpenDSS -> reroute."""
import faulthandler
import importlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grid"))
sys.path.insert(0, str(Path(__file__).parent))
import plots            # noqa: E402

st.set_page_config(page_title="Feeder operator dashboard", layout="wide")


@st.cache_resource
def dss_worker():
    """OpenDSS is not thread-safe and Streamlit starts a new thread on every rerun.
    So the engine is imported and used ONLY inside one permanent worker thread."""
    crash_log = open(ROOT / "dss_crash.log", "a")          # written if the process dies natively
    faulthandler.enable(file=crash_log, all_threads=True)
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dss")
    engine = ex.submit(importlib.import_module, "engine").result()
    return ex, engine, crash_log


EXEC, e, _CRASH_LOG = dss_worker()


def dss(fn, *args):
    """Run fn(*args) on the dedicated OpenDSS thread and wait for the result."""
    return EXEC.submit(fn, *args).result()


def _kw_calc(threshold):
    e.load_network()
    e.set_loads({t: 300.0 for t in e.TX})
    e.solve()
    return threshold * 300.0 / e.metrics()["max_loading"]


SCENARIOS = ["Normal", "Heatwave (feeder A)", "EV evening surge", "Severe stress"]
DEFAULTS = {"scenario": "Normal", "heat": 1.6, "ev_kw": 200, "ev_tx": "T4", "stress": 2.5, "thr": 85, "moves": []}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("log", [])
st.session_state.setdefault("last", None)


# ------------------------------------------------------------------ data
@st.cache_data
def load_forecast():
    fc = pd.read_csv(ROOT / "data" / "forecast.csv", parse_dates=["timestamp"])
    return fc, fc.pivot(index="timestamp", columns="transformer_id", values="predicted_kw")


@st.cache_data
def forecast_nrmse():
    try:
        m = pd.read_csv(ROOT / "docs" / "metrics.csv")
        return float(m[(m["transformer"] == "ALL") & (m["model"] == "XGBoost (train+val refit)")]["nRMSE_%"].iloc[0])
    except Exception:
        return None


@st.cache_data
def kw_at_limit(threshold):
    """kW per transformer that corresponds to `threshold` % loading (from one power-flow run)."""
    return dss(_kw_calc, threshold)


fc, pred = load_forecast()


def params():
    s = st.session_state
    return (s["scenario"], s["heat"], s["ev_kw"], s["ev_tx"], s["stress"])


def adjust(tx, ts, kw, p):
    scenario, heat, ev_kw, ev_tx, stress = p
    if scenario.startswith("Heatwave") and tx in ("T1", "T2", "T3"):
        return kw * heat
    if scenario.startswith("EV") and tx == ev_tx and 18 <= ts.hour < 22:
        return kw + ev_kw
    if scenario.startswith("Severe"):
        return kw * stress
    return kw


def loads_at(ts, p):
    ts = pd.Timestamp(ts)
    return {tx: float(adjust(tx, ts, pred.loc[ts, tx], p)) for tx in e.TX}


@st.cache_data
def eval_state(loads_items, moves):
    return dss(e.evaluate_state, dict(loads_items), list(moves))


def evaluate(loads, moves):
    return eval_state(tuple((k, round(v, 3)) for k, v in loads.items()), tuple(moves))


@st.cache_data
def peak_time(p):
    best, bt = -1.0, None
    for ts, row in pred.iterrows():
        m = max(adjust(tx, ts, row[tx], p) for tx in e.TX)
        if m > best:
            best, bt = m, ts
    return bt.to_pydatetime()


@st.cache_data
def day_scan(day, p, moves, thr):
    """Hourly loading for one calendar day under the given switch state -> overload alert table."""
    day = pd.Timestamp(day)
    hours = pred.index[(pred.index >= day) & (pred.index < day + pd.Timedelta(days=1))]
    rec = []
    for ts in hours:
        m = evaluate(loads_at(ts, p), moves)
        if m is None:
            continue
        for tx, pct in m["loading_by_tx"].items():
            rec.append({"timestamp": ts, "tx": tx, "pct": pct})
    d = pd.DataFrame(rec)
    if d.empty:
        return pd.DataFrame()
    over = d[d["pct"] > thr]
    if over.empty:
        return pd.DataFrame()
    g = over.groupby("tx").agg(first_overload=("timestamp", "min"), hours_over=("timestamp", "size"),
                               peak_pct=("pct", "max")).reset_index()
    g["severity"] = g["peak_pct"].apply(lambda x: "CRITICAL" if x > e.HARD_LIMIT else "WARNING")
    g["first_overload"] = g["first_overload"].dt.strftime("%d %b %H:%M")
    g["peak_pct"] = g["peak_pct"].round(1)
    return g.sort_values("peak_pct", ascending=False)


# ------------------------------------------------------------------ callbacks
tmin, tmax = pred.index.min().to_pydatetime(), pred.index.max().to_pydatetime()
st.session_state.setdefault("t", peak_time(params()))


def shift(hours):
    st.session_state["t"] = min(max(st.session_state["t"] + timedelta(hours=hours), tmin), tmax)


def jump_peak():
    st.session_state["t"] = peak_time(params())


def reset_switches():
    st.session_state["moves"] = []
    st.session_state["last"] = None
    st.session_state["log"].append("Switches reset to normal configuration.")


def do_reroute():
    s = st.session_state
    t, p, thr = pd.Timestamp(s["t"]), params(), float(s["thr"])
    res = dss(e.evaluate_and_reroute, loads_at(t, p), thr)
    s["last"] = {"t": t, "res": res, "scenario": s["scenario"]}
    stamp = f"{t:%d %b %H:%M} [{s['scenario']}]"
    if res["status"] == "rerouted":
        s["moves"] = re.findall(r"move (T\d)", " ".join(res["action"]))
        s["log"].append(f"{stamp} REROUTED: {'; '.join(res['action'])} | max loading "
                        f"{res['before']['max_loading']}% -> {res['after']['max_loading']}% | {res['compute_s']} s")
    elif res["status"] == "no_action":
        s["moves"] = []
        s["log"].append(f"{stamp} NO ACTION: all transformers below {thr:.0f}%.")
    else:
        s["log"].append(f"{stamp} CRITICAL: no safe reconfiguration. "
                        f"Best partial: {', '.join(res.get('best_partial', [])) or 'none'}.")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Replay and scenario")
    st.slider("Replay time", min_value=tmin, max_value=tmax, step=timedelta(hours=1),
              key="t", format="DD MMM YYYY HH:mm")
    c1, c2, c3 = st.columns(3)
    c1.button("-1 h", on_click=shift, args=(-1,))
    c2.button("+1 h", on_click=shift, args=(1,))
    c3.button("+1 day", on_click=shift, args=(24,))
    st.button("Jump to peak hour", on_click=jump_peak)
    st.selectbox("Scenario", SCENARIOS, key="scenario")
    with st.expander("Scenario parameters", expanded=True):
        st.slider("Heatwave multiplier (T1 to T3)", 1.0, 2.5, step=0.05, key="heat")
        st.slider("EV surge (kW, 18:00 to 22:00)", 0, 400, step=10, key="ev_kw")
        st.selectbox("EV surge transformer", e.TX, key="ev_tx")
        st.slider("Severe stress multiplier (all)", 1.0, 3.5, step=0.1, key="stress")
    st.slider("Overload threshold (%)", 70, 100, key="thr")
    st.caption("Forecast = held-out test period. Scenarios scale the forecast to create overloads.")

# ------------------------------------------------------------------ main
t = pd.Timestamp(st.session_state["t"])
p, thr, moves = params(), float(st.session_state["thr"]), list(st.session_state["moves"])
loads = loads_at(t, p)
cur = evaluate(loads, moves)

st.title("AI-driven load forecasting and dynamic feeder rerouting")
st.caption(f"Replay time: **{t:%a %d %b %Y, %H:%M}** | Scenario: **{st.session_state['scenario']}** | "
           f"Active LV moves: **{', '.join(moves) if moves else 'none (normal configuration)'}**")
if cur is None:
    st.error("Power flow did not converge at this load level. Reduce the scenario multiplier.")
    st.stop()

loading = cur["loading_by_tx"]
n_over = sum(v > thr for v in loading.values())
nrmse = forecast_nrmse()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Transformers overloaded", f"{n_over} of 6")
k2.metric("Peak loading", f"{cur['max_loading']:.1f} %")
k3.metric("Minimum voltage", f"{cur['min_voltage_pu']:.4f} pu")
k4.metric("Forecast nRMSE (test)", f"{nrmse:.1f} %" if nrmse is not None else "n/a")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Network")
    st.plotly_chart(plots.topology_figure(loading, moves, thr))
    st.caption("Green below 70%, amber 70% to threshold, red above threshold. Dotted grey: open tie switches. "
               "Solid orange: LV tie closed, block b moved to a neighbour.")
with right:
    st.subheader("Load forecast")
    sel = st.selectbox("Transformer", e.TX, key="sel_tx")
    d = fc[fc["transformer_id"] == sel].reset_index(drop=True)
    adj = None
    if st.session_state["scenario"] != "Normal":
        adj = pd.Series([adjust(sel, ts, kw, p) for ts, kw in zip(d["timestamp"], d["predicted_kw"])],
                        index=d["timestamp"])
    st.plotly_chart(plots.load_chart(d, t, kw_at_limit(thr), adj))

st.divider()
a, b = st.columns(2)
with a:
    st.subheader("Day-ahead overload alerts")
    st.caption(f"All 24 hours of {t:%d %b %Y}, under the current switch state.")
    alerts = day_scan(t.normalize(), p, tuple(moves), thr)
    if alerts.empty:
        st.success("No transformer is forecast above the threshold on this day.")
    else:
        alerts = alerts.rename(columns={"tx": "transformer", "first_overload": "first overload",
                                        "hours_over": "hours over", "peak_pct": "peak %"})
        st.dataframe(alerts[["severity", "transformer", "first overload", "hours over", "peak %"]],
                     hide_index=True)
    st.subheader("Action log")
    if st.session_state["log"]:
        for line in reversed(st.session_state["log"][-8:]):
            st.write("- " + line)
    else:
        st.caption("No actions yet.")
with b:
    st.subheader("Operator controls")
    r1, r2 = st.columns(2)
    r1.button("Reroute", type="primary", on_click=do_reroute)
    r2.button("Reset switches", on_click=reset_switches)
    st.multiselect("Manual LV moves (block b to neighbour)", e.TX, key="moves",
                   format_func=lambda x: f"{x} -> {e.MOVES[x][2]}")
    last = st.session_state["last"]
    if last:
        res = last["res"]
        st.markdown(f"**Last reroute** ({last['t']:%d %b %H:%M}, {last['scenario']}): `{res['status']}`"
                    + (f", {res['compute_s']} s" if "compute_s" in res else ""))
        if res.get("before") and res.get("after"):
            st.plotly_chart(plots.before_after_chart(res["before"]["loading_by_tx"],
                                                     res["after"]["loading_by_tx"], thr))
        with st.expander("Algorithm log"):
            for line in res["log"]:
                st.write("- " + line)