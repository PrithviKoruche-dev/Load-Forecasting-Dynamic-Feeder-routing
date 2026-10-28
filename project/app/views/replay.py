"""What-if replay (sandbox page of the dashboard): forecast -> risk -> OpenDSS -> recommendation -> decision,
for any hour picked with the slider.

This page is a sandbox, separate from the live network that the simulation service runs: it has its own
controller ("default" in the database), its own switch configuration and its own time slider, so studying
an hour here never changes the live simulation.

Scenario presets and slider ranges come from config/scenarios.yaml, limits, risk bands and the default
operating mode from config/operations.yaml. Decisions go through grid/decisions.py (operating modes,
approval, rollback, emergency stop).

Storage (Phase 3): forecasts and measured loads are read from the database (config/storage.yaml, or
LFDR_DB_URL), and one controller per dashboard process is restored from it, so the configuration, a pending
recommendation and the whole audit log survive a restart. Every browser session shares that controller.
Operator actions are also written to logs/dashboard.log.
"""
import faulthandler
import importlib
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "grid", ROOT / "ml", ROOT / "app"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import plots            # noqa: E402
from core import config, scenarios  # noqa: E402
from core.logging_setup import get_logger, setup as setup_logging  # noqa: E402
from core.paths import LOGS_DIR  # noqa: E402

setup_logging(log_file=LOGS_DIR / "dashboard.log")
log = get_logger("dashboard")

@st.cache_resource
def dss_worker():
    """OpenDSS is not thread-safe and Streamlit starts a new thread on every rerun.
    So the engine is imported and used ONLY inside one permanent worker thread.

    The crash reporter (faulthandler) is switched on only AFTER OpenDSS has loaded: on Windows the
    OpenDSS library raises and handles an internal exception (code 0xE0465043, "FPC") while loading,
    and faulthandler reports every native exception, even handled ones. Enabling it earlier filled
    the crash log with that harmless event on every launch."""
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dss")
    engine = ex.submit(importlib.import_module, "engine").result()
    LOGS_DIR.mkdir(exist_ok=True)
    crash_log = open(LOGS_DIR / "dss_crash.log", "a")      # written only if the process dies natively
    faulthandler.enable(file=crash_log, all_threads=True)
    log.info("dashboard started; OpenDSS engine loaded")
    return ex, engine, crash_log


EXEC, e, _CRASH_LOG = dss_worker()


def dss(fn, *args):
    """Run fn(*args) on the dedicated OpenDSS thread and wait for the result."""
    return EXEC.submit(fn, *args).result()




_dec = sys.modules.get("decisions")                     # this folder has a page called decisions.py too:
if _dec is None or Path(getattr(_dec, "__file__", "")).parent != ROOT / "grid":     # make sure it is grid's
    import importlib.util
    _spec = importlib.util.spec_from_file_location("decisions", ROOT / "grid" / "decisions.py")
    _dec = importlib.util.module_from_spec(_spec)
    sys.modules["decisions"] = _dec
    _spec.loader.exec_module(_dec)
MODES, SEVERITY, Controller, DecisionError = _dec.MODES, _dec.SEVERITY, _dec.Controller, _dec.DecisionError
from risk import LABELS as RISK_LABELS, RiskModel  # noqa: E402
from storage import Store  # noqa: E402
from core import ids  # noqa: E402


@st.cache_resource
def services(db_url):
    """The database and the one controller of this dashboard process, restored from the database.
    Built on the OpenDSS thread because restoring may need to load the network."""
    store = Store(db_url)
    store.import_forecast_csv(ROOT / "data" / "forecast.csv")
    store.import_intervals_csv(ROOT / "data" / "forecast_intervals.csv")
    store.import_meter_registry(ROOT / "data" / "processed" / "meter_registry.csv")
    try:
        import metadata
        metadata.register(store, ROOT / "ml" / "model_xgb.json")
    except (OSError, ValueError) as ex:                    # a missing or mismatched metadata file is reported
        log.warning("model not registered: %s", ex)
    ctl = EXEC.submit(lambda: Controller(e._default, store=store)).result()
    log.info("database %s ready: %s", db_url, {k: v for k, v in store.counts().items() if v})
    return store, ctl


DB_URL = config.database_url()
STORE, CTL = services(DB_URL)


def _kw_calc(threshold):
    e.load_network()
    e.set_loads({t: 300.0 for t in e.TX})
    e.solve()
    return threshold * 300.0 / e.metrics()["max_loading"]


PRESETS = {p.label: p for p in scenarios.dashboard_presets()}
SCENARIOS = list(PRESETS)
_OPS = config.operations()
RISK = RiskModel()
PARAM_KEYS = []                                  # session keys of every scenario control, in sidebar order
DEFAULTS = {"scenario": SCENARIOS[0]}
for _p in PRESETS.values():
    if _p.slider:
        PARAM_KEYS.append(_p.slider["key"])
        DEFAULTS[_p.slider["key"]] = _p.slider["default"]
    if _p.choose_transformer:
        PARAM_KEYS.append(_p.choose_transformer["key"])
        DEFAULTS[_p.choose_transformer["key"]] = _p.choose_transformer["default"]
DEFAULTS.update({"thr": int(_OPS["limits"]["overload_threshold_pct"]),
                 "use_mv": bool(_OPS["rerouting"]["use_mv_transfers"])})
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state["ctl"] = CTL
st.session_state.setdefault("flash", None)
st.session_state.setdefault("auto_key", None)
# widgets that mirror controller state are refreshed from it on every run
st.session_state["mode_widget"] = CTL.mode
st.session_state["moves"] = list(CTL.active)


# ------------------------------------------------------------------ data
@st.cache_data
def load_forecast(db_url):
    """Forecasts and measured loads from the database (same shape as data/forecast.csv)."""
    fc = STORE.forecast_frame()
    upper_col = next((c for c in reversed(list(fc.columns)) if c[0] == "p" and c.endswith("_kw") and c[1:-3].isdigit()), None)
    upper = fc.pivot(index="timestamp", columns="transformer_id", values=upper_col) if upper_col else None
    return (fc, fc.pivot(index="timestamp", columns="transformer_id", values="predicted_kw"),
            fc.pivot(index="timestamp", columns="transformer_id", values="actual_kw"), upper, upper_col)


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


fc, pred, actual, upper, UPPER_COL = load_forecast(DB_URL)
USE_UPPER = upper is not None and config.forecasting()["intervals"]["use_upper_for_risk"]


def params():
    """Hashable snapshot of the scenario controls: (preset label, {control key: value})."""
    s = st.session_state
    return (s["scenario"], tuple((k, s[k]) for k in PARAM_KEYS))


@st.cache_data
def scenario_for(p):
    """The core.scenarios.Scenario selected by the controls (same rules as the benchmark)."""
    label, values = p
    values, preset = dict(values), PRESETS[label]
    value = values.get(preset.slider["key"]) if preset.slider else None
    tx = values.get(preset.choose_transformer["key"]) if preset.choose_transformer else None
    return preset.build(value, tx)


def adjust(tx, ts, kw, p):
    return scenario_for(p).adjust(tx, ts, kw)


def loads_at(ts, p, frame=None):
    ts = pd.Timestamp(ts)
    frame = pred if frame is None else frame
    return {tx: float(adjust(tx, ts, frame.loc[ts, tx], p)) for tx in e.TX}


def context():
    """Everything besides the time that the loads and limits depend on (for stale-approval checks)."""
    s = st.session_state
    return f"{s['scenario']} {dict(params()[1])} threshold={s['thr']} mv={s['use_mv']}"


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


def _now_label():
    return f"{pd.Timestamp(st.session_state['t']):%Y-%m-%d %H:%M}"


def _guard(fn, *args, ok=None):
    """Run a controller operation on the OpenDSS thread; show a refusal (DecisionError) as a message.
    Every NEW event it produced goes to logs/dashboard.log (only new ones: an assessment that creates no
    event must not log the previous event again)."""
    n = len(CTL.history)
    try:
        out = dss(fn, *args)
        if ok:
            st.session_state["flash"] = ("success", ok)
        for ev in CTL.history[n:]:
            log.info(ev["message"])
        return out
    except DecisionError as ex:
        st.session_state["flash"] = ("error", str(ex))
        log.warning("refused: %s", ex)


def shift(hours):
    st.session_state["t"] = min(max(st.session_state["t"] + timedelta(hours=hours), tmin), tmax)


def jump_peak():
    st.session_state["t"] = peak_time(params())


def on_mode():
    _guard(CTL.set_mode, st.session_state["mode_widget"], _now_label())


def on_manual():
    _guard(CTL.set_configuration, list(st.session_state["moves"]), _now_label(), "manual switching by operator")


def do_recommend():
    s = st.session_state
    t = pd.Timestamp(s["t"])
    rec = _guard(CTL.recommend, loads_at(t, params()), _now_label(), context(), float(s["thr"]), s["use_mv"])
    if rec is not None and rec.kind in ("no_action", "no_change"):
        s["flash"] = ("success", rec.summary)


def do_approve(rec_id):
    _guard(CTL.approve, rec_id, _now_label(), context(), ok=f"{rec_id} applied.")


def do_reject(rec_id):
    _guard(CTL.reject, rec_id, "rejected from the dashboard")


def do_rollback():
    _guard(CTL.rollback, _now_label(), ok="Rolled back to the previous configuration.")


def do_reset():
    _guard(CTL.reset_to_normal, _now_label())


def do_estop():
    _guard(CTL.emergency_stop, _now_label())
    st.session_state["flash"] = ("error", "EMERGENCY STOP: automation halted. Switches are left as they are.")


def do_clear_estop():
    _guard(CTL.clear_emergency_stop, _now_label(), ok="Emergency stop cleared. Choose an operating mode to resume.")


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
        for _p in PRESETS.values():
            if _p.slider:
                sl = _p.slider
                st.slider(sl["label"], sl["min"], sl["max"], step=sl["step"], key=sl["key"])
            if _p.choose_transformer:
                st.selectbox(_p.choose_transformer["label"], e.TX, key=_p.choose_transformer["key"])
    _thr = _OPS["dashboard"]["threshold_slider"]
    st.slider("Overload threshold (%)", _thr["min"], _thr["max"], key="thr")
    st.caption("Forecast = held-out test period. Scenarios scale the forecast to create overloads.")
    st.header("Operation")
    st.radio("Operating mode", list(MODES), format_func=MODES.get, key="mode_widget", on_change=on_mode,
             disabled=CTL.estopped)
    st.checkbox("Allow MV feeder transfers", key="use_mv",
                help="Also consider opening a sectionaliser and closing the tie between feeders A and B.")
    if CTL.estopped:
        st.button("Clear emergency stop", on_click=do_clear_estop)
    else:
        st.button("EMERGENCY STOP", type="primary", on_click=do_estop,
                  help="Halts automation at once: mode drops to monitor-only, pending proposals are cancelled. "
                       "Switches are left as they are.")

# ------------------------------------------------------------------ automatic mode
t = pd.Timestamp(st.session_state["t"])
p, thr = params(), float(st.session_state["thr"])
loads = loads_at(t, p)
if CTL.mode == "auto" and not CTL.estopped:
    key = (_now_label(), context())
    if st.session_state["auto_key"] != key:                 # assess once per time step / setting change
        st.session_state["auto_key"] = key
        _guard(CTL.recommend, loads, _now_label(), context(), thr, st.session_state["use_mv"])
moves = list(CTL.active)
cur = evaluate(loads, moves)
measured_loads = loads_at(t, p, actual) if not actual.loc[t].isna().any() else None
measured = evaluate(measured_loads, moves) if measured_loads else None
upper_eval = evaluate(loads_at(t, p, upper), moves) if USE_UPPER and not upper.loc[t].isna().any() else None
STORE.score_forecasts(t)                                 # every forecast whose hour has passed gets scored

# ------------------------------------------------------------------ main
st.title("What-if replay (sandbox)")
st.caption("Study any hour of the test period. This page has its own controller and switch configuration: "
           "nothing done here changes the live simulation.")
st.caption(f"Replay time: **{t:%a %d %b %Y, %H:%M}** | Scenario: **{st.session_state['scenario']}** | "
           f"Mode: **{MODES[CTL.mode]}**{' | **EMERGENCY STOP**' if CTL.estopped else ''} | "
           f"Configuration: **{', '.join(moves) if moves else 'normal'}**")
flash = st.session_state["flash"]
if flash:
    getattr(st, flash[0])(flash[1])
    st.session_state["flash"] = None
if cur is None:
    st.error("Power flow did not converge at this load level (or the configuration is not radial). "
             "Reduce the scenario multiplier or reset the switches.")
    st.button("Reset to normal configuration", on_click=do_reset)
    st.stop()

loading = cur["loading_by_tx"]
risk = RISK.transformers(loading, measured["loading_by_tx"] if measured else None, thr,
                         upper_eval["loading_by_tx"] if upper_eval else None)
STORE.record_alerts(_now_label(),
                    [{"asset": ids.transformer_id(t_), "kind": "transformer", "level": r["risk_class"],
                      "value": max(v for v in (r["forecast_pct"], r["measured_pct"]) if v is not None),
                      "text": f"{t_}: forecast {r['forecast_pct']}%, measured {r['measured_pct']}% ({r['risk_class']})"}
                     for t_, r in risk.items() if r["risk_class"] != "normal"]
                    + [{"asset": x["element"], "kind": x["type"], "level": x["level"], "value": x["value"], "text": x["text"]}
                       for x in RISK.network_alerts(cur)],
                    context=f"{context()} config={','.join(moves) or 'normal'}")
n_over = sum(v > thr for v in loading.values())
nrmse = forecast_nrmse()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Transformers overloaded", f"{n_over} of 6")
k2.metric("Peak loading", f"{cur['max_loading']:.1f} %")
k3.metric("Minimum voltage", f"{cur['min_voltage_pu']:.4f} pu", help=f"lowest bus: {cur.get('min_voltage_bus', '')}")
k4.metric("Losses", f"{cur['losses_kw']:.1f} kW")
k5.metric("Forecast nRMSE (test)", f"{nrmse:.1f} %" if nrmse is not None else "n/a")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Network")
    st.plotly_chart(plots.topology_figure(loading, moves, thr, {t_: r["risk_class"] for t_, r in risk.items()},
                                          e._default.switch_state(moves)))
    st.caption("Transformer colour = risk class: green normal, amber elevated, orange approaching, red predicted "
               "overload, dark red actual (measured) overload. Dotted grey: open switches. Blue: closed tie, "
               "block b or feeder section moved.")
with right:
    st.subheader("Load forecast")
    sel = st.selectbox("Transformer", e.TX, key="sel_tx")
    d = fc[fc["transformer_id"] == sel].reset_index(drop=True)
    adj = band = None
    stressed = PRESETS[st.session_state["scenario"]].rule is not None

    def _series(col):
        vals = d[col] if not stressed else [adjust(sel, ts, kw, p) for ts, kw in zip(d["timestamp"], d[col])]
        return pd.Series(list(vals), index=d["timestamp"])
    if stressed:
        adj = _series("predicted_kw")
    qcols = sorted((c for c in d.columns if c[0] == "p" and c.endswith("_kw") and c[1:-3].isdigit()),
                   key=lambda c: int(c[1:-3]))
    if len(qcols) >= 2 and d[qcols[-1]].notna().any():
        band = (f"{qcols[0][:-3].upper()}-{qcols[-1][:-3].upper()} band" + (" + scenario" if stressed else ""),
                _series(qcols[0]), _series(qcols[-1]))
    st.plotly_chart(plots.load_chart(d, t, kw_at_limit(thr), adj, band=band))

st.divider()
a, b = st.columns(2)
with a:
    st.subheader("Risk now")
    up_label = UPPER_COL[:-3].upper() if USE_UPPER else None
    st.caption("Forecast = the 24 h-ahead forecast for this hour; measured = what the meters recorded. "
               + (f"Forecast risk uses the {up_label} upper bound, so forecast uncertainty can raise it. " if up_label else "")
               + "All under the current scenario and switch state.")
    st.dataframe(pd.DataFrame([{"transformer": t_, "forecast %": r["forecast_pct"],
                                **({f"{up_label} %": r["upper_pct"]} if up_label else {}),
                                "measured %": r["measured_pct"],
                                "forecast risk": RISK_LABELS.get(r["forecast_class"], "n/a"),
                                "measured risk": RISK_LABELS.get(r["measured_class"], "n/a"),
                                "risk class": RISK_LABELS[r["risk_class"]]} for t_, r in risk.items()]),
                 hide_index=True)
    net_alerts = RISK.network_alerts(cur)
    if net_alerts:
        st.dataframe(pd.DataFrame([{"level": x["level"], "type": x["type"], "where": x["element"],
                                    "detail": x["text"]} for x in net_alerts]), hide_index=True)
    else:
        st.caption("No voltage or line alerts.")
    perf = STORE.forecast_performance(until=t)
    total = STORE.forecast_performance()
    with st.expander("Forecast accuracy so far", expanded=False):
        n_sc = int(perf.set_index("transformer").loc["ALL", "scored"]) if not perf.empty else 0
        n_pend = int(total.set_index("transformer").loc["ALL", "pending"]) if not total.empty else 0
        st.caption(f"{n_sc} forecasts scored up to {t:%d %b %H:%M}; {n_pend} still pending (their hour has not come "
                   "yet). nRMSE is relative to the mean actual load; bias > 0 means over-forecasting.")
        if n_sc:
            st.dataframe(perf.round(2), hide_index=True)
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

with b:
    st.subheader("Decision support")
    if CTL.estopped:
        st.error("Emergency stop is active: no switching is proposed or applied. "
                 "Roll back or reset manually if needed, then clear the stop.")
    elif CTL.mode == "monitor":
        st.info("Monitor-only mode: risks and alerts are shown; no switching is proposed.")
    elif CTL.mode == "recommend":
        st.button("Get recommendation", type="primary", on_click=do_recommend)
    else:
        st.info("Automatic mode: each time the replay time or settings change, the engine assesses the network "
                "and applies a safe reconfiguration (or a return to normal) by itself.")
    rec = CTL.last
    if rec is not None and rec.kind != "no_action":
        st.markdown(f"**{rec.rec_id}** for {rec.for_time} &nbsp;·&nbsp; `{rec.kind}` &nbsp;·&nbsp; status `{rec.status}`"
                    + (f" &nbsp;·&nbsp; {rec.compute_s} s" if rec.compute_s is not None else ""))
        {"no_safe_option": st.error, "no_change": st.success, "mitigate": st.warning}.get(rec.kind, st.info)(rec.summary)
        for line in rec.binding:
            st.write("- " + line)
        if rec.plan:
            st.dataframe(pd.DataFrame(rec.plan)[["step", "switch", "switch_id", "operation", "action"]],
                         hide_index=True)
        if rec.before and rec.after and rec.kind != "no_safe_option":
            st.plotly_chart(plots.before_after_chart(rec.before["loading_by_tx"], rec.after["loading_by_tx"], rec.threshold))
            st.dataframe(pd.DataFrame([
                {"": "peak loading %", "before": rec.before["max_loading"], "after": rec.after["max_loading"]},
                {"": "lowest voltage pu", "before": rec.before["min_voltage_pu"], "after": rec.after["min_voltage_pu"]},
                {"": "losses kW", "before": rec.losses_before_kw, "after": rec.losses_after_kw}]), hide_index=True)
        if rec.confidence:
            st.markdown(f"Confidence: **{rec.confidence['label']}** ({rec.confidence['probability']:.0%})",
                        help=rec.confidence["basis"])
        if rec.checks:
            with st.expander("Constraint checks"):
                st.dataframe(pd.DataFrame(rec.checks), hide_index=True)
        if rec.status == "pending" and CTL.mode == "recommend" and not CTL.estopped:
            c1, c2 = st.columns(2)
            c1.button("Approve", type="primary", on_click=do_approve, args=(rec.rec_id,))
            c2.button("Reject", on_click=do_reject, args=(rec.rec_id,))
        if rec.candidates:
            with st.expander(f"Candidates evaluated ({len(rec.candidates)})"):
                st.dataframe(pd.DataFrame([{"id": c["id"], "round": c["round"], "status": c.get("status", ""),
                                            "option": c["label"], "reason": c["reason"], "peak %": c["max_loading"],
                                            "losses kW": c["losses_kw"]} for c in rec.candidates]), hide_index=True)

    st.markdown("**Manual switching**")
    options = [k for k, a_ in e.ACTIONS.items() if a_.kind == "lv" or st.session_state["use_mv"] or k in moves]
    st.multiselect("Active switching actions", options, key="moves", on_change=on_manual,
                   format_func=lambda k: e.ACTIONS[k].label)
    r1, r2 = st.columns(2)
    r1.button("Roll back last change", on_click=do_rollback, disabled=not CTL.can_rollback)
    r2.button("Reset to normal configuration", on_click=do_reset, disabled=not moves)

st.subheader("Audit log")
st.caption(f"Stored in the database ({STORE.dialect}); survives restarts. Newest first.")
tab_ev, tab_rec, tab_sw, tab_al = st.tabs(["Events", "Recommendations", "Switching operations", "Alerts"])
with tab_ev:
    all_days = sorted({(ev["sim_time"] or "")[:10] for ev in STORE.events(controller=CTL.name, limit=5000)} - {""})
    f1, f2, f3, f4 = st.columns(4)
    day = f1.selectbox("Day (simulated)", ["All days"] + all_days, key="log_day")
    tx_filter = f2.selectbox("Transformer", ["All"] + e.TX, key="log_tx")
    type_filter = f3.multiselect("Event type", list(SEVERITY), key="log_types")
    sev_filter = f4.multiselect("Severity", ["info", "warning", "critical"], key="log_sev")
    rows = STORE.events(controller=CTL.name, sim_from=None if day == "All days" else day,
                        sim_to=None if day == "All days" else day, types=type_filter or None,
                        severities=sev_filter or None,
                        asset_id=None if tx_filter == "All" else ids.transformer_id(tx_filter), limit=500)
    if rows:
        st.dataframe(pd.DataFrame([{"#": r["seq"], "sim time": r["sim_time"] or "", "severity": r["severity"],
                                    "event": r["type"], "assets": ", ".join(r["assets"]), "details": r["message"]}
                                   for r in rows]), hide_index=True)
    else:
        st.caption("No events match these filters.")
with tab_rec:
    recs = STORE.recommendations(controller=CTL.name)
    if recs.empty:
        st.caption("No recommendations yet.")
    else:
        st.dataframe(recs.rename(columns={"rec_id": "id", "for_time": "for", "losses_before_kw": "losses before kW",
                                          "losses_after_kw": "losses after kW", "compute_s": "s"}), hide_index=True)
        pick = st.selectbox("Candidates of", recs["rec_id"].tolist(), key="log_rec")
        cands = STORE.candidates(pick)
        if not cands.empty:
            st.dataframe(cands[["cand_id", "round", "status", "label", "reason", "max_loading", "losses_kw"]],
                         hide_index=True)
with tab_sw:
    acts = STORE.actions(controller=CTL.name)
    if acts.empty:
        st.caption("No switch has been operated yet.")
    else:
        st.dataframe(acts[["sim_time", "rec_id", "how", "step", "switch", "switch_id", "operation", "action_key"]],
                     hide_index=True)
with tab_al:
    al = STORE.alerts(limit=500)
    if al.empty:
        st.caption("No alerts recorded yet.")
    else:
        st.dataframe(al[["sim_time", "asset", "kind", "level", "value", "text"]], hide_index=True)
