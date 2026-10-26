"""Closed-loop orchestrator (gap item G10): one call of step(t) runs every module once, in order.

Step t is the boundary at the end of hour t:

    1 measure      the meters' loads for hour t (scenario applied), which ran in the configuration in force
    2 power flow   OpenDSS on the measured loads in that configuration, and in the normal configuration
                   (what would have happened without control)
    3 forecast     the 24 h-ahead forecast for the next hour t + 1 (issued at t - 23 h) and its upper bound;
                   the live source also issues the forecast for t + 24 h; forecasts whose hour has passed
                   are scored
    4 detect       risk classes for hour t (its forecast against its measurement, as on the dashboard) and
                   voltage/line alerts from the measured state; stored as alerts
    5 validate     if the next hour's forecast breaks a limit, or the network is not in its normal
                   configuration, the controller (grid/decisions.py) evaluates the candidates
    6 apply or     monitor: nothing; recommend: a proposal awaits the operator (the service pauses the clock
      propose      so it cannot go stale); auto: a safe reconfiguration is applied before hour t + 1 begins
    7 record       one sim_steps row with what was measured, forecast and decided, and how long each stage took

Deciding at the boundary means a switching change, automatic or approved, is in force for the whole of the
hour it was computed for, and the next step measures its effect. Rerouting decisions use the point forecast;
the upper bound only raises the forecast risk class (decision recorded in the README, Phase 4/5).
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "grid", ROOT / "sim"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from core import config, ids, scenarios  # noqa: E402
from core.logging_setup import get_logger  # noqa: E402

log = get_logger("orchestrator")
STAGES = ("measure", "power_flow", "forecast", "detect", "decide", "record")


def time_label(t):
    return f"{pd.Timestamp(t):%Y-%m-%d %H:%M}"


def prepare_database(store):
    """Import the committed forecasts, measured loads, intervals and meter registry, and register the model
    (all idempotent: the same as the dashboard does at start-up)."""
    store.import_forecast_csv(ROOT / "data" / "forecast.csv")
    store.import_intervals_csv(ROOT / "data" / "forecast_intervals.csv")
    store.import_meter_registry(ROOT / "data" / "processed" / "meter_registry.csv")
    try:
        sys.path.insert(0, str(ROOT / "ml"))
        import metadata
        metadata.register(store, ROOT / "ml" / "model_xgb.json")
    except (OSError, ValueError) as ex:
        log.warning("model not registered: %s", ex)


def benchmark_scenario(name):
    for s in scenarios.benchmark():
        if s.name == name:
            return s
    raise ValueError(f"unknown scenario {name!r}; choose one of {[s.name for s in scenarios.benchmark()]}")


class Orchestrator:
    def __init__(self, store, engine=None, demand=None, forecasts=None, controller=None, scenario=None,
                 service=None, forecast_source=None, threshold=None, use_mv=None):
        import engine as engine_mod                       # OpenDSS loads here, on the caller's thread
        from decisions import Controller
        from risk import RiskModel
        from sources import ClassicSource, forecast_source as make_forecasts
        sim = config.simulation()["orchestrator"]
        self.store = store
        self.engine = engine or engine_mod._default
        frame = None
        if demand is None or forecasts is None:
            frame = store.forecast_frame()
        self.demand = demand or ClassicSource(store, frame)
        self.forecasts = forecasts or make_forecasts(forecast_source or sim["forecast_source"], store, frame)
        self.service = service or sim["controller"]
        self.ctl = controller or Controller(self.engine, store=store, name=self.service)
        self.risk = RiskModel()
        self.threshold = float(self.engine.threshold if threshold is None else threshold)
        self.use_mv = use_mv
        self.scenario = benchmark_scenario(scenario or sim["scenario"])
        self.step_delta = self.demand.resolution

    # ------------------------------------------------------------------ settings
    def set_scenario(self, name):
        self.scenario = benchmark_scenario(name)
        return self.scenario

    def settings(self):
        """Operator settings that change decisions (persisted by the service)."""
        return {"threshold": self.threshold, "use_mv": self.use_mv}

    def apply_settings(self, s):
        if s.get("threshold") is not None:
            thr = float(s["threshold"])
            if not 0 < thr <= self.engine.hard_limit:
                raise ValueError(f"threshold must be above 0 and at most the {self.engine.hard_limit:g}% hard limit")
            self.threshold = thr
        if "use_mv" in s:
            self.use_mv = None if s["use_mv"] is None else bool(s["use_mv"])
        return self.settings()

    def context(self):
        """Everything besides the time that a recommendation depends on (for the stale-approval check)."""
        return f"service {self.scenario.name} threshold={self.threshold:g} mv={self.use_mv}"

    def _apply_scenario(self, loads, t):
        return None if loads is None else self.scenario.loads(loads, pd.Timestamp(t))

    def _state(self, loads, keys):
        """Power flow for loads in a configuration -> engine metrics (None if not solvable)."""
        if loads is None:
            return None
        return self.engine.evaluate_configuration(loads, keys)["metrics"]

    # ------------------------------------------------------------------ the step
    def step(self, t, run_id=0):
        """Run one closed-loop step at simulated time t; returns the recorded sim_steps row."""
        t = pd.Timestamp(t)
        nxt = t + self.step_delta
        timings, wall0 = {}, time.perf_counter()
        clock = [time.perf_counter()]

        def lap(stage):
            now = time.perf_counter()
            timings[stage] = round(1000 * (now - clock[0]), 3)
            clock[0] = now

        ctl, thr = self.ctl, self.threshold
        pend = ctl.pending                                     # a proposal whose hour has begun undecided expires
        if pend is not None and pend.status == "pending" and pend.for_time <= time_label(t):
            ctl.cancel_pending(f"expired: hour {pend.for_time} began without a decision")
        config_before = list(ctl.active)

        # 1 measure
        measured = self._apply_scenario(self.demand.measured(t), t)
        lap("measure")

        # 2 power flow: hour t as it ran, and without control
        meas = self._state(measured, config_before)
        meas_normal = meas if not config_before else self._state(measured, [])
        fc_now = self._apply_scenario(self.forecasts.point(t), t)
        up_now = self._apply_scenario(self.forecasts.upper(t), t)
        fc_now_state = self._state(fc_now, config_before)
        up_now_state = self._state(up_now, config_before)
        lap("power_flow")

        # 3 forecast for the next hour (issued 23 h ago), issue t + 24 h, score what has passed
        fc_next = self._apply_scenario(self.forecasts.point(nxt), nxt)
        up_next = self._apply_scenario(self.forecasts.upper(nxt), nxt)
        self.forecasts.issue(t)
        self.store.score_forecasts(t, model_version=self.forecasts.model_version)
        fc_next_state = self._state(fc_next, config_before)
        up_next_state = self._state(up_next, config_before)
        lap("forecast")

        # 4 detect: hour t, forecast against measurement, and network alerts from the measured state
        risk, n_alerts = {}, 0
        if fc_now_state is not None:
            risk = self.risk.transformers(fc_now_state["loading_by_tx"], meas["loading_by_tx"] if meas else None, thr,
                                          up_now_state["loading_by_tx"] if up_now_state else None)
            net = self.risk.network_alerts(meas or fc_now_state)
            rows = ([{"asset": ids.transformer_id(tx), "kind": "transformer", "level": r["risk_class"],
                      "value": max(v for v in (r["forecast_pct"], r["measured_pct"]) if v is not None),
                      "text": f"{tx}: forecast {r['forecast_pct']}%, measured {r['measured_pct']}% ({r['risk_class']})"}
                     for tx, r in risk.items() if r["risk_class"] != "normal"]
                    + [{"asset": x["element"], "kind": x["type"], "level": x["level"], "value": x["value"],
                        "text": x["text"]} for x in net])
            self.store.record_alerts(time_label(t), rows,
                                     context=f"{self.context()} config={','.join(config_before) or 'normal'}")
            n_alerts = len(rows)
        lap("detect")

        # 5-6 validate candidates for the next hour, then apply (auto) or propose (recommend)
        rec, decision = None, None
        if fc_next is None:
            decision = "end"                                   # no forecast for the next hour: nothing to decide
        elif ctl.estopped:
            decision = "estop"
        elif ctl.mode == "monitor":
            decision = "monitor"
        else:
            needs = (config_before or fc_next_state is None or not self.engine.is_feasible(fc_next_state)
                     or any(v > thr for v in fc_next_state["loading_by_tx"].values()))
            if needs:
                rec = ctl.recommend(fc_next, time_label(nxt), self.context(), thr, self.use_mv)
                decision = rec.kind if rec else "monitor"
            else:
                decision = "no_action"
        config_after = list(ctl.active)
        if config_after != config_before:                      # the next hour as it will now run
            fc_next_after, up_next_after = self._state(fc_next, config_after), self._state(up_next, config_after)
        else:
            fc_next_after, up_next_after = fc_next_state, up_next_state
        lap("decide")

        # 7 record
        over = lambda m: None if m is None else int(sum(v > thr for v in m["loading_by_tx"].values()))  # noqa: E731
        by_tx = lambda m: None if m is None else {k: round(v, 2) for k, v in m["loading_by_tx"].items()}  # noqa: E731
        row = {"service": self.service, "run_id": int(run_id), "sim_time": t,
               "wall_time": pd.Timestamp.now().isoformat(timespec="seconds"), "scenario": self.scenario.name,
               "mode": ctl.mode, "config_before": config_before, "config_after": config_after,
               "decision": decision, "rec_id": rec.rec_id if rec else None,
               "forecast_max_pct": None if fc_next_state is None else round(fc_next_state["max_loading"], 2),
               "forecast_over": over(fc_next_state),
               "upper_max_pct": None if up_next_state is None else round(up_next_state["max_loading"], 2),
               "measured_max_pct": None if meas is None else round(meas["max_loading"], 2),
               "measured_over": over(meas),
               "measured_over_normal": over(meas_normal),
               "measured_max_normal_pct": None if meas_normal is None else round(meas_normal["max_loading"], 2),
               "min_voltage_pu": None if meas is None else meas["min_voltage_pu"],
               "losses_kw": None if meas is None else meas["losses_kw"],
               "risk": {tx: r["risk_class"] for tx, r in risk.items()}, "alerts": n_alerts,
               "min_voltage_bus": None if meas is None else meas.get("min_voltage_bus"),
               "loads_kw": None if measured is None else {k: round(v, 2) for k, v in measured.items()},
               "measured_pct": by_tx(meas), "forecast_pct": by_tx(fc_now_state), "upper_pct": by_tx(up_now_state),
               "next_forecast_pct": by_tx(fc_next_after), "next_upper_pct": by_tx(up_next_after),
               "tx_detail": None if meas is None else meas.get("transformers"),
               "feeders": None if meas is None else meas.get("feeders"),
               "switches": self.engine.switch_state(config_before),
               "switches_after": self.engine.switch_state(config_after)}
        lap("record")                                          # time to build the row; the write is timed below
        t_write = time.perf_counter()
        row["timings_ms"] = timings
        row["duration_ms"] = round(1000 * (t_write - wall0), 3)
        self.store.record_step(row)
        timings["record"] = round(timings["record"] + 1000 * (time.perf_counter() - t_write), 3)
        row["duration_ms"] = round(1000 * (time.perf_counter() - wall0), 3)
        return {**row, "rec": rec}

    @property
    def awaiting_operator(self):
        """A switching proposal is pending in recommend mode."""
        p = self.ctl.pending
        return self.ctl.mode == "recommend" and p is not None and p.status == "pending"
