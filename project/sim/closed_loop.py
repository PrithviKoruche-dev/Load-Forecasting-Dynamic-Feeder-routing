"""Closed-loop results: every benchmark scenario run hour by hour through the orchestrator in automatic mode.

    python sim/closed_loop.py            # writes docs/closed_loop.csv (about 3 minutes)

Unlike grid/run_results.py, which re-solves each hour on its own from the normal configuration, here the
switch configuration carries over from hour to hour: a reroute stays in force until the controller restores
the normal configuration, decisions are taken one hour ahead on the 24 h-ahead forecast, and the effect is
judged on what the meters then measure. Each scenario runs on its own throwaway database.

Columns
    forecast_tx_hours_over     transformer-hours whose forecast is above the threshold in the configuration in
                               force (what the controller reacts to)
    reroutes, restores, mitigations, no_safe_option   decisions taken
    switch_operations          switch operations executed
    hours_rerouted             hours run in a configuration other than normal
    measured_tx_hours_over_*   measured transformer-hours above the threshold, without control (normal
                               configuration) and with the closed loop
    measured_hours_over_100_*  hours in which some transformer was measured above 100% of its rating

For a synthetic city (--city), "hours" are the city's 15-minute steps, the default window is its peak month
(May), and two more columns describe the persistence forecast over that window: forecast_nRMSE_% and
band_coverage_% (share of true loads inside the P10-P90 band; the target is 80%). With meters (Phase 8),
what the operator saw is compared with the truth:
    tx_steps_not_metered       transformer-steps without a transformer-meter reading (smart-meter estimate,
                               forecast or last value used instead)
    observed_error_pp_mean, _p99   |observed - true| loading, percentage points
    overloads_missed           transformer-steps above the threshold that the operator saw at or below it
    overloads_false            transformer-steps the operator saw above the threshold that were not

    python sim/closed_loop.py --city demo --data-quality     # docs/data_quality_city_demo.csv: every scenario
                                                             # with perfect measurement and with the meters
"""
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "grid", ROOT / "sim"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from core import scenarios  # noqa: E402

OUT = ROOT / "docs" / "closed_loop.csv"


def forecast_quality(o, times):
    """A synthetic city's forecast over these times, against the true loads: nRMSE and P10-P90 band coverage."""
    f = o.forecasts
    f._refresh()
    a, p = o.demand.table.loc[times], f.point_.loc[times]
    lo, hi = f.lower_.loc[times], f.upper_.loc[times]
    ok = p.notna() & lo.notna()
    err = (p - a)[ok]
    inside = ((a >= lo) & (a <= hi))[ok]
    return {"forecast_nRMSE_%": round(float(100 * (err ** 2).stack().mean() ** 0.5 / a[ok].stack().mean()), 1),
            "band_coverage_%": round(float(100 * inside.stack().mean()), 1)}


def visibility(steps, threshold):
    """What the operator saw against the truth, per transformer-step (empty without meters)."""
    if "observed_pct" not in steps or steps["observed_pct"].isna().all():
        return {"tx_steps_not_metered": 0, "observed_error_pp_mean": 0.0, "observed_error_pp_p99": 0.0,
                "overloads_missed": 0, "overloads_false": 0}
    rows = steps.dropna(subset=["observed_pct", "measured_pct"])
    true = pd.DataFrame(list(rows["measured_pct"]))
    seen = pd.DataFrame(list(rows["observed_pct"]))[true.columns]
    q = pd.DataFrame(list(rows["data_quality"]))
    err = (seen - true).abs().stack()
    return {"tx_steps_not_metered": int((q != "measured").sum().sum()),
            "observed_error_pp_mean": round(float(err.mean()), 2),
            "observed_error_pp_p99": round(float(err.quantile(0.99)), 2),
            "overloads_missed": int(((true > threshold) & (seen <= threshold)).sum().sum()),
            "overloads_false": int(((seen > threshold) & (true <= threshold)).sum().sum())}


def run(scenario_name, mode="auto", db_dir=None, frame=None, start=None, end=None, meters=None, forecast=None):
    """Run one scenario over the whole period (or start..end). Returns (summary row, steps DataFrame).
    meters (a synthetic city): None as configured in city.yaml, False perfect measurement, True with meters.
    forecast: the forecast source (None: as configured; a city: short or persistence)."""
    from orchestrator import Orchestrator, benchmark_scenario, city_name, prepare_database
    from sources import CitySource
    from storage import Store
    city = city_name()
    with tempfile.TemporaryDirectory(dir=db_dir) as d:
        store = Store(f"sqlite:///{Path(d).as_posix()}/closed_loop.db")
        if city:
            prepare_database(store)
        else:
            store.import_forecast_csv(ROOT / "data" / "forecast.csv")
            store.import_intervals_csv(ROOT / "data" / "forecast_intervals.csv")
        demand = (CitySource(store, city, benchmark_scenario(scenario_name), meters=meters)
                  if city and meters is not None else None)
        o = Orchestrator(store, scenario=scenario_name, service=f"closed-loop {mode}", demand=demand,
                         forecast_source=forecast or (None if city else "precomputed"))
        o.ctl.set_mode(mode)
        times = [t for t in o.demand.times() if (start is None or t >= pd.Timestamp(start))
                 and (end is None or t <= pd.Timestamp(end))]
        t0 = time.perf_counter()
        steps = pd.DataFrame([o.step(t, run_id=1) for t in times]).drop(columns=["rec"])
        wall = time.perf_counter() - t0
        ops = store.actions(controller=o.service, limit=None)       # every operation (the default view stops at 500)
        quality = forecast_quality(o, times) if city else {}
        store.engine.dispose()
    dec = steps["decision"].value_counts()
    nn = lambda s: int(s.fillna(0).sum())  # noqa: E731
    no_ctl, with_ctl = nn(steps["measured_over_normal"]), nn(steps["measured_over"])
    row = {"scenario": scenario_name, "mode": mode, **({"forecast": o.forecasts.name} if city else {}),
           "steps": len(steps),
           "forecast_tx_hours_over": nn(steps["forecast_over"]),
           "reroutes": int(dec.get("reroute", 0)), "restores": int(dec.get("restore", 0)),
           "mitigations": int(dec.get("mitigate", 0)), "no_safe_option": int(dec.get("no_safe_option", 0)),
           "switch_operations": len(ops),
           "hours_rerouted": int((steps["config_before"].map(len) > 0).sum()),
           "measured_tx_hours_over_no_control": no_ctl, "measured_tx_hours_over_with_control": with_ctl,
           "reduction_%": round(100 * (no_ctl - with_ctl) / no_ctl, 1) if no_ctl else None,
           "measured_hours_over_100_no_control": int((steps["measured_max_normal_pct"] > 100).sum()),
           "measured_hours_over_100_with_control": int((steps["measured_max_pct"] > 100).sum()),
           **quality, **(visibility(steps, o.threshold) if city else {}), "mean_step_ms": round(float(steps["duration_ms"].mean()), 1), "wall_s": round(wall, 1)}
    return row, steps


TIMING = ["mean_step_ms", "wall_s"]           # machine-dependent columns (left out of the golden comparison)


CITY_WINDOW = ("2023-05-01 00:00", "2023-05-31 23:45")     # the peak month of the demo city (heatwave included)


def main(argv=None):
    import argparse
    from core import config
    ap = argparse.ArgumentParser(description="Closed-loop results for every benchmark scenario.")
    ap.add_argument("--city", help="a synthetic city (cities/<name>); default: the eMARC test period")
    ap.add_argument("--from", dest="start", help="first step (default: a city's peak month, May)")
    ap.add_argument("--to", dest="end")
    ap.add_argument("--data-quality", action="store_true",
                    help="a city: run every scenario with perfect measurement and with the meters")
    ap.add_argument("--scenario", action="append", help="only this scenario (repeatable)")
    ap.add_argument("--out", help="write here instead of docs/")
    ap.add_argument("--forecast", choices=["short", "persistence"], help="a city's forecast (default: as configured)")
    a = ap.parse_args(argv)
    if a.data_quality and not a.city:
        ap.error("--data-quality needs --city")
    config.use_city(a.city)
    start, end = (a.start or CITY_WINDOW[0], a.end or CITY_WINDOW[1]) if a.city else (a.start, a.end)
    out = OUT if not a.city else ROOT / "docs" / f"closed_loop_city_{a.city}.csv"
    if a.data_quality:
        out = ROOT / "docs" / f"data_quality_city_{a.city}.csv"
    if a.city and a.forecast and a.forecast != config.simulation()["orchestrator"]["forecast_source"]:
        out = out.with_name(out.stem + f"_{a.forecast}.csv")
    out = Path(a.out) if a.out else out
    variants = [("perfect", False), ("meters", True)] if a.data_quality else [(None, None)]
    rows = []
    for s in scenarios.benchmark():
        if a.scenario and s.name not in a.scenario:
            continue
        for label, meters in variants:
            row, _ = run(s.name, start=start, end=end, meters=meters, forecast=a.forecast)
            if label:
                row = {"measurement": label, **row}
            if a.city:
                row = {**row, "from": start, "to": end, "step_minutes": config.simulation()["clock"]["step_minutes"]}
            rows.append(row)
            print(f"{s.name:40s} {label or '':8s} measured transformer-steps over: "
                  f"{row['measured_tx_hours_over_no_control']:5d} -> {row['measured_tx_hours_over_with_control']:5d}"
                  f"   ({row['wall_s']} s)", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
