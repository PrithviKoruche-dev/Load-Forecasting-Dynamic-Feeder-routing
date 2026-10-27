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
band_coverage_% (share of measured loads inside the P10-P90 band; the target is 80%).
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
    """Persistence forecast of a synthetic city over these times: nRMSE and P10-P90 band coverage."""
    f = o.forecasts
    f._refresh()
    a, p = o.demand.table.loc[times], f.point_.loc[times]
    lo, hi = f.lower_.loc[times], f.upper_.loc[times]
    ok = p.notna() & lo.notna()
    err = (p - a)[ok]
    inside = ((a >= lo) & (a <= hi))[ok]
    return {"forecast_nRMSE_%": round(float(100 * (err ** 2).stack().mean() ** 0.5 / a[ok].stack().mean()), 1),
            "band_coverage_%": round(float(100 * inside.stack().mean()), 1)}


def run(scenario_name, mode="auto", db_dir=None, frame=None, start=None, end=None):
    """Run one scenario over the whole period (or start..end). Returns (summary row, steps DataFrame)."""
    from orchestrator import Orchestrator, city_name, prepare_database
    from storage import Store
    city = city_name()
    with tempfile.TemporaryDirectory(dir=db_dir) as d:
        store = Store(f"sqlite:///{Path(d).as_posix()}/closed_loop.db")
        if city:
            prepare_database(store)
        else:
            store.import_forecast_csv(ROOT / "data" / "forecast.csv")
            store.import_intervals_csv(ROOT / "data" / "forecast_intervals.csv")
        o = Orchestrator(store, scenario=scenario_name, service=f"closed-loop {mode}",
                         forecast_source=None if city else "precomputed")
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
    row = {"scenario": scenario_name, "mode": mode, "steps": len(steps),
           "forecast_tx_hours_over": nn(steps["forecast_over"]),
           "reroutes": int(dec.get("reroute", 0)), "restores": int(dec.get("restore", 0)),
           "mitigations": int(dec.get("mitigate", 0)), "no_safe_option": int(dec.get("no_safe_option", 0)),
           "switch_operations": len(ops),
           "hours_rerouted": int((steps["config_before"].map(len) > 0).sum()),
           "measured_tx_hours_over_no_control": no_ctl, "measured_tx_hours_over_with_control": with_ctl,
           "reduction_%": round(100 * (no_ctl - with_ctl) / no_ctl, 1) if no_ctl else None,
           "measured_hours_over_100_no_control": int((steps["measured_max_normal_pct"] > 100).sum()),
           "measured_hours_over_100_with_control": int((steps["measured_max_pct"] > 100).sum()),
           **quality, "mean_step_ms": round(float(steps["duration_ms"].mean()), 1), "wall_s": round(wall, 1)}
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
    a = ap.parse_args(argv)
    config.use_city(a.city)
    start, end = (a.start or CITY_WINDOW[0], a.end or CITY_WINDOW[1]) if a.city else (a.start, a.end)
    out = OUT if not a.city else ROOT / "docs" / f"closed_loop_city_{a.city}.csv"
    rows = []
    for s in scenarios.benchmark():
        row, _ = run(s.name, start=start, end=end)
        if a.city:
            row = {**row, "from": start, "to": end, "step_minutes": config.simulation()["clock"]["step_minutes"]}
        rows.append(row)
        print(f"{s.name:40s} measured transformer-steps over: {row['measured_tx_hours_over_no_control']:5d} -> "
              f"{row['measured_tx_hours_over_with_control']:5d}   ({row['wall_s']} s)")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
