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


def run(scenario_name, mode="auto", db_dir=None, frame=None):
    """Run one scenario over the whole test period. Returns (summary row, steps DataFrame)."""
    from orchestrator import Orchestrator
    from storage import Store
    with tempfile.TemporaryDirectory(dir=db_dir) as d:
        store = Store(f"sqlite:///{Path(d).as_posix()}/closed_loop.db")
        store.import_forecast_csv(ROOT / "data" / "forecast.csv")
        store.import_intervals_csv(ROOT / "data" / "forecast_intervals.csv")
        o = Orchestrator(store, scenario=scenario_name, service=f"closed-loop {mode}", forecast_source="precomputed")
        o.ctl.set_mode(mode)
        t0 = time.perf_counter()
        steps = pd.DataFrame([o.step(t, run_id=1) for t in o.demand.times()]).drop(columns=["rec"])
        wall = time.perf_counter() - t0
        ops = store.actions(controller=o.service)
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
           "mean_step_ms": round(float(steps["duration_ms"].mean()), 1), "wall_s": round(wall, 1)}
    return row, steps


TIMING = ["mean_step_ms", "wall_s"]           # machine-dependent columns (left out of the golden comparison)


def main():
    rows = []
    for s in scenarios.benchmark():
        row, _ = run(s.name)
        rows.append(row)
        print(f"{s.name:40s} measured transformer-hours over: {row['measured_tx_hours_over_no_control']:4d} -> "
              f"{row['measured_tx_hours_over_with_control']:4d}   ({row['wall_s']} s)")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
