"""Performance of the closed loop (gap item H7): step time per stage, throughput, memory, database growth,
the headroom against the accelerated clock, and how closely the real clock holds a target speed.

    python sim/benchmark.py                  # writes docs/performance.csv (about 1 minute)
    python sim/benchmark.py --steps 200      # quicker

The numbers depend on the computer; the file records which one (platform, CPU, Python).
"""
import argparse
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "grid", ROOT / "sim"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from core import config  # noqa: E402

OUT = ROOT / "docs" / "performance.csv"
SCENARIO = "S4 Severe stress (all x2.5)"      # the busiest scenario: most candidate evaluations per step


def _rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2 ** 20
    except ImportError:
        return float("nan")


CITY_SCENARIO, CITY_START = "C4 Heatwave and EV adoption", "2023-05-08 00:00"   # the city's busiest week


def loop_benchmark(steps=None, scenario=None, db_dir=None, start=None):
    """Run the orchestrator as fast as possible; per-stage timings, throughput, memory and database growth."""
    from orchestrator import STAGES, Orchestrator, city_name, prepare_database
    from storage import Store
    city = city_name()
    scenario = scenario or (CITY_SCENARIO if city else SCENARIO)
    start = start or (CITY_START if city else None)
    rows = []
    with tempfile.TemporaryDirectory(dir=db_dir) as d:
        path = Path(d) / "bench.db"
        store = Store(f"sqlite:///{path.as_posix()}")
        if city:
            prepare_database(store)
        else:
            store.import_forecast_csv(ROOT / "data" / "forecast.csv")
            store.import_intervals_csv(ROOT / "data" / "forecast_intervals.csv")
        rss0 = _rss_mb()
        o = Orchestrator(store, scenario=scenario, service="benchmark", forecast_source=None if city else "precomputed")
        o.ctl.set_mode("auto")
        db0 = path.stat().st_size + (Path(str(path) + "-wal").stat().st_size if Path(str(path) + "-wal").exists() else 0)
        rss_start, rss_peak = _rss_mb(), 0.0
        times = [t for t in o.demand.times() if start is None or t >= pd.Timestamp(start)]
        times = times[:steps] if steps else times
        warm = min(100, len(times) // 5)                  # memory growth is measured after a warm-up
        rss_warm = None
        t0 = time.perf_counter()
        for i, t in enumerate(times):
            r = o.step(t, run_id=1)
            rows.append({**r["timings_ms"], "total": r["duration_ms"]})
            rss_peak = max(rss_peak, _rss_mb())
            if i + 1 == warm:
                rss_warm = _rss_mb()
        wall = time.perf_counter() - t0
        rss_end = _rss_mb()
        with store.engine.connect() as c:                      # fold the write-ahead log into the file
            c.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        db1 = path.stat().st_size
        store.engine.dispose()
    t = pd.DataFrame(rows)
    out = []
    for stage in list(STAGES) + ["total"]:
        v = t[stage].to_numpy()
        out += [(f"step_{stage}_mean", np.mean(v), "ms"), (f"step_{stage}_p95", np.percentile(v, 95), "ms"),
                (f"step_{stage}_max", np.max(v), "ms")]
    step_s = config.simulation()["clock"]["step_minutes"] * 60
    mean_s = t["total"].mean() / 1000
    acc = config.simulation()["clock"]["accelerated_factor"]
    out += [("steps", len(t), "steps"), ("scenario", scenario, ""), ("throughput", len(t) / wall, "steps/s"),
            ("max_speed_factor", step_s / mean_s, "simulated s per wall s"),
            ("simulated_year_wall_time_at_max", 365 * 86400 / step_s * mean_s / 60, "min"),
            ("accelerated_budget_per_60min_step", 3600 / acc * 1000, "ms"),
            ("accelerated_budget_per_15min_step", 900 / acc * 1000, "ms"),
            ("accelerated_budget_used_60min", 100 * mean_s / (3600 / acc), "%"),
            ("accelerated_budget_used_this_step", 100 * mean_s / (step_s / acc), f"% of {step_s / acc * 1000:.0f} ms"),
            ("memory_rss_before_engine", rss0, "MB"), ("memory_rss_start", rss_start, "MB"),
            ("memory_rss_peak", rss_peak, "MB"), ("memory_rss_end", rss_end, "MB"),
            ("memory_growth_per_1000_steps", (rss_end - rss_warm) / max(1, len(t) - warm) * 1000 if rss_warm else
             float("nan"), "MB (after warm-up)"),
            ("database_growth_per_step", (db1 - db0) / len(t) / 1024, "KB"),
            ("database_growth_per_simulated_year", (db1 - db0) / len(t) * 365 * 86400 / step_s / 2 ** 20, "MB")]
    return out


def clock_benchmark(target_factor=36000.0, steps=30, db_dir=None):
    """Run the real service loop at a target speed and measure the achieved speed and the timing error."""
    from service import Service
    from storage import Store
    with tempfile.TemporaryDirectory(dir=db_dir) as d:
        store = Store(f"sqlite:///{(Path(d) / 'clock.db').as_posix()}")
        svc = Service(store, name="clock-benchmark")
        svc.ctl.set_mode("monitor")
        svc.execute("speed", {"speed": target_factor})
        svc.execute("resume")
        t0 = time.monotonic()
        svc.run(max_steps=steps)
        wall = time.monotonic() - t0
        achieved = svc.clock.achieved_factor
        expected = (steps - 1) * svc.clock.interval_s           # the first step runs at once
        store.engine.dispose()
    return [("clock_target_factor", target_factor, "simulated s per wall s"),
            ("clock_interval", svc.clock.interval_s * 1000, "ms"),
            ("clock_achieved_factor", achieved, "simulated s per wall s"),
            ("clock_speed_error", 100 * (achieved - target_factor) / target_factor, "%"),
            ("clock_wall_for_steps", wall, "s"), ("clock_expected_wall_for_steps", expected, "s")]


def machine():
    try:
        import psutil
        ram = f"{psutil.virtual_memory().total / 2 ** 30:.1f} GB RAM"
    except ImportError:
        ram = ""
    return [("machine", f"{platform.system()} {platform.release()}, {platform.processor() or platform.machine()}, "
                        f"{os.cpu_count()} CPUs {ram}".strip(), ""),
            ("python", platform.python_version(), "")]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Measure the closed loop's performance.")
    ap.add_argument("--steps", type=int, help="steps to time (default: the whole test period; a city: one week)")
    ap.add_argument("--city", help="a synthetic city (cities/<name>)")
    a = ap.parse_args(argv)
    from core import config as cfg
    cfg.use_city(a.city)
    steps = a.steps or (7 * 96 if a.city else None)
    rows = machine() + loop_benchmark(steps) + clock_benchmark(target_factor=3600.0 if a.city else 36000.0)
    df = pd.DataFrame(rows, columns=["metric", "value", "unit"])
    df["value"] = [round(v, 3) if isinstance(v, float) else v for v in df["value"]]
    out = OUT if not a.city else ROOT / "docs" / f"performance_city_{a.city}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
