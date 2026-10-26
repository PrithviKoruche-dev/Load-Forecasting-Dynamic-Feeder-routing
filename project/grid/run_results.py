"""Run scenarios S1-S5 over the whole held-out test period on the real forecast.

Writes docs/results.csv, docs/results_worst_hour.csv, docs/results_mitigation.csv and docs/results.md
(tables for the report).
Scenarios come from config/scenarios.yaml (the "benchmark" list) via core/scenarios.py,
the same implementation the dashboard uses.

The work is done by run_all(), so tests can call it without writing files.
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project/ -> import core
import engine as e  # noqa: E402
from core import scenarios  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
THR = e.THRESHOLD


def load_predictions(path=ROOT / "data" / "forecast.csv"):
    """Hourly predicted kW, one column per transformer."""
    fc = pd.read_csv(path, parse_dates=["timestamp"])
    return fc.pivot(index="timestamp", columns="transformer_id", values="predicted_kw")


def run_scenario(scenario, pred, threshold=THR):
    """Run one scenario over every hour. Returns (summary row, worst-hour row or None, per-hour records)."""
    name = scenario.name
    recs = []
    for ts, r in pred.iterrows():
        loads = {tx: float(scenario.adjust(tx, ts, r[tx])) for tx in e.TX}
        res = e.evaluate_and_reroute(loads, threshold)
        b, a = res["before"], res["after"]
        recs.append({
            "ts": ts, "status": res["status"], "compute_s": res["compute_s"],
            "n_over": sum(v > threshold for v in b["loading_by_tx"].values()) if b else 0,
            "before_max": b["max_loading"] if b else None, "after_max": a["max_loading"] if a else None,
            "before_v": b["min_voltage_pu"] if b else None, "after_v": a["min_voltage_pu"] if a else None,
            "n_moves": len(res["action"]),
            "partial": len(res.get("best_partial_target") or []),
        })
    d = pd.DataFrame(recs)
    over = d[d["status"] != "no_action"]          # same definition the algorithm used
    fixed = d[d["status"] == "rerouted"]
    unsolved = d[d["status"] == "no_safe_option"]
    row = {
        "scenario": name, "hours": len(d), "overload_hours": len(over),
        "transformer_hours_over": int(d["n_over"].sum()),
        "rerouted": len(fixed), "no_safe_option": len(unsolved),
        "resolved_%": round(100 * len(fixed) / len(over), 1) if len(over) else None,
        "mean_peak_before_%": round(over["before_max"].mean(), 1) if len(over) else None,
        "mean_peak_after_% (rerouted)": round(fixed["after_max"].mean(), 1) if len(fixed) else None,
        "mean_moves": round(fixed["n_moves"].mean(), 2) if len(fixed) else None,
        "mean_time_s": round(d["compute_s"].mean(), 3), "max_time_s": round(d["compute_s"].max(), 3),
    }
    worst = None
    if len(over):                                   # the single worst hour of this scenario
        w = over.loc[over["before_max"].idxmax()]
        worst = {"scenario": name, "worst_hour": w["ts"].strftime("%d %b %H:%M"), "status": w["status"],
                 "peak_before_%": w["before_max"], "peak_after_%": w["after_max"],
                 "min_V_before_pu": w["before_v"], "min_V_after_pu": w["after_v"]}
    return row, worst, d


def mitigation_row(name, d):
    """Hours with no complete fix, and how many of them have a mitigation: a configuration that respects
    every hard limit and lowers the peak (what grid/decisions.py offers as kind "mitigate")."""
    nso = d[d["status"] == "no_safe_option"]
    mit = nso[nso["partial"] > 0]
    removes = mit[(mit["before_max"] > e.HARD_LIMIT) & (mit["after_max"] <= e.HARD_LIMIT)]
    return {"scenario": name, "no_safe_option_hours": len(nso), "mitigation_available": len(mit),
            "removes_transformer_over_100%": len(removes),
            "mean_peak_before_% (mitigated)": round(mit["before_max"].mean(), 1) if len(mit) else None,
            "mean_peak_after_% (mitigated)": round(mit["after_max"].mean(), 1) if len(mit) else None}


def run_all(pred=None, threshold=THR, verbose=False, with_mitigation=False):
    """Run every scenario. Returns (summary DataFrame, worst-hour DataFrame), plus the mitigation
    DataFrame when with_mitigation is True."""
    pred = load_predictions() if pred is None else pred
    rows, worst_rows, mit_rows = [], [], []
    for sc in scenarios.benchmark():
        t0 = time.time()
        row, worst, hourly = run_scenario(sc, pred, threshold)
        rows.append(row)
        mit_rows.append(mitigation_row(sc.name, hourly))
        if worst is not None:
            worst_rows.append(worst)
        if verbose:
            print(f"{sc.name}: done in {time.time() - t0:.0f} s")
    out = (pd.DataFrame(rows), pd.DataFrame(worst_rows))
    return out + (pd.DataFrame(mit_rows),) if with_mitigation else out


def main():
    summary, worst, mitigation = run_all(verbose=True, with_mitigation=True)
    summary.to_csv(ROOT / "docs" / "results.csv", index=False)
    worst.to_csv(ROOT / "docs" / "results_worst_hour.csv", index=False)
    mitigation.to_csv(ROOT / "docs" / "results_mitigation.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\nSUMMARY OVER THE WHOLE TEST PERIOD\n", summary.set_index("scenario").T.to_string())
    print("\nWORST OVERLOAD HOUR PER SCENARIO (before vs after the algorithm's best attempt)\n", worst.to_string(index=False))
    print("\nMITIGATION WHEN NO COMPLETE FIX EXISTS\n", mitigation.to_string(index=False))

    try:
        md = [f"## Scenario results (test period, threshold {THR:g}%)\n", summary.to_markdown(index=False), "",
              "## Worst overload hour per scenario\n", worst.to_markdown(index=False), "",
              "## Mitigation when no complete fix exists\n", mitigation.to_markdown(index=False), ""]
        (ROOT / "docs" / "results.md").write_text("\n".join(md))
    except ImportError:
        print("(skipped results.md: run  pip install tabulate  to also get the Markdown tables)")
    print("\nSaved docs/results.csv, docs/results_worst_hour.csv and docs/results_mitigation.csv")


if __name__ == "__main__":
    main()
