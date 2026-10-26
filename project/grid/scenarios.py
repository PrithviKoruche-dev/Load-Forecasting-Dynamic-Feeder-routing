"""Quick check of evaluate_and_reroute on a flat load (run: python grid/scenarios.py).

Scenarios come from the "quick" list in config/scenarios.yaml. For the full benchmark on the
real forecast, run grid/run_results.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project/ -> import core
import engine as e  # noqa: E402
from core import scenarios  # noqa: E402


def main():
    base_kw, quick = scenarios.quick()
    base = {t: float(base_kw) for t in e.TX}
    for sc in quick:
        r = e.evaluate_and_reroute(sc.loads(base))
        print(f"\n=== {sc.name} ===  ({sc.describe()})")
        print("status:", r["status"], "| action:", r["action"], "| time:", r["compute_s"], "s")
        if r["before"]:
            print("before:", r["before"]["max_loading"], "% | after:", r["after"]["max_loading"], "%")
            print("min V before/after:", r["before"]["min_voltage_pu"], "/", r["after"]["min_voltage_pu"])
        for line in r["log"]:
            print("  -", line)


if __name__ == "__main__":
    main()
