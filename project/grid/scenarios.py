"""Test scenarios S1-S5 for evaluate_and_reroute (run: python grid/scenarios.py)."""
import engine as e

BASE = {t: 300.0 for t in e.TX}          # ~64% loading each


def scale(base, factor, only=None):
    return {t: kw * (factor if only is None or t in only else 1.0) for t, kw in base.items()}


SCENARIOS = {
    "S1 Normal day":       BASE,
    "S2 Heatwave (T1 x1.4)": scale(BASE, 1.4, only=["T1"]),
    "S3 EV surge (T4 +150 kW)": {**BASE, "T4": BASE["T4"] + 150},
    "S4 Severe stress (all x1.6)": scale(BASE, 1.6),
    "S5 Edge case (T3 x4)":  scale(BASE, 4.0, only=["T3"]),
}

if __name__ == "__main__":
    for name, loads in SCENARIOS.items():
        r = e.evaluate_and_reroute(loads)
        print(f"\n=== {name} ===")
        print("status:", r["status"], "| action:", r["action"], "| time:", r["compute_s"], "s")
        if r["before"]:
            print("before:", r["before"]["max_loading"], "% | after:", r["after"]["max_loading"], "%")
            print("min V before/after:", r["before"]["min_voltage_pu"], "/", r["after"]["min_voltage_pu"])
        for line in r["log"]:
            print("  -", line)