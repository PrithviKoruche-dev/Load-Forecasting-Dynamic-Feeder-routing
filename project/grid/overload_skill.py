"""How well does the forecast predict overloads? (gap-analysis item E9, overload classification)

For every benchmark scenario and every test hour, the loads are put through the power flow three times, in
the normal configuration: the point forecast, the P90 upper bound (data/forecast_intervals.csv) and what was
actually measured, each with the scenario's stress applied. Each transformer-hour is then a classification:

    predicted overload  forecast (or P90) loading above the threshold
    actual overload     measured loading above the threshold

Precision: of the transformer-hours flagged in advance, the share that really overloaded.
Recall:    of the transformer-hours that really overloaded, the share flagged 24 h in advance.

    python grid/overload_skill.py   ->  docs/metrics_overload.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine as e  # noqa: E402
from core import scenarios  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_frames():
    fc = pd.read_csv(ROOT / "data" / "forecast.csv", parse_dates=["timestamp"])
    iv = pd.read_csv(ROOT / "data" / "forecast_intervals.csv", parse_dates=["timestamp"])
    upper = [c for c in iv.columns if c.startswith("p") and c.endswith("_kw")][-1]
    piv = lambda df, col: df.pivot(index="timestamp", columns="transformer_id", values=col)  # noqa: E731
    return piv(fc, "predicted_kw"), piv(iv, upper), piv(fc, "actual_kw"), upper


def loadings(loads):
    m = e.evaluate_state(loads, [])
    return m["loading_by_tx"] if m else None


def classify(scenario, point, upper, actual, threshold):
    rows = []
    for ts in point.index:
        get = lambda frame: {tx: float(scenario.adjust(tx, ts, frame.loc[ts, tx])) for tx in e.TX}  # noqa: E731
        lp, lu, la = loadings(get(point)), loadings(get(upper)), loadings(get(actual))
        if lp is None or lu is None or la is None:
            continue
        for tx in e.TX:
            rows.append({"ts": ts, "tx": tx, "actual": la[tx] > threshold,
                         "point": lp[tx] > threshold, "upper": lu[tx] > threshold})
    return pd.DataFrame(rows)


def scores(d, col):
    tp = int((d[col] & d["actual"]).sum())
    fp = int((d[col] & ~d["actual"]).sum())
    fn = int((~d[col] & d["actual"]).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    return {"flagged": tp + fp, "actual_overloads": tp + fn, "true_pos": tp, "false_pos": fp, "missed": fn,
            "precision_%": round(100 * prec, 1) if prec is not None else None,
            "recall_%": round(100 * rec, 1) if rec is not None else None,
            "F1_%": round(100 * f1, 1) if f1 is not None else None}


def run(threshold=None):
    threshold = e.THRESHOLD if threshold is None else threshold
    point, upper, actual, upper_name = load_frames()
    rows = []
    for sc in scenarios.benchmark():
        d = classify(sc, point, upper, actual, threshold)
        for col, label in (("point", "point forecast"), ("upper", f"upper bound ({upper_name[:-3].upper()})")):
            rows.append({"scenario": sc.name, "flag_on": label, "transformer_hours": len(d), **scores(d, col)})
    return pd.DataFrame(rows)


def main():
    res = run()
    res.to_csv(ROOT / "docs" / "metrics_overload.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False))
    print("\nSaved docs/metrics_overload.csv")


if __name__ == "__main__":
    main()
