"""Calibrate the synthetic city's residential demand from the real eMARC households (gap items C1, C4).

    python city/calibrate.py        # needs data/processed/clusters.parquet; writes city/profiles/residential_emarc.csv

Model, per household and per hour of the day h (least squares over every hour of the eMARC data):

    kW(t) = base[season(t), daytype(t), h] + cooling[h] * max(0, T(t) - 24 °C)

season from the month (config/simulation.yaml calendar: winter Dec-Feb, summer Mar-May, monsoon Jun-Sep,
transition Oct-Nov), daytype weekday or weekend. The eMARC data runs from June 2019 to 24 March 2020, so
"summer" is fitted from March only; the hotter April and May of a simulated year come through the cooling
term. The output is committed, so the city works without the eMARC data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import config  # noqa: E402
from core.clock import season  # noqa: E402

OUT = Path(__file__).resolve().parent / "profiles" / "residential_emarc.csv"
COOLING_BASE_C = 24.0


def fit(clusters):
    d = clusters.dropna(subset=["load_kw_raw"]).copy()
    d["kw"] = d["load_kw_raw"] / d["n_households"]                   # per household
    seasons = config.simulation()["seasons"]
    d["season"] = d["timestamp"].map(lambda t: season(t, seasons))
    d["daytype"] = np.where(d["timestamp"].dt.day_of_week >= 5, "weekend", "weekday")
    d["hour"] = d["timestamp"].dt.hour
    d["cdd"] = np.clip(d["temp_c"] - COOLING_BASE_C, 0, None)
    keys = sorted(set(zip(d["season"], d["daytype"], d["hour"])))
    index = {k: i for i, k in enumerate(keys)}
    n_base = len(keys)
    X = np.zeros((len(d), n_base + 24))
    rows = np.arange(len(d))
    X[rows, [index[k] for k in zip(d["season"], d["daytype"], d["hour"])]] = 1.0
    X[rows, n_base + d["hour"].to_numpy()] = d["cdd"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, d["kw"].to_numpy(), rcond=None)
    pred = X @ coef
    resid = d["kw"].to_numpy() - pred
    base = pd.DataFrame([{"kind": "base", "season": s, "daytype": dt_, "hour": h, "value": round(float(coef[i]), 5)}
                         for (s, dt_, h), i in index.items()])
    cool = pd.DataFrame([{"kind": "cooling_kw_per_c", "season": "", "daytype": "", "hour": h,
                          "value": round(float(max(coef[n_base + h], 0.0)), 5)} for h in range(24)])
    stats = {"hours": len(d), "households": int(clusters.groupby("transformer_id")["n_households"].first().sum()),
             "r2": round(1 - resid.var() / d["kw"].var(), 3), "mean_kw": round(float(d["kw"].mean()), 4)}
    return pd.concat([base, cool], ignore_index=True), stats


def main():
    src = ROOT / "data" / "processed" / "clusters.parquet"
    if not src.exists():
        raise SystemExit(f"needs {src} (the eMARC clusters); the committed {OUT.name} is used otherwise")
    table, stats = fit(pd.read_parquet(src))
    OUT.parent.mkdir(exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"Saved {OUT.relative_to(ROOT)}: {stats}")


if __name__ == "__main__":
    main()
