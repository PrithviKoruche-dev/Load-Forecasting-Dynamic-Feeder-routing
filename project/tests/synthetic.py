"""Synthetic data in the clusters.parquet format, for testing the forecasting code without the eMARC data."""
import numpy as np
import pandas as pd


def make_clusters(days=150, start="2019-06-01", seed=0):
    """Hourly load_kw_raw and temp_c for T1-T6 with daily and weekly cycles, temperature response and noise."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=24 * days, freq="1h")
    hour, dow, doy = ts.hour.to_numpy(), ts.day_of_week.to_numpy(), ts.day_of_year.to_numpy()
    temp = 26 + 5 * np.sin(2 * np.pi * (doy - 100) / 365) + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    parts = []
    for i, tx in enumerate(["T1", "T2", "T3", "T4", "T5", "T6"]):
        base = 1.5 + 0.4 * i
        shape = 1 + 0.6 * np.exp(-((hour - 20) ** 2) / 6) + 0.3 * np.exp(-((hour - 8) ** 2) / 4)
        load = base * shape * (1 + 0.08 * (dow >= 5)) * (1 + 0.03 * np.clip(temp - 22, 0, None))
        load = load * rng.lognormal(0, 0.12, len(ts))
        parts.append(pd.DataFrame({"timestamp": ts, "transformer_id": tx, "region": "synthetic", "n_households": 10,
                                   "load_kw_raw": load, "temp_c": temp + rng.normal(0, 0.5, len(ts))}))
    return pd.concat(parts, ignore_index=True)
