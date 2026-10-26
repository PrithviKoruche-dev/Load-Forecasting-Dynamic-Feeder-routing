"""Feature engineering for the 24-hour-ahead (direct) forecast.

Every feature for target hour t uses only information available 24 h earlier (lags >= 24),
plus calendar fields and temperature at t (treated as a perfect weather forecast).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project/ -> import core
from core import config  # noqa: E402

# The feature definitions below ARE the model's input contract (model_xgb.json stores these names),
# so they stay in code. Settings that are not part of that contract come from config/.
TX_LIST = list(config.grid()["transformers"])
CDD_BASE_C = config.forecasting()["features"]["cdd_base_c"]
LAGS = [24, 48, 168]
FEATURES = ["tx", "hour", "dow", "is_weekend",
            "lag_24", "lag_48", "lag_168",
            "roll_mean_3", "roll_mean_24", "roll_mean_168", "roll_max_24",
            "temp", "temp_lag_24", "cdd"]


def split_times(timestamps, train=None, val=None):
    """Time-based split boundaries: train < t1 <= val < t2 <= test. Shares default to config/forecasting.yaml."""
    split = config.forecasting()["split"]
    train = split["train"] if train is None else train
    val = split["val"] if val is None else val
    ts = np.sort(pd.unique(timestamps))
    i1, i2 = int(len(ts) * train), int(len(ts) * (train + val))
    return pd.Timestamp(ts[i1]), pd.Timestamp(ts[i2])


def train_stats(df, t1):
    tr = df[df["timestamp"] < t1]
    mean = tr.groupby("transformer_id")["load_kw_raw"].mean()
    peak = tr.groupby("transformer_id")["load_kw_raw"].max()
    return mean, peak


def build_features(df, train_mean):
    """Load is divided by each transformer's TRAIN mean so one model can serve all six."""
    out = []
    for tx, g in df.sort_values("timestamp").groupby("transformer_id"):
        g = g.set_index("timestamp")
        y = g["load_kw_raw"] / train_mean[tx]
        f = pd.DataFrame(index=g.index)
        f["transformer_id"] = tx
        f["tx"] = pd.Categorical([tx] * len(g), categories=TX_LIST)
        f["hour"] = g.index.hour
        f["dow"] = g.index.day_of_week
        f["is_weekend"] = (f["dow"] >= 5).astype(int)
        for L in LAGS:
            f[f"lag_{L}"] = y.shift(L)
        s = y.shift(24)                                   # newest value known at forecast time
        f["roll_mean_3"] = s.rolling(3, min_periods=2).mean()
        f["roll_mean_24"] = s.rolling(24, min_periods=18).mean()
        f["roll_mean_168"] = s.rolling(168, min_periods=120).mean()
        f["roll_max_24"] = s.rolling(24, min_periods=18).max()
        f["temp"] = g["temp_c"]
        f["temp_lag_24"] = g["temp_c"].shift(24)
        f["cdd"] = (g["temp_c"] - CDD_BASE_C).clip(lower=0)   # cooling-degree feature
        f["y"] = y
        f["actual_raw"] = g["load_kw_raw"]
        out.append(f.reset_index())
    return pd.concat(out, ignore_index=True)