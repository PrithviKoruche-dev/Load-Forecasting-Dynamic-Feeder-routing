"""Short-horizon forecast for a synthetic city (Phase 9): every transformer's load one step (15 minutes) ahead.

The persistence forecast of Phase 7 repeats yesterday's load 24 hours later, so it cannot see a heatwave
coming and expects yesterday's peaks at yesterday's times. The switching decision is for the next step only,
so the forecast that matters is the one for the next step, made at the end of this one, from:

    what the operator has recorded   the transformer meters (Phase 8), up to now: the last readings, the
                                     last 4 hours, the same time yesterday and a week ago
    the weather forecast             the temperature expected at the target time (the true temperature plus a
                                     forecast error, city.yaml forecast.temperature_error_c)
    the calendar                     time of day, weekday / Saturday / Sunday or holiday
    the transformer                  the mix of consumers behind it (shares of residential, commercial, ...)

One XGBoost model serves every transformer (loads divided by each transformer's mean load over the training
months). It predicts the change from the last reading. It learns from the city's own recorded history
(city.yaml forecast.train), stops training and sets a P10-P90 band on the following month (forecast.calibrate),
and is tested on May with its heatwave, which it never saw (forecast.test).

    python ml/train_short.py --city demo     # trains it; writes cities/demo/forecast/ and docs/forecast_short_city_demo.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "city"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FEATURES = ["y0", "y1", "y2", "y4", "mean_4h", "day", "day_now", "week", "temp_fc", "temp_now", "temp_yday",
            "hour", "dow", "daytype", "share_residential", "share_commercial", "share_industrial", "share_mixed"]
WEEK = 7 * 96                                    # steps of history the features need (15-minute steps)
DEFAULTS = {"horizon_steps": 1, "temperature_error_c": 0.5, "train": None, "calibrate": None, "test": None,
            "scenarios": None}
XGB = dict(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=5, tree_method="hist", random_state=0, n_jobs=2)
EARLY_STOP = 50
QUANTILES = (0.1, 0.9)
VERSION = "short-15m-v1"                       # model_version in the database
BLOCKS = 6                                       # the band is calibrated per 4-hour block of the day


def settings(spec):
    f = {**DEFAULTS, **(spec.get("forecast") or {})}
    unknown = set(spec.get("forecast") or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"city.yaml: unknown settings forecast.{sorted(unknown)}")
    if not isinstance(f["horizon_steps"], int) or not 1 <= f["horizon_steps"] <= 96:
        raise ValueError("city.yaml: forecast.horizon_steps must be a whole number from 1 to 96")
    if not isinstance(f["temperature_error_c"], (int, float)) or f["temperature_error_c"] < 0:
        raise ValueError("city.yaml: forecast.temperature_error_c must be a number of at least 0")
    return f


def model_dir(city):
    from demand import city_dir
    return city_dir(city) / "forecast"


def temperature_forecast(spec, temp):
    """The weather forecast for each time: the true temperature plus a forecast error (the same in every
    scenario: it is drawn from the seed, not from the events)."""
    sd = settings(spec)["temperature_error_c"]
    rng = np.random.default_rng([int(spec["seed"]), 505])
    return np.asarray(temp, dtype=float) + rng.normal(0, sd, len(temp))


def statics(demand, transformers):
    """Each transformer's mix of consumers: share of its connected size by type."""
    sz = demand.sizes.loc[transformers]
    return (sz.div(sz.sum(axis=1), axis=0)).to_numpy()


def features(history, temp, temp_fc, cal, mix, scale, horizon=1):
    """Feature rows for every origin o (end of step o) and transformer, forecasting step o + horizon.
    history: times x transformers (recorded kW); temp, temp_fc: per time; mix: transformers x 4; scale: per
    transformer. Returns (X DataFrame, target index j per row, transformer index per row, valid origin mask).
    Only values up to o are used, apart from the calendar and the weather forecast for the target time."""
    Y = history.to_numpy(dtype=float) / np.asarray(scale, dtype=float)
    T, K = Y.shape
    o = np.arange(WEEK, T - horizon)
    j = o + horizon
    csum = np.vstack([np.zeros((1, K)), np.cumsum(Y, axis=0)])
    mean4 = (csum[o + 1] - csum[o + 1 - 16]) / 16
    cols = {"y0": Y[o], "y1": Y[o - 1], "y2": Y[o - 2], "y4": Y[o - 4], "mean_4h": mean4,
            "day": Y[j - 96], "day_now": Y[o - 96], "week": Y[j - WEEK]}
    temp, temp_fc = np.asarray(temp, dtype=float), np.asarray(temp_fc, dtype=float)
    per_time = {"temp_fc": temp_fc[j], "temp_now": temp[o], "temp_yday": temp[j - 96],
                "hour": cal.minutes[j] / 60.0, "dow": cal.times.day_of_week.to_numpy()[j],
                "daytype": np.select([cal.daytype[j] == "weekday", cal.daytype[j] == "saturday"], [0, 1], 2)}
    n = len(o)
    X = {k: v.reshape(-1) for k, v in cols.items()}
    X.update({k: np.repeat(np.asarray(v, dtype=float), K) for k, v in per_time.items()})
    for c, name in enumerate(("share_residential", "share_commercial", "share_industrial", "share_mixed")):
        X[name] = np.tile(mix[:, c], n)
    X = pd.DataFrame(X)[FEATURES].astype("float32")
    return X, np.repeat(j, K), np.tile(np.arange(K), n), Y


class ShortModel:
    """The trained model and what it needs at run time (cities/<city>/forecast/model_short.json + .meta.json)."""

    def __init__(self, city):
        import xgboost as xgb
        d = model_dir(city)
        path = d / "model_short.json"
        if not path.exists():
            raise FileNotFoundError(f"no short-horizon model at {path}: run python ml/train_short.py --city {city}")
        self.meta = json.loads((d / "model_short.meta.json").read_text(encoding="utf-8"))
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        self.version = self.meta["version"]
        self.horizon = int(self.meta["horizon_steps"])

    def tables(self, demand, history, events):
        """Point forecast and P10/P90 band for every target time of the year (DataFrames like history):
        each value is the forecast made `horizon` steps before its time."""
        txs = list(history.columns)
        scale = np.array([self.meta["scale_kw"][t] for t in txs])
        temp = demand.temperature(events).to_numpy()
        X, j, k, Y = features(history, temp, temperature_forecast(demand.spec, temp), demand.cal,
                              statics(demand, txs), scale, self.horizon)
        delta = self.model.predict(X)
        pred = np.full(Y.shape, np.nan)
        pred[j, k] = Y[j - self.horizon, k] + delta
        pred = np.clip(pred, 0, None)
        block = (demand.cal.hour * BLOCKS // 24)
        lo_q = np.array(self.meta["band"]["p10"])[block][:, None]
        hi_q = np.array(self.meta["band"]["p90"])[block][:, None]
        point = pred * scale
        lower = np.clip(pred + lo_q, 0, None) * scale
        upper = np.clip(pred + hi_q, 0, None) * scale
        idx = history.index
        return (pd.DataFrame(point, index=idx, columns=txs), pd.DataFrame(lower, index=idx, columns=txs),
                pd.DataFrame(upper, index=idx, columns=txs))
