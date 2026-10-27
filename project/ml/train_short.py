"""Train the short-horizon forecast of a synthetic city (Phase 9) and compare it with the simpler forecasts.

    python ml/train_short.py --city demo

Writes
    cities/<city>/forecast/model_short.json, model_short.meta.json   the model, its scales, band and settings
    docs/forecast_short_city_<city>.csv            accuracy on the test month, every scenario, three forecasts:
        model           this model, one step ahead
        last value      the last reading, one step ahead (the simplest short-horizon forecast)
        persistence 24h the same time yesterday (the forecast of Phase 7 and 8)
    docs/feature_importance_short_city_<city>.csv

Everything is scored against the true loads (what the network will carry); the model only ever sees what the
transformer meters recorded. Columns: MAE_kW, nRMSE_% (RMSE / mean true load), bias_kW (positive = too high),
nRMSE_heatwave_% (the heatwave days), nRMSE_peak_% (the 10% highest true loads of each transformer), and
band_coverage_% (true loads inside the model's P10-P90 band; target 80%).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "city", ROOT / "ml"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from short_horizon import (BLOCKS, EARLY_STOP, FEATURES, QUANTILES, VERSION, XGB, features,  # noqa: E402
                           model_dir, settings, statics, temperature_forecast)


def histories(demand, meter_model, events):
    """(true loads, recorded loads) for a scenario's events."""
    from meters import observed_history
    truth = demand.matrix(events)
    rec = observed_history(meter_model, truth) if meter_model is not None else truth
    return truth, rec


def window_mask(times, w):
    return (times >= pd.Timestamp(w[0])) & (times < pd.Timestamp(w[1]) + pd.Timedelta(days=1))


def scores(pred, truth, mask, peak, heat, lower=None, upper=None):
    e = (pred - truth)[mask]
    t = truth[mask]
    ok = ~np.isnan(e)
    rm = lambda m: float(100 * np.sqrt(np.nanmean(e[m] ** 2)) / np.mean(t[m])) if m.any() else np.nan  # noqa: E731
    out = {"MAE_kW": float(np.nanmean(np.abs(e))), "nRMSE_%": rm(ok), "bias_kW": float(np.nanmean(e)),
           "nRMSE_heatwave_%": rm(ok & heat[mask]), "nRMSE_peak_%": rm(ok & peak[mask])}
    if lower is not None:
        inside = (truth >= lower) & (truth <= upper)
        out["band_coverage_%"] = float(100 * inside[mask][ok].mean())
    return out


def main(argv=None):
    import xgboost as xgb
    ap = argparse.ArgumentParser(description="Train a city's short-horizon forecast.")
    ap.add_argument("--city", required=True)
    a = ap.parse_args(argv)
    from core import config
    config.use_city(a.city)
    from demand import CityDemand
    from meters import MeterModel, settings as meter_settings
    demand = CityDemand(a.city)
    spec = demand.spec
    fs = settings(spec)
    h = fs["horizon_steps"]
    names = {s["name"]: s.get("events", []) for s in spec["scenarios"]}
    train_sc = fs["scenarios"] or list(names)
    meters = MeterModel(demand) if meter_settings(spec)["enabled"] else None
    txs = list(config.grid()["transformers"])
    times = demand.times
    tr_w, ca_w, te_w = window_mask(times, fs["train"]), window_mask(times, fs["calibrate"]), window_mask(times, fs["test"])
    mix = statics(demand, txs)

    recs = {}
    for sc in train_sc:
        recs[sc] = histories(demand, meters, names[sc])[1][txs]
    scale = pd.concat([r[tr_w] for r in recs.values()]).mean()                # mean recorded load per transformer

    def rows(sc, rec):
        temp = demand.temperature(names[sc]).to_numpy()
        X, j, k, Y = features(rec, temp, temperature_forecast(spec, temp), demand.cal, mix, scale.to_numpy(), h)
        y = Y[j, k] - Y[j - h, k]
        return X, j, k, y

    parts = {"train": [], "cal": []}
    for sc, rec in recs.items():
        X, j, k, y = rows(sc, rec)
        parts["train"].append((X[tr_w[j]], y[tr_w[j]]))
        parts["cal"].append((X[ca_w[j]], y[ca_w[j]], j[ca_w[j]], k[ca_w[j]], rec))
    Xtr = pd.concat([p[0] for p in parts["train"]])
    ytr = np.concatenate([p[1] for p in parts["train"]])
    Xca = pd.concat([p[0] for p in parts["cal"]])
    yca = np.concatenate([p[1] for p in parts["cal"]])
    model = xgb.XGBRegressor(**XGB, early_stopping_rounds=EARLY_STOP)
    model.fit(Xtr, ytr, eval_set=[(Xca, yca)], verbose=False)
    print(f"trained on {len(Xtr):,} rows, stopped at iteration {model.best_iteration} on {len(Xca):,} rows")

    # band: residuals on the calibration month (recorded loads), per block of the day, in scaled units
    res = yca - model.predict(Xca)
    blk = np.concatenate([demand.cal.hour[p[2]] * BLOCKS // 24 for p in parts["cal"]])
    band = {"p10": [float(np.quantile(res[blk == b], QUANTILES[0])) for b in range(BLOCKS)],
            "p90": [float(np.quantile(res[blk == b], QUANTILES[1])) for b in range(BLOCKS)]}

    d = model_dir(a.city)
    d.mkdir(exist_ok=True)
    model.save_model(d / "model_short.json")
    meta = {"version": VERSION, "kind": "xgboost, change from the last reading, all transformers",
            "horizon_steps": h, "features": FEATURES, "scale_kw": {t: round(float(scale[t]), 4) for t in txs},
            "band": band, "band_quantiles": list(QUANTILES), "band_blocks": BLOCKS,
            "train": fs["train"], "calibrate": fs["calibrate"], "test": fs["test"], "scenarios": train_sc,
            "temperature_error_c": fs["temperature_error_c"], "trained_on": "recorded loads (transformer meters)"
            if meters is not None else "true loads", "best_iteration": int(model.best_iteration),
            "xgboost": xgb.__version__, "params": {**XGB, "early_stopping_rounds": EARLY_STOP}}
    (d / "model_short.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # evaluation on the test month, every scenario, against the true loads
    from short_horizon import ShortModel
    sm = ShortModel(a.city)
    heat_ev = [e for e in spec.get("events", {}).values() if e["kind"] == "heatwave"]
    heat = np.zeros(len(times), dtype=bool)
    for e in heat_ev:
        heat |= window_mask(times, (e["from"], e["to"]))
    out = []
    for sc, ev in names.items():
        truth, rec = histories(demand, meters, ev)
        truth, rec = truth[txs], rec[txs]
        T = truth.to_numpy()
        peak = T >= np.nanquantile(T[te_w], 0.9, axis=0)
        point, lower, upper = sm.tables(demand, rec, ev)
        R = rec.to_numpy()
        last = np.full(R.shape, np.nan)
        last[h:] = R[:-h]
        day = np.full(R.shape, np.nan)
        day[96:] = R[:-96]
        heat2 = np.repeat(heat[:, None], T.shape[1], axis=1)
        m2 = np.repeat(te_w[:, None], T.shape[1], axis=1)
        for name, P, lo, hi in (("model", point.to_numpy(), lower.to_numpy(), upper.to_numpy()),
                                ("last value", last, None, None), ("persistence 24h", day, None, None)):
            s = scores(P, T, m2, peak, heat2, lo, hi)
            out.append({"scenario": sc, "forecast": name, **{k: round(v, 2) for k, v in s.items()}})
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "docs" / f"forecast_short_city_{a.city}.csv", index=False)
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    imp.rename("gain_share").round(4).to_csv(ROOT / "docs" / f"feature_importance_short_city_{a.city}.csv",
                                             index_label="feature")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
