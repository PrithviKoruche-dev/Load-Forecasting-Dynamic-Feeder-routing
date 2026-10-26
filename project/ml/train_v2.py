"""v2 forecasting models next to the published v1 model (gap-analysis items E4, E8, E10, E11).

On the same data and the same chronological split as v1 (config/forecasting.yaml), it trains and compares:
    XGBoost, v1 features (retrained)   control: on the real data this reproduces the published model
    XGBoost, v2 features               adds ramp_1h, ramp_24h, rating_kva, share_residential (E4)
    Random Forest, v2 features         comparison model (E8)
    XGBoost quantile, v2 features      P10 / P50 / P90 in one model (E11)
    same hour yesterday / last week    persistence baselines
and backtests weekly retraining on past data only (E10): before each week of the test period the v2 model is
retrained on everything known by then, and compared with the model trained once.

    python ml/train_v2.py                         # uses data/processed/clusters.parquet
    python ml/train_v2.py --clusters other.parquet

Writes: data/forecast_v2.csv, ml/models/xgb-24h-v2.json (+ .meta.json), ml/models/xgb-24h-v2-quantile.json
(+ .meta.json), docs/metrics_v2.csv, docs/metrics_v2_peak.csv, docs/metrics_v2_intervals.csv,
docs/backtest_v2.csv, docs/feature_importance_v2.csv. The published v1 files are not touched.
The model files in ml/models/ are gitignored (up to 10 MB each, and reproducible: every seed is fixed);
the tables and data/forecast_v2.csv are the results to commit.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metadata  # noqa: E402
from core import config, ids  # noqa: E402
from features import FEATURES, FEATURES_V2, build_features, split_times, train_stats  # noqa: E402


def _xgb_params():
    p = dict(config.forecasting()["model"]["xgboost"])
    return p, p.pop("early_stopping_rounds")


def fit_xgb(tr, va, trv, feats, **extra):
    """Early stopping on validation with a train-only model, then refit on train+val with that tree count
    (exactly the v1 procedure). Returns (refit model, train-only model)."""
    params, es = _xgb_params()
    params = {**params, **extra}
    m = xgb.XGBRegressor(**params, early_stopping_rounds=es, enable_categorical=True)
    m.fit(tr[feats], tr["y"], eval_set=[(va[feats], va["y"])], verbose=False)
    final = xgb.XGBRegressor(**{**params, "n_estimators": m.best_iteration + 1}, enable_categorical=True)
    final.fit(trv[feats], trv["y"], verbose=False)
    return final, m


def _rf_matrix(F, feats):
    X = F[feats].copy()
    X["tx"] = X["tx"].cat.codes
    return X


def fit_rf(trv, feats):
    rf = RandomForestRegressor(**config.forecasting()["v2"]["random_forest"], n_jobs=-1)
    rf.fit(_rf_matrix(trv, feats), trv["y"])
    return rf


def _score(g, col, actual="actual_kw"):
    e = g[col] - g[actual]
    mean = g[actual].mean()
    big = g[actual] > 0.1 * mean
    return {"MAE_kW": e.abs().mean(), "RMSE_kW": np.sqrt((e ** 2).mean()), "nRMSE_%": 100 * np.sqrt((e ** 2).mean()) / mean,
            "MAPE_%": 100 * (e[big].abs() / g.loc[big, actual]).mean(), "bias_kW": e.mean()}


def tables(test, models):
    """Accuracy per transformer and pooled, and at each transformer's top 10% load hours."""
    rows, peak = [], []
    for tx, g in list(test.groupby("transformer_id")) + [("ALL", test)]:
        for name, col in models.items():
            rows.append({"transformer": tx, "model": name, **_score(g, col)})
    for tx, g in list(test.groupby("transformer_id")) + [("ALL", None)]:
        if g is None:
            g = pd.concat(grp[grp["actual_kw"] >= grp["actual_kw"].quantile(0.9)] for _, grp in test.groupby("transformer_id"))
        else:
            g = g[g["actual_kw"] >= g["actual_kw"].quantile(0.9)]
        for name, col in models.items():
            s = _score(g, col)
            peak.append({"transformer": tx, "model": name, "hours": len(g), "MAE_kW": s["MAE_kW"], "bias_kW": s["bias_kW"]})
    return pd.DataFrame(rows), pd.DataFrame(peak)


def interval_table(test, lo_col, hi_col, lo, hi):
    from intervals import pinball
    rows = []
    for tx, g in list(test.groupby("transformer_id")) + [("ALL", test)]:
        y = g["actual_kw"]
        rows.append({"transformer": tx, "hours": len(g),
                     "coverage_%": 100 * ((y >= g[lo_col]) & (y <= g[hi_col])).mean(),
                     "above_upper_%": 100 * (y > g[hi_col]).mean(), "below_lower_%": 100 * (y < g[lo_col]).mean(),
                     "pinball_lower": pinball(y, g[lo_col], lo), "pinball_upper": pinball(y, g[hi_col], hi)})
    return pd.DataFrame(rows)


def backtest(F, t2, feats, n_estimators, every_days, scale_k, train_mean):
    """Retrain once per block of `every_days` in the test period and predict that block. The forecast for the
    first hour s of a block is issued at s - 24 h, so the retrained model may only use loads up to then."""
    params, _ = _xgb_params()
    params = {**params, "n_estimators": n_estimators}
    test = F[(F["timestamp"] >= t2) & F["y"].notna()]
    starts = pd.date_range(t2, test["timestamp"].max(), freq=f"{every_days}D")
    parts = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else test["timestamp"].max() + pd.Timedelta(hours=1)
        cutoff = s - pd.Timedelta(hours=24)                         # issue time of the block's first forecast
        past = F[(F["timestamp"] <= cutoff) & F["y"].notna()]
        assert past["timestamp"].max() <= cutoff
        m = xgb.XGBRegressor(**params, enable_categorical=True).fit(past[feats], past["y"], verbose=False)
        blk = test[(test["timestamp"] >= s) & (test["timestamp"] < e)]
        k = blk["transformer_id"].map(scale_k) * blk["transformer_id"].map(train_mean)
        parts.append(blk[["timestamp", "transformer_id"]].assign(
            retrained_kw=(m.predict(blk[feats]) * k).clip(lower=0), retrain_cutoff=cutoff))
    return pd.concat(parts, ignore_index=True), len(starts)


def run(df, out_root=ROOT, verbose=True):
    """Train and evaluate everything on a clusters frame; write the outputs under out_root. Returns a dict of
    the result tables."""
    out_root = Path(out_root)
    cfg = config.forecasting()
    v2 = cfg["v2"]
    say = print if verbose else (lambda *a, **k: None)
    t_start = time.perf_counter()

    t1, t2 = split_times(df["timestamp"])
    train_mean, train_peak = train_stats(df, t1)
    scale_k = cfg["scaling"]["target_peak_kw"] / train_peak
    F = build_features(df, train_mean, extended=True)
    F["split"] = np.where(F["timestamp"] < t1, "train", np.where(F["timestamp"] < t2, "val", "test"))
    tr, va = F[(F["split"] == "train") & F["y"].notna()], F[(F["split"] == "val") & F["y"].notna()]
    trv = F[F["split"].isin(["train", "val"]) & F["y"].notna()]
    say(f"split: train < {t1} <= val < {t2} <= test; rows {len(tr)}/{len(va)}/{(F['split'] == 'test').sum()}")

    m_v1, _ = fit_xgb(tr, va, trv, FEATURES)
    m_v2, m_v2_tr = fit_xgb(tr, va, trv, FEATURES_V2)
    say(f"XGBoost trees: v1 features {m_v1.n_estimators}, v2 features {m_v2.n_estimators}")
    rf = fit_rf(trv, FEATURES_V2)
    qs = v2["quantiles"]
    m_q, _ = fit_xgb(tr, va, trv, FEATURES_V2, objective="reg:quantileerror", quantile_alpha=np.array(qs))

    test = F[(F["split"] == "test") & F["y"].notna()].copy()
    k = test["transformer_id"].map(scale_k) * test["transformer_id"].map(train_mean)
    out = test[["timestamp", "transformer_id"]].copy()
    out["actual_kw"] = test["y"] * k
    out["xgb_v1_features_kw"] = (m_v1.predict(test[FEATURES]) * k).clip(lower=0)
    out["xgb_v2_kw"] = (m_v2.predict(test[FEATURES_V2]) * k).clip(lower=0)
    out["rf_v2_kw"] = (rf.predict(_rf_matrix(test, FEATURES_V2)) * k).clip(lower=0)
    qpred = np.sort(np.atleast_2d(m_q.predict(test[FEATURES_V2])), axis=1)      # sorted: no crossing quantiles
    qcols = [f"q{round(100 * q):02d}_kw" for q in qs]
    for j, c in enumerate(qcols):
        out[c] = (qpred[:, j] * k.to_numpy()).clip(min=0)
    out["lag24_kw"] = test["lag_24"] * k
    out["lag168_kw"] = test["lag_168"] * k
    bt, n_retrains = backtest(F, t2, FEATURES_V2, m_v2.n_estimators, v2["retrain_every_days"], scale_k, train_mean)
    out = out.merge(bt[["timestamp", "transformer_id", "retrained_kw"]], on=["timestamp", "transformer_id"], how="left")

    models = {"XGBoost v1 features (retrained)": "xgb_v1_features_kw", "XGBoost v2 features": "xgb_v2_kw",
              "Random Forest v2 features": "rf_v2_kw", f"XGBoost quantile median ({qcols[len(qs) // 2]})": qcols[len(qs) // 2],
              f"XGBoost v2, retrained every {v2['retrain_every_days']} d": "retrained_kw",
              "Same hour yesterday": "lag24_kw", "Same hour last week": "lag168_kw"}
    scored = out.dropna(subset=["actual_kw", "lag24_kw", "lag168_kw"])
    metrics, peak = tables(scored, models)
    intervals = interval_table(scored, qcols[0], qcols[-1], qs[0], qs[-1])
    bt_summary = metrics[metrics["model"].isin(["XGBoost v2 features", models_key(models, "retrained_kw")])]
    importance = pd.DataFrame({"feature": FEATURES_V2, "importance": m_v2.feature_importances_}).sort_values(
        "importance", ascending=False)

    # ------------------------------------------------------------------ write
    (out_root / "ml" / "models").mkdir(parents=True, exist_ok=True)
    (out_root / "docs").mkdir(exist_ok=True)
    (out_root / "data").mkdir(exist_ok=True)
    num = out.select_dtypes("number").columns
    ids.add_transformer_ids(out.assign(**{c: out[c].round(4) for c in num})).to_csv(
        out_root / "data" / "forecast_v2.csv", index=False)
    windows = metadata.windows_from(F.assign(timestamp=F["timestamp"]), split_col="split")
    params, es = _xgb_params()
    pooled = metrics[metrics["transformer"] == "ALL"].set_index("model")
    for model, path, kind, extra_params, key in (
            (m_v2, out_root / "ml" / "models" / f"{v2['model_version']}.json", "xgboost-24h-direct", {}, "XGBoost v2 features"),
            (m_q, out_root / "ml" / "models" / f"{v2['model_version']}-quantile.json", "xgboost-24h-quantile",
             {"objective": "reg:quantileerror", "quantile_alpha": qs}, None)):
        model.save_model(str(path))
        m = {"test_pooled": pooled.loc[key].to_dict()} if key else {"test_intervals": intervals.set_index("transformer").loc["ALL"].to_dict()}
        meta = metadata.build(path.stem, kind, path, FEATURES_V2,
                              {**params, "early_stopping_rounds": es, "n_estimators_used": int(model.n_estimators), **extra_params},
                              windows, m, horizon_h=24, quantiles=qs if key is None else None,
                              train_mean_raw=train_mean.to_dict(), scale_k=scale_k.to_dict())
        metadata.write(meta, path)
    metrics.round(3).to_csv(out_root / "docs" / "metrics_v2.csv", index=False)
    peak.round(3).to_csv(out_root / "docs" / "metrics_v2_peak.csv", index=False)
    intervals.round(3).to_csv(out_root / "docs" / "metrics_v2_intervals.csv", index=False)
    bt_summary.round(3).assign(retrains=n_retrains).to_csv(out_root / "docs" / "backtest_v2.csv", index=False)
    importance.round(4).to_csv(out_root / "docs" / "feature_importance_v2.csv", index=False)

    say("\nTest period, pooled (grid kW):")
    say(pooled[["MAE_kW", "RMSE_kW", "nRMSE_%", "bias_kW"]].round(2).to_string())
    say(f"\n{qcols[0]}-{qcols[-1]} coverage (target {round(100 * (qs[-1] - qs[0]))}%): "
        f"{intervals.set_index('transformer').loc['ALL', 'coverage_%']:.1f}%")
    say(f"Backtest: {n_retrains} retrains. Done in {time.perf_counter() - t_start:.0f} s.")
    return {"metrics": metrics, "peak": peak, "intervals": intervals, "forecast": out, "importance": importance,
            "retrains": n_retrains}


def models_key(models, col):
    return next(k for k, v in models.items() if v == col)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clusters", default=str(ROOT / "data" / "processed" / "clusters.parquet"))
    args = ap.parse_args()
    path = Path(args.clusters)
    if not path.exists():
        raise SystemExit(f"{path} not found: run ml/prepare_data.py first (it needs the eMARC files in data/raw/).")
    run(pd.read_parquet(path))
    print("\nSaved data/forecast_v2.csv, ml/models/, docs/metrics_v2*.csv, docs/backtest_v2.csv, "
          "docs/feature_importance_v2.csv")


if __name__ == "__main__":
    main()
