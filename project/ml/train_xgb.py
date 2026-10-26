"""Train one XGBoost model for all transformers; write forecast files.

Model settings, the split and the grid scaling target come from config/forecasting.yaml.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project/ -> import core
from features import FEATURES, split_times, train_stats, build_features  # noqa: E402
from core import config, ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = config.forecasting()
TARGET_PEAK_KW = float(CFG["scaling"]["target_peak_kw"])   # training peak -> about 73% of a 500 kVA transformer
XGB = dict(CFG["model"]["xgboost"])
EARLY_STOP = XGB.pop("early_stopping_rounds")

df = pd.read_parquet(ROOT / "data" / "processed" / "clusters.parquet")
t1, t2 = split_times(df["timestamp"])
print(f"train: < {t1}   val: < {t2}   test: >= {t2}")

train_mean, train_peak = train_stats(df, t1)
F = build_features(df, train_mean)
F["split"] = np.where(F["timestamp"] < t1, "train", np.where(F["timestamp"] < t2, "val", "test"))
for s in ("train", "val", "test"):
    sub = F[F["split"] == s]
    print(f"{s:5s}: {len(sub):6d} rows, missing target hours: {int(sub['y'].isna().sum())}")

tr = F[(F["split"] == "train") & F["y"].notna()]
va = F[(F["split"] == "val") & F["y"].notna()]
model = xgb.XGBRegressor(**XGB, early_stopping_rounds=EARLY_STOP, enable_categorical=True)
model.fit(tr[FEATURES], tr["y"], eval_set=[(va[FEATURES], va["y"])], verbose=False)
print("best iteration:", model.best_iteration)

# Final model: refit on train+val with the iteration count fixed by early stopping
trv = F[F["split"].isin(["train", "val"]) & F["y"].notna()]
final = xgb.XGBRegressor(**{**XGB, "n_estimators": model.best_iteration + 1}, enable_categorical=True)
final.fit(trv[FEATURES], trv["y"], verbose=False)

mean_map = F["transformer_id"].map(train_mean)
F["predicted_kw_raw"] = (model.predict(F[FEATURES]) * mean_map).clip(lower=0)          # train-only model
F["predicted_refit_raw"] = (final.predict(F[FEATURES]) * mean_map).clip(lower=0)       # train+val refit (FINAL)
F["lag24_raw"] = F["lag_24"] * mean_map                    # baseline: same hour yesterday
F["lag168_raw"] = F["lag_168"] * mean_map                  # baseline: same hour last week

k = (TARGET_PEAK_KW / train_peak).to_dict()                # per-transformer scale to grid kW
F["scale_k"] = F["transformer_id"].map(k)
F["predicted_kw"] = F["predicted_refit_raw"] * F["scale_k"]          # dashboard uses the final model
F["actual_kw"] = F["actual_raw"] * F["scale_k"]
F = F.rename(columns={"actual_raw": "actual_kw_raw"})

cols = ["timestamp", "transformer_id", "split", "predicted_kw_raw", "actual_kw_raw",
        "predicted_refit_raw", "lag24_raw", "lag168_raw", "scale_k", "predicted_kw", "actual_kw"]
ids.add_transformer_ids(F[cols]).to_csv(ROOT / "data" / "forecast_all.csv", index=False)
test = F[F["split"] == "test"][["timestamp", "transformer_id", "predicted_kw", "actual_kw"]]
ids.add_transformer_ids(test.assign(predicted_kw=test["predicted_kw"].round(3), actual_kw=test["actual_kw"].round(3))).to_csv(
    ROOT / "data" / "forecast.csv", index=False)
final.save_model(str(ROOT / "ml" / "model_xgb.json"))
json.dump({"target_peak_kw": TARGET_PEAK_KW, "scale_k": k, "train_mean_raw": train_mean.to_dict()},
          open(ROOT / "data" / "scale_factors.json", "w"), indent=2)

imp = pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\ntop features:\n", imp.head(6).round(3).to_string())
print(f"\nSaved forecast.csv ({len(test)} test rows) and forecast_all.csv")
print("Peak of scaled TEST forecast per transformer (kW, 500 kVA ~ 475 kW):")
print(test.groupby("transformer_id")["predicted_kw"].max().round(0).to_string())