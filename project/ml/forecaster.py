"""Step-by-step forecasting (gap-analysis item E10): forecast_at(t) with only the information available when
the forecast is issued, 24 hours before t.

    f = Forecaster()                                   # the published model and its scale factors
    f.forecast_at("2020-02-29 21:00", history)         # -> one row per transformer

`history` has the clusters.parquet columns: timestamp, transformer_id, load_kw_raw, temp_c. Every load value
later than t - 24 h is blanked before the features are built, so the forecast cannot use it even by accident;
the temperature at t is used, as in training (it stands for the weather forecast).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metadata  # noqa: E402
from features import FEATURES, build_features  # noqa: E402

HORIZON = pd.Timedelta(hours=24)
LOOKBACK = pd.Timedelta(hours=168 + 168 + 48)    # enough history for every lag and rolling window


class Forecaster:
    def __init__(self, model_path=ROOT / "ml" / "model_xgb.json", scale_path=ROOT / "data" / "scale_factors.json"):
        self.model_path = Path(model_path)
        self.meta = metadata.read(self.model_path)           # refuses a model that does not match its metadata
        if self.meta["features"] != FEATURES:
            raise ValueError("the model's feature list differs from ml/features.py")
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(self.model_path))
        sc = json.loads(Path(scale_path).read_text())
        self.train_mean = pd.Series(sc["train_mean_raw"])
        self.scale_k = pd.Series(sc["scale_k"])
        self.version = self.meta["version"]

    def predict_features(self, F):
        """Batch prediction from a build_features() frame: raw cluster kW and grid kW."""
        raw = (self.model.predict(F[FEATURES]) * F["transformer_id"].map(self.train_mean).to_numpy()).clip(min=0)
        return pd.DataFrame({"timestamp": F["timestamp"].to_numpy(), "transformer_id": F["transformer_id"].to_numpy(),
                             "predicted_kw_raw": raw,
                             "predicted_kw": raw * F["transformer_id"].map(self.scale_k).to_numpy()})

    def forecast_at(self, target_time, history):
        """24 h-ahead forecast for target_time, issued at target_time - 24 h."""
        t = pd.Timestamp(target_time)
        issued = t - HORIZON
        h = history[(history["timestamp"] > t - LOOKBACK) & (history["timestamp"] <= t)].copy()
        if h.empty or not (h["timestamp"] == t).any():
            raise ValueError(f"history has no row for {t} (its temperature is needed)")
        h.loc[h["timestamp"] > issued, "load_kw_raw"] = np.nan      # not yet known when the forecast is issued
        F = build_features(h, self.train_mean)
        F = F[F["timestamp"] == t]
        return self.predict_features(F).assign(issued_at=issued, model_version=self.version).reset_index(drop=True)


def forecasts_for(history, targets, forecaster=None):
    """forecast_at for many target times (one row per transformer and target)."""
    f = forecaster or Forecaster()
    return pd.concat([f.forecast_at(t, history) for t in targets], ignore_index=True)
