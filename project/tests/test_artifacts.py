"""The committed data files and model agree with each other and with the code that produced them."""
import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from features import FEATURES, TX_LIST


@pytest.fixture(scope="module")
def forecast(project_root):
    return pd.read_csv(project_root / "data" / "forecast.csv", parse_dates=["timestamp"])


@pytest.fixture(scope="module")
def forecast_all(project_root):
    return pd.read_csv(project_root / "data" / "forecast_all.csv", parse_dates=["timestamp"])


@pytest.fixture(scope="module")
def scale(project_root):
    return json.loads((project_root / "data" / "scale_factors.json").read_text())


def test_forecast_contract(forecast):
    """data/forecast.csv is the interface between the ML and grid modules."""
    assert list(forecast.columns) == ["timestamp", "transformer_id", "predicted_kw", "actual_kw"]
    assert sorted(forecast["transformer_id"].unique()) == TX_LIST
    assert forecast["predicted_kw"].notna().all()
    assert (forecast["predicted_kw"] >= 0).all()
    for tx, g in forecast.groupby("transformer_id"):
        assert len(g) == 1073, tx
        assert g["timestamp"].is_unique
        assert (g["timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all(), f"{tx} has gaps"


def test_forecast_is_test_split_of_forecast_all(forecast, forecast_all):
    test = forecast_all[forecast_all["split"] == "test"].reset_index(drop=True)
    assert len(test) == len(forecast)
    np.testing.assert_allclose(forecast["predicted_kw"], test["predicted_kw"].round(3), atol=1e-9)


def test_splits_are_in_time_order(forecast_all):
    first = forecast_all.groupby("split")["timestamp"].min()
    last = forecast_all.groupby("split")["timestamp"].max()
    assert last["train"] < first["val"] and last["val"] < first["test"]


def test_scale_factors_match_training_data(forecast_all, scale):
    """scale_k maps each transformer's TRAINING peak to target_peak_kw (340 kW)."""
    train = forecast_all[forecast_all["split"] == "train"]
    for tx in TX_LIST:
        g = train[train["transformer_id"] == tx]
        assert g["actual_kw_raw"].max() * scale["scale_k"][tx] == pytest.approx(scale["target_peak_kw"])
        assert g["actual_kw_raw"].mean() == pytest.approx(scale["train_mean_raw"][tx])


def test_grid_kw_is_raw_kw_times_scale(forecast_all, scale):
    k = forecast_all["transformer_id"].map(scale["scale_k"])
    np.testing.assert_allclose(forecast_all["predicted_kw"], forecast_all["predicted_refit_raw"] * k, rtol=1e-9)
    ok = forecast_all["actual_kw_raw"].notna()
    np.testing.assert_allclose(forecast_all.loc[ok, "actual_kw"], (forecast_all["actual_kw_raw"] * k)[ok], rtol=1e-9)


def test_model_loads_with_matching_features(project_root):
    booster = xgb.Booster()
    booster.load_model(str(project_root / "ml" / "model_xgb.json"))
    assert booster.feature_names == FEATURES
    assert booster.num_features() == len(FEATURES)
