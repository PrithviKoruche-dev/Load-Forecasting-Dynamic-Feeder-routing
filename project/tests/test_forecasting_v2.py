"""Forecasting v2 (Phase 4): model metadata (E5), peak-hour and overload metrics (E9), intervals (E11),
step-wise forecasting and pending evaluation (E10), v2 features / Random Forest / quantile model / backtest
(E4, E8) and cold start (E6)."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import evaluate
import intervals as iv
import metadata
from features import FEATURES, FEATURES_V2, build_features, train_stats
from forecaster import Forecaster
from storage import Store
from synthetic import make_clusters


def roundtrip(df):
    import io
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


@pytest.fixture(scope="module")
def fa(project_root):
    return pd.read_csv(project_root / "data" / "forecast_all.csv", parse_dates=["timestamp"])


# ------------------------------------------------------------------ E5 metadata
def test_v1_metadata_describes_the_committed_model(project_root, fa):
    meta = metadata.read(project_root / "ml" / "model_xgb.json")           # verifies the SHA-256
    assert meta["version"] == "xgb-24h-v1" and meta["features"] == FEATURES
    assert meta["params"]["n_estimators_used"] == 121
    assert meta["windows"] == metadata.windows_from(fa)
    assert meta["metrics"]["test"]["pooled"]["nRMSE_%"] == 25.33


def test_metadata_refuses_a_different_model_file(project_root, tmp_path):
    model = tmp_path / "model_xgb.json"
    shutil.copy(project_root / "ml" / "model_xgb.json", model)
    shutil.copy(project_root / "ml" / "model_xgb.meta.json", tmp_path / "model_xgb.meta.json")
    assert metadata.read(model)["version"] == "xgb-24h-v1"
    model.write_text(model.read_text().replace('"num_trees":"121"', '"num_trees":"120"'))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        metadata.read(model)


def test_model_is_registered(tmp_path, project_root):
    store = Store(f"sqlite:///{tmp_path / 'r.db'}")
    metadata.register(store, project_root / "ml" / "model_xgb.json")
    reg = store.models()
    assert reg["version"].tolist() == ["xgb-24h-v1"] and reg.iloc[0]["features"] == FEATURES


# ------------------------------------------------------------------ E9 peak hours and overload classification
def test_peak_hour_metrics_match_docs(project_root):
    got = roundtrip(evaluate.peak_table(evaluate.load_test_split()).round(2))
    pd.testing.assert_frame_equal(got, pd.read_csv(project_root / "docs" / "metrics_peak.csv"))


def test_every_model_under_forecasts_the_peaks(project_root):
    peak = pd.read_csv(project_root / "docs" / "metrics_peak.csv")
    assert (peak[peak["transformer"] == "ALL"]["bias_kW"] < 0).all()


def test_overload_scores():
    from overload_skill import scores
    d = pd.DataFrame({"actual": [True, True, False, False], "point": [True, False, True, False]})
    s = scores(d, "point")
    assert (s["true_pos"], s["false_pos"], s["missed"]) == (1, 1, 1)
    assert s["precision_%"] == 50.0 and s["recall_%"] == 50.0
    none = scores(pd.DataFrame({"actual": [False], "point": [False]}), "point")
    assert none["precision_%"] is None and none["recall_%"] is None


@pytest.mark.slow
def test_overload_metrics_match_docs(project_root):
    from overload_skill import run
    got = roundtrip(run())
    pd.testing.assert_frame_equal(got, pd.read_csv(project_root / "docs" / "metrics_overload.csv"))


def test_p90_catches_far_more_real_overloads(project_root):
    m = pd.read_csv(project_root / "docs" / "metrics_overload.csv")
    for sc in ("S2", "S3"):
        rows = m[m["scenario"].str.startswith(sc)].set_index("flag_on")["recall_%"]
        assert rows["point forecast"] < 20 and rows["upper bound (P90)"] > 80


# ------------------------------------------------------------------ E11 intervals
def test_intervals_match_docs(project_root):
    _, test, cov = iv.build()
    pd.testing.assert_frame_equal(roundtrip(cov.round(3)), pd.read_csv(project_root / "docs" / "metrics_intervals.csv"))
    committed = pd.read_csv(project_root / "data" / "forecast_intervals.csv")
    np.testing.assert_allclose(committed["p90_kw"], test["p_upper_kw"].round(3), atol=1e-9)
    assert (committed["p10_kw"] <= committed["p90_kw"]).all()


def test_rolling_bands_use_only_actuals_known_at_issue_time(fa):
    """Change every actual value from t - 24 h onwards: the band for t must not move."""
    t = pd.Timestamp("2020-03-01 21:00")
    rows = fa[(fa["split"] == "test") & (fa["timestamp"] == t)][["timestamp", "transformer_id", "predicted_kw", "actual_kw"]]
    tampered = fa.copy()
    tampered.loc[tampered["timestamp"] > t - pd.Timedelta(hours=24), "actual_kw_raw"] *= 3
    a = iv.apply_rolling(fa, rows)[["p_lower_kw", "p_upper_kw"]].to_numpy()
    b = iv.apply_rolling(tampered, rows)[["p_lower_kw", "p_upper_kw"]].to_numpy()
    np.testing.assert_array_equal(a, b)
    tampered.loc[tampered["timestamp"] <= t - pd.Timedelta(hours=24), "actual_kw_raw"] *= 3
    assert not np.array_equal(a, iv.apply_rolling(tampered, rows)[["p_lower_kw", "p_upper_kw"]].to_numpy())


def test_rolling_beats_static_and_is_near_target(project_root):
    cov = pd.read_csv(project_root / "docs" / "metrics_intervals.csv")
    pooled = cov[cov["transformer"] == "ALL"].set_index("method")["coverage_%"]
    assert pooled["rolling 28 d"] > pooled["static"]
    assert 75 <= pooled["rolling 28 d"] <= 85


# ------------------------------------------------------------------ E10 pending evaluation in the database
def test_forecasts_are_scored_as_their_hour_passes(tmp_path, project_root):
    s = Store(f"sqlite:///{tmp_path / 'f.db'}")
    s.import_forecast_csv(project_root / "data" / "forecast.csv")
    assert s.import_intervals_csv(project_root / "data" / "forecast_intervals.csv") == 2 * 6438
    assert s.import_intervals_csv(project_root / "data" / "forecast_intervals.csv") == 0
    frame = s.forecast_frame()
    assert {"p10_kw", "p90_kw"} <= set(frame.columns) and frame["p90_kw"].notna().all()
    assert s.forecast_performance().set_index("transformer").loc["ALL", "pending"] == 6438
    assert s.score_forecasts("2020-02-15 00:00") == 6 * 138
    assert s.score_forecasts("2020-02-15 00:00") == 0                         # never scored twice
    early = s.forecast_performance(until="2020-02-15 00:00").set_index("transformer")
    assert early.loc["ALL", "scored"] == 828 and early.loc["ALL", "pending"] == 0
    s.score_forecasts("2020-03-24 23:00")
    final = s.forecast_performance().set_index("transformer")
    published = pd.read_csv(project_root / "docs" / "metrics.csv")
    published = published[published["model"] == "XGBoost (train+val refit)"].set_index("transformer")["nRMSE_%"]
    for tx in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        assert final.loc[tx, "nRMSE_%"] == pytest.approx(published[tx], abs=0.01)


# ------------------------------------------------------------------ E10 forecast_at(t)
@pytest.fixture(scope="module")
def history():
    return make_clusters(days=60)


def test_forecast_at_equals_batch_prediction(history):
    f = Forecaster()
    F = build_features(history, f.train_mean)
    batch = f.predict_features(F)
    for t in pd.to_datetime(["2019-06-20 19:00", "2019-07-25 07:00"]):
        one = f.forecast_at(t, history).set_index("transformer_id")["predicted_kw"]
        ref = batch[batch["timestamp"] == t].set_index("transformer_id")["predicted_kw"]
        np.testing.assert_allclose(one.sort_index(), ref.sort_index(), rtol=0, atol=0)


def test_forecast_at_cannot_see_the_future(history):
    f = Forecaster()
    t = pd.Timestamp("2019-07-10 20:00")
    tampered = history.copy()
    tampered.loc[tampered["timestamp"] > t - pd.Timedelta(hours=24), "load_kw_raw"] *= 5
    out = f.forecast_at(t, tampered)
    assert (out["issued_at"] == t - pd.Timedelta(hours=24)).all() and (out["model_version"] == "xgb-24h-v1").all()
    np.testing.assert_array_equal(out["predicted_kw"], f.forecast_at(t, history)["predicted_kw"])
    with pytest.raises(ValueError, match="no row for"):
        f.forecast_at("2030-01-01 00:00", history)


def test_forecast_at_hides_loads_after_the_issue_time(history, monkeypatch):
    """Second guard: the features are built from a history whose loads after t - 24 h are blanked, so even a
    future feature with a shorter lag could not see them."""
    import forecaster
    seen = {}
    real = forecaster.build_features

    def spy(h, mean):
        seen["h"] = h.copy()
        return real(h, mean)
    monkeypatch.setattr(forecaster, "build_features", spy)
    monkeypatch.setattr(forecaster.Forecaster, "predict_features", lambda self, F: pd.DataFrame({"x": [0]}))
    t = pd.Timestamp("2019-07-10 20:00")
    Forecaster().forecast_at(t, history)
    h = seen["h"]
    assert h["timestamp"].max() == t                                       # temperature up to t is allowed
    assert h.loc[h["timestamp"] > t - pd.Timedelta(hours=24), "load_kw_raw"].isna().all()
    assert h.loc[h["timestamp"] <= t - pd.Timedelta(hours=24), "load_kw_raw"].notna().all()


# ------------------------------------------------------------------ E4 features
def test_v2_features_do_not_leak():
    h = make_clusters(days=30)
    mean, _ = train_stats(h, h["timestamp"].max())
    t = h["timestamp"].sort_values().unique()[500]
    F0 = build_features(h, mean, extended=True)
    tampered = h.copy()
    tampered.loc[tampered["timestamp"] > t - pd.Timedelta(hours=24), "load_kw_raw"] *= 10
    F1 = build_features(tampered, mean, extended=True)
    cols = [c for c in FEATURES_V2 if c.startswith(("lag_", "roll_", "ramp_"))]
    pd.testing.assert_frame_equal(F0[F0["timestamp"] == t][cols].reset_index(drop=True),
                                  F1[F1["timestamp"] == t][cols].reset_index(drop=True))
    assert FEATURES_V2[:len(FEATURES)] == FEATURES                             # v1 contract unchanged
    assert set(build_features(h, mean).columns).isdisjoint({"ramp_1h", "ramp_24h"})


# ------------------------------------------------------------------ E4 / E8 / E11 / E10 v2 training
@pytest.mark.slow
def test_train_v2_end_to_end(tmp_path):
    import train_v2
    r = train_v2.run(make_clusters(days=150), out_root=tmp_path, verbose=False)
    pooled = r["metrics"][r["metrics"]["transformer"] == "ALL"].set_index("model")
    assert {"XGBoost v1 features (retrained)", "XGBoost v2 features", "Random Forest v2 features",
            "Same hour yesterday"} <= set(pooled.index)
    assert pooled.loc["XGBoost v2 features", "nRMSE_%"] < pooled.loc["Same hour yesterday", "nRMSE_%"]
    cov = r["intervals"].set_index("transformer").loc["ALL", "coverage_%"]
    assert 65 <= cov <= 95
    test_days = (r["forecast"]["timestamp"].max() - r["forecast"]["timestamp"].min()) / pd.Timedelta(days=1)
    assert r["retrains"] == int(test_days // 7) + 1
    for name in ("xgb-24h-v2", "xgb-24h-v2-quantile"):
        meta = metadata.read(tmp_path / "ml" / "models" / f"{name}.json")
        assert meta["features"] == FEATURES_V2
    fc = pd.read_csv(tmp_path / "data" / "forecast_v2.csv")
    assert (fc["q10_kw"] <= fc["q50_kw"]).all() and (fc["q50_kw"] <= fc["q90_kw"]).all()  # no crossing
    for f in ("metrics_v2.csv", "metrics_v2_peak.csv", "metrics_v2_intervals.csv", "backtest_v2.csv",
              "feature_importance_v2.csv"):
        assert (tmp_path / "docs" / f).exists()


def test_backtest_retrains_only_on_loads_known_at_issue_time():
    """Each weekly retrain may use labels up to the issue time (t - 24 h) of its block's first forecast; a
    change to any load after that cutoff must leave the block's forecasts unchanged."""
    import train_v2
    h = make_clusters(days=60)
    mean, _ = train_stats(h, h["timestamp"].max())
    t2 = pd.Timestamp("2019-07-20 00:00")
    F = build_features(h, mean, extended=True)
    args = (FEATURES_V2, 20, 7, {t: 1.0 for t in mean.index}, mean)
    bt, n = train_v2.backtest(F, t2, *args)
    assert n == 2 and (bt.groupby("retrain_cutoff")["timestamp"].min() - bt["retrain_cutoff"].unique()
                       == pd.Timedelta(hours=24)).all()
    F2 = F.copy()                       # labels the first retrain must not see: after its cutoff, before t2
    F2.loc[(F2["timestamp"] > t2 - pd.Timedelta(hours=24)) & (F2["timestamp"] < t2), "y"] *= 4
    first_week = bt["timestamp"] < t2 + pd.Timedelta(days=7)
    bt2, _ = train_v2.backtest(F2, t2, *args)
    np.testing.assert_array_equal(bt.loc[first_week, "retrained_kw"], bt2.loc[first_week, "retrained_kw"])
    assert not np.array_equal(bt.loc[~first_week, "retrained_kw"], bt2.loc[~first_week, "retrained_kw"])


# ------------------------------------------------------------------ E6 cold start
def test_cold_start_status_and_missing_data(tmp_path, project_root):
    import cold_start
    st = cold_start.status(project_root)
    assert st["model"] and st["forecasts"] and st["metadata_ok"]
    empty = tmp_path / "p"
    (empty / "ml").mkdir(parents=True)
    shutil.copy(project_root / "ml" / "metadata.py", empty / "ml")
    with pytest.raises(SystemExit, match="eMARC"):
        cold_start.ensure(empty)


@pytest.mark.slow
def test_cold_start_trains_from_clusters(tmp_path, project_root):
    proj = tmp_path / "project"
    shutil.copytree(project_root, proj, ignore=shutil.ignore_patterns("tests", "logs", "*.db*", "__pycache__",
                                                                      ".pytest_cache"))
    for f in ("ml/model_xgb.json", "ml/model_xgb.meta.json", "data/forecast.csv", "data/forecast_all.csv"):
        (proj / f).unlink()
    (proj / "data" / "processed").mkdir(parents=True, exist_ok=True)
    make_clusters(days=150).to_parquet(proj / "data" / "processed" / "clusters.parquet", index=False)
    out = subprocess.run([sys.executable, str(proj / "ml" / "cold_start.py")], capture_output=True, text=True, cwd=proj)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "train_xgb.py" in out.stdout and "intervals.py" in out.stdout and "prepare_data.py" not in out.stdout
    meta = json.loads((proj / "ml" / "model_xgb.meta.json").read_text())
    assert meta["version"] == "xgb-24h-v1"
    again = subprocess.run([sys.executable, str(proj / "ml" / "cold_start.py")], capture_output=True, text=True, cwd=proj)
    assert "nothing to do" in again.stdout
