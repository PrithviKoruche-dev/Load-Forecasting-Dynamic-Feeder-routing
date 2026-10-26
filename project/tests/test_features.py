"""Feature engineering must not leak information from less than 24 hours before the target hour."""
import numpy as np
import pandas as pd
import pytest

from features import FEATURES, LAGS, build_features, split_times, train_stats

LOAD_FEATURES = [f for f in FEATURES if f.startswith(("lag_", "roll_"))]


def synthetic_clusters(hours=24 * 30, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2019-06-01", periods=hours, freq="1h")
    rows = []
    for tx in ("T1", "T2"):
        daily = 3 + np.sin(2 * np.pi * ts.hour / 24)
        rows.append(pd.DataFrame({"timestamp": ts, "transformer_id": tx, "region": "x", "n_households": 10,
                                  "load_kw_raw": daily + rng.normal(0, 0.3, hours),
                                  "temp_c": 25 + 5 * np.sin(2 * np.pi * ts.hour / 24)}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def clusters():
    return synthetic_clusters()


def features_for(df):
    mean, _ = train_stats(df, df["timestamp"].max())
    return build_features(df, mean)


def test_lags_point_to_the_right_hour(clusters):
    F = features_for(clusters)
    g = F[F["transformer_id"] == "T1"].set_index("timestamp")
    t = g.index[400]
    for L in LAGS:
        assert g.loc[t, f"lag_{L}"] == pytest.approx(g.loc[t - pd.Timedelta(hours=L), "y"])


def test_no_feature_sees_the_last_23_hours(clusters):
    """Change every load value from t-23 h onwards: no load-based feature at t may change."""
    t = clusters["timestamp"].sort_values().unique()[500]
    F0 = features_for(clusters)

    tampered = clusters.copy()
    recent = tampered["timestamp"] > t - pd.Timedelta(hours=24)
    tampered.loc[recent, "load_kw_raw"] *= 10
    # keep the normalising mean identical so only the leakage path is tested
    mean, _ = train_stats(clusters, clusters["timestamp"].max())
    F1 = build_features(tampered, mean)

    row0 = F0[F0["timestamp"] == t].set_index("transformer_id")[LOAD_FEATURES]
    row1 = F1[F1["timestamp"] == t].set_index("transformer_id")[LOAD_FEATURES]
    pd.testing.assert_frame_equal(row0, row1)


def test_feature_list_is_complete(clusters):
    F = features_for(clusters)
    assert set(FEATURES) <= set(F.columns)
    assert len(FEATURES) == 14


def test_split_is_chronological(clusters):
    t1, t2 = split_times(clusters["timestamp"])
    ts = np.sort(clusters["timestamp"].unique())
    assert ts[0] < t1 < t2 < ts[-1]
    share_train = (ts < t1).mean()
    share_val = ((ts >= t1) & (ts < t2)).mean()
    assert share_train == pytest.approx(0.70, abs=0.01)
    assert share_val == pytest.approx(0.15, abs=0.01)
