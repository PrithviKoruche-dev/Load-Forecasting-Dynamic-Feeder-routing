"""Phase 9: the short-horizon forecast of the synthetic city - no leakage, reproduces its published accuracy,
beats the simpler forecasts, and runs in the closed loop."""
import copy
import sys

import numpy as np
import pandas as pd
import pytest

from core import config

C1, C4 = "C1 Normal year", "C4 Heatwave and EV adoption"
EVENTS_C4 = ("may_heatwave", "ev_adoption")


@pytest.fixture
def city(project_root, monkeypatch):
    monkeypatch.setenv("LFDR_CONFIG_DIR", str(project_root / "cities" / "demo" / "config"))
    config.reload()
    yield project_root / "cities" / "demo"
    monkeypatch.delenv("LFDR_CONFIG_DIR")
    config.reload()


@pytest.fixture(scope="module")
def parts(project_root):
    sys.path.insert(0, str(project_root / "ml"))
    from demand import CityDemand
    from meters import MeterModel, observed_history
    from short_horizon import ShortModel
    d = CityDemand("demo")
    m = MeterModel(d)
    txs = list(d.transformers)
    truth = d.matrix(EVENTS_C4)[txs]
    rec = observed_history(m, truth)[txs]
    return d, ShortModel("demo"), truth, rec


@pytest.fixture
def city_store(city, tmp_path, monkeypatch):
    url = f"sqlite:///{(tmp_path / 'city.db').as_posix()}"
    monkeypatch.setenv("LFDR_DB_URL", url)
    from orchestrator import prepare_database
    from storage import Store
    s = Store(url)
    prepare_database(s)
    return s


def test_forecast_settings_are_checked(parts):
    from short_horizon import settings
    d = parts[0]
    assert settings(d.spec)["horizon_steps"] == 1
    for k, v, msg in (("horizon_steps", 0, "horizon_steps"), ("temperature_error_c", -1, "temperature_error_c"),
                      ("typo", 1, "unknown")):
        spec = copy.deepcopy(d.spec)
        spec["forecast"][k] = v
        with pytest.raises(ValueError, match=msg):
            settings(spec)


def test_the_forecast_for_a_step_uses_nothing_recorded_after_the_step_before(parts):
    d, sm, truth, rec = parts
    t = pd.Timestamp("2023-05-10 19:00")
    point, lower, upper = sm.tables(d, rec, EVENTS_C4)
    later = rec.copy()
    later.loc[later.index >= t] *= 3                                         # everything recorded from t on
    p2, l2, u2 = sm.tables(d, later, EVENTS_C4)
    assert np.allclose(point.loc[t], p2.loc[t]) and np.allclose(upper.loc[t], u2.loc[t])
    nxt = t + pd.Timedelta(minutes=15)
    assert not np.allclose(point.loc[nxt], p2.loc[nxt])                      # the next one does see the reading at t
    assert (lower.loc[t] <= point.loc[t]).all() and (point.loc[t] <= upper.loc[t]).all()


def test_the_weather_forecast_is_the_same_in_every_scenario(parts):
    from short_horizon import temperature_forecast
    d = parts[0]
    a = temperature_forecast(d.spec, d.temperature(()).to_numpy())
    b = temperature_forecast(d.spec, d.temperature(("may_heatwave",)).to_numpy())
    diff = b - a
    heat = d.spec["events"]["may_heatwave"]["extra_c"]
    assert set(np.round(np.unique(diff), 6)) <= {0.0, heat}                   # only the heatwave differs
    err = a - d.temperature(()).to_numpy()
    assert err.std() == pytest.approx(d.spec["forecast"]["temperature_error_c"], rel=0.05)


def test_the_committed_model_reproduces_its_published_accuracy(parts, project_root):
    """docs/forecast_short_city_demo.csv, C4 row of the model: recomputed from the committed model."""
    import train_short
    d, sm, truth, rec = parts
    times = d.times
    te = train_short.window_mask(times, d.spec["forecast"]["test"])
    T = truth.to_numpy()
    peak = T >= np.nanquantile(T[te], 0.9, axis=0)
    heat = np.repeat(train_short.window_mask(times, (d.spec["events"]["may_heatwave"]["from"],
                                                      d.spec["events"]["may_heatwave"]["to"]))[:, None], T.shape[1], 1)
    point, lower, upper = sm.tables(d, rec, EVENTS_C4)
    s = train_short.scores(point.to_numpy(), T, np.repeat(te[:, None], T.shape[1], 1), peak, heat,
                           lower.to_numpy(), upper.to_numpy())
    doc = pd.read_csv(project_root / "docs" / "forecast_short_city_demo.csv")
    row = doc[(doc["scenario"] == C4) & (doc["forecast"] == "model")].iloc[0]
    for k in ("MAE_kW", "nRMSE_%", "bias_kW", "nRMSE_heatwave_%", "nRMSE_peak_%", "band_coverage_%"):
        assert s[k] == pytest.approx(row[k], abs=0.006), k


def test_the_model_beats_the_simpler_forecasts_in_every_scenario(project_root):
    doc = pd.read_csv(project_root / "docs" / "forecast_short_city_demo.csv").set_index(["scenario", "forecast"])
    for sc in doc.index.get_level_values(0).unique():
        m, last, day = (doc.loc[(sc, f)] for f in ("model", "last value", "persistence 24h"))
        assert m["nRMSE_%"] < 0.8 * last["nRMSE_%"] < 0.3 * day["nRMSE_%"]
        assert m["nRMSE_heatwave_%"] < last["nRMSE_heatwave_%"] and m["nRMSE_peak_%"] <= last["nRMSE_peak_%"]
        assert 70 <= m["band_coverage_%"] <= 90


def test_the_city_uses_the_short_forecast_and_stores_it(city_store):
    from orchestrator import Orchestrator
    import engine as e
    o = Orchestrator(city_store, engine=e.GridEngine(), scenario=C4)
    assert o.forecasts.name == "short" and o.forecasts.model_version == city_store.import_cfg["model_version"]
    t = pd.Timestamp("2023-05-10 18:00")
    o.step(t, run_id=1)
    o.step(t + pd.Timedelta(minutes=15), run_id=1)
    with city_store.engine.connect() as c:
        f = pd.read_sql("select target_time, horizon_h, status from forecasts", c)
        q = pd.read_sql("select distinct quantile from forecast_quantiles", c)
    n = len(config.grid()["transformers"])
    assert len(f) == 2 * n and set(f["horizon_h"]) == {0.25}
    assert pd.to_datetime(f["target_time"]).min() == t + pd.Timedelta(minutes=15)
    m = city_store.measurements("dt-scada", sim_from=t + pd.Timedelta(minutes=15), sim_to=t + pd.Timedelta(minutes=15))
    assert (f["status"] == "scored").sum() == m["kw"].notna().sum() >= n - 3   # scored where its meter reported
    assert sorted(q["quantile"]) == [0.1, 0.9]


def test_the_short_forecast_is_refused_for_the_classic_network(tmp_path, monkeypatch, project_root):
    import shutil
    import yaml
    d = tmp_path / "config"
    shutil.copytree(project_root / "config", d)
    sim = yaml.safe_load((d / "simulation.yaml").read_text())
    sim["orchestrator"]["forecast_source"] = "short"
    (d / "simulation.yaml").write_text(yaml.safe_dump(sim))
    monkeypatch.setenv("LFDR_CONFIG_DIR", str(d))
    config.reload()
    try:
        with pytest.raises(config.ConfigError, match="persistence or short is for a synthetic city"):
            config.simulation()
    finally:
        monkeypatch.delenv("LFDR_CONFIG_DIR")
        config.reload()


def test_a_missing_model_is_explained(tmp_path):
    from short_horizon import ShortModel
    (tmp_path / "city.yaml").write_text("seed: 1\n")
    with pytest.raises(FileNotFoundError, match="train_short.py"):
        ShortModel(tmp_path)


@pytest.mark.slow
def test_training_again_gives_the_published_accuracy(project_root, tmp_path, monkeypatch):
    monkeypatch.setenv("LFDR_CONFIG_DIR", str(project_root / "config"))
    """Retraining (same data, same settings) reproduces docs/forecast_short_city_demo.csv closely. XGBoost is
    deterministic on one machine; other machines can differ slightly (as for the eMARC model, Phase 4.1)."""
    import shutil
    import train_short
    doc = pd.read_csv(project_root / "docs" / "forecast_short_city_demo.csv")
    src = project_root / "cities" / "demo" / "forecast"
    keep = tmp_path / "keep"
    shutil.copytree(src, keep)
    docs_keep = {f: (project_root / "docs" / f).read_bytes() for f in ("forecast_short_city_demo.csv",
                                                                         "feature_importance_short_city_demo.csv")}
    try:
        train_short.main(["--city", "demo"])
        new = pd.read_csv(project_root / "docs" / "forecast_short_city_demo.csv")
    finally:
        shutil.rmtree(src)
        shutil.copytree(keep, src)
        for f, b in docs_keep.items():
            (project_root / "docs" / f).write_bytes(b)
        monkeypatch.delenv("LFDR_CONFIG_DIR", raising=False)
        config.reload()
    m = lambda d: d[d["forecast"] == "model"].set_index("scenario")["nRMSE_%"]  # noqa: E731
    assert np.allclose(m(new), m(doc), atol=0.3)
    assert new[new["forecast"] != "model"].reset_index(drop=True).equals(doc[doc["forecast"] != "model"].reset_index(drop=True))
