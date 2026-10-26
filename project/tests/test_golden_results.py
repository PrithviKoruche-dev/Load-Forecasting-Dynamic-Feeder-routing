"""Golden tests: the code must reproduce the published results exactly.

These lock in the numbers in docs/ (and quoted in the README and report) so that every later
change to the engine, the scenarios or the evaluation is caught if it alters them.
Only the timing columns are excluded, because they depend on the machine.
"""
import io

import pandas as pd
import pytest

import evaluate
import run_results

TIMING = ["mean_time_s", "max_time_s"]


def roundtrip(df):
    """Write and re-read through CSV so types match the committed file (None -> NaN, etc.)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return pd.read_csv(buf)


@pytest.fixture(scope="module")
def benchmark():
    return run_results.run_all(with_mitigation=True)


@pytest.mark.slow
def test_scenario_summary_matches_docs(benchmark, project_root):
    summary = benchmark[0]
    expected = pd.read_csv(project_root / "docs" / "results.csv").drop(columns=TIMING)
    got = roundtrip(summary).drop(columns=TIMING)
    pd.testing.assert_frame_equal(got, expected)


@pytest.mark.slow
def test_worst_hours_match_docs(benchmark, project_root):
    worst = benchmark[1]
    expected = pd.read_csv(project_root / "docs" / "results_worst_hour.csv")
    pd.testing.assert_frame_equal(roundtrip(worst), expected)


@pytest.mark.slow
def test_reroute_is_fast(benchmark):
    """Loose sanity bound: about 5-10 ms per decision on a laptop, so 0.25 s means something broke."""
    summary = benchmark[0]
    assert (summary["max_time_s"] < 0.25).all()


@pytest.mark.slow
def test_mitigation_table_matches_docs(benchmark, project_root):
    expected = pd.read_csv(project_root / "docs" / "results_mitigation.csv")
    pd.testing.assert_frame_equal(roundtrip(benchmark[2]), expected)


def test_forecast_metrics_match_docs(project_root):
    test = evaluate.load_test_split()
    expected = pd.read_csv(project_root / "docs" / "metrics.csv")
    got = roundtrip(evaluate.score_table(test).round(2))
    pd.testing.assert_frame_equal(got, expected)


def test_xgboost_beats_both_baselines_everywhere():
    res = evaluate.score_table(evaluate.load_test_split())
    piv = res.pivot(index="transformer", columns="model", values="nRMSE_%")
    xgb = piv["XGBoost (train+val refit)"]
    assert (xgb < piv["Same hour yesterday"]).all()
    assert (xgb < piv["Same hour last week"]).all()


def test_aggregation_effect():
    """Forecasting the sum of all six transformers is far more accurate (10.8% vs about 25%)."""
    test = evaluate.load_test_split()
    agg = evaluate.aggregation_table(test, evaluate.score_table(test)).set_index("model")
    row = agg.loc["XGBoost (train+val refit)"]
    assert row["all_six_sum_nRMSE_%"] == pytest.approx(10.8, abs=0.05)
    assert row["single_tx_avg_nRMSE_%"] == pytest.approx(25.4, abs=0.05)
