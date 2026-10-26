"""Accuracy report on the held-out test period: XGBoost vs. two persistence baselines.

The calculations live in functions (load_test_split, score_table, aggregation_table) so the
tests can check them without writing files or drawing the figure.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {"XGBoost (train only)": "predicted_kw_raw", "XGBoost (train+val refit)": "predicted_refit_raw",
          "Same hour yesterday": "lag24_raw", "Same hour last week": "lag168_raw"}


def load_test_split(path=ROOT / "data" / "forecast_all.csv"):
    """Scored test rows: the test split with every model column present."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df[df["split"] == "test"].dropna(subset=["actual_kw_raw", "lag24_raw", "lag168_raw", "predicted_refit_raw"])


def score(g, col):
    e = g[col] - g["actual_kw_raw"]
    mean = g["actual_kw_raw"].mean()
    big = g["actual_kw_raw"] > 0.1 * mean                       # avoid MAPE blow-up near zero
    return {"MAE_kW": e.abs().mean(), "RMSE_kW": np.sqrt((e ** 2).mean()),
            "nRMSE_%": 100 * np.sqrt((e ** 2).mean()) / mean,
            "MAPE_%": 100 * (e[big].abs() / g.loc[big, "actual_kw_raw"]).mean()}


def score_table(test):
    """One row per (transformer or ALL, model)."""
    rows = []
    for tx, g in list(test.groupby("transformer_id")) + [("ALL", test)]:
        for name, col in MODELS.items():
            rows.append({"transformer": tx, "model": name, **score(g, col)})
    return pd.DataFrame(rows)


def aggregation_table(test, res):
    """nRMSE of each model per transformer (average) vs. for the SUM of all six transformers."""
    piv = {c: test.pivot(index="timestamp", columns="transformer_id", values=c)
           for c in ["actual_kw_raw"] + list(MODELS.values())}
    ok = piv["actual_kw_raw"].dropna().index
    tot_actual = piv["actual_kw_raw"].loc[ok].sum(axis=1)
    rows = []
    for name, col in MODELS.items():
        per_tx = res[(res["model"] == name) & (res["transformer"] != "ALL")]["nRMSE_%"].mean()
        tot_pred = piv[col].loc[ok].sum(axis=1)
        agg = 100 * np.sqrt(((tot_pred - tot_actual) ** 2).mean()) / tot_actual.mean()
        rows.append({"model": name, "single_tx_avg_nRMSE_%": per_tx, "all_six_sum_nRMSE_%": agg})
    return pd.DataFrame(rows)


def plot_first_week(test, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=False)
    t0 = test["timestamp"].min()
    for ax, (tx, g) in zip(axes.ravel(), test.groupby("transformer_id")):
        w = g[g["timestamp"] < t0 + pd.Timedelta(days=7)]
        ax.plot(w["timestamp"], w["actual_kw_raw"], label="actual", lw=1.5)
        ax.plot(w["timestamp"], w["predicted_refit_raw"], label="predicted", lw=1.5)
        ax.set_title(tx); ax.tick_params(axis="x", rotation=30); ax.set_ylabel("cluster load (kW)")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    test = load_test_split()
    print(f"test period: {test['timestamp'].min()} -> {test['timestamp'].max()}  ({len(test)} scored hours)")
    res = score_table(test)
    res.round(2).to_csv(ROOT / "docs" / "metrics.csv", index=False)

    print("\nnRMSE (%) on the test set (lower is better):")
    print(res.pivot(index="transformer", columns="model", values="nRMSE_%").round(1)[list(MODELS)].to_string())
    print("\nOverall (all transformers pooled):")
    print(res[res["transformer"] == "ALL"].drop(columns="transformer").round(2).to_string(index=False))

    # ---- aggregation analysis: error of the SUM of all transformers (bigger customer pool) ----
    print("\nAggregation effect (nRMSE %): single transformer (avg) vs. sum of all six")
    for _, r in aggregation_table(test, res).iterrows():
        print(f"  {r['model']:28s} single-transformer avg {r['single_tx_avg_nRMSE_%']:5.1f}%   "
              f"all-six sum {r['all_six_sum_nRMSE_%']:5.1f}%")

    plot_first_week(test, ROOT / "docs" / "fig_actual_vs_pred.png")
    print("\nSaved docs/metrics.csv and docs/fig_actual_vs_pred.png")


if __name__ == "__main__":
    main()
