"""Forecast intervals for the published 24 h-ahead model (gap-analysis item E11), by conformal calibration.

The band comes from the ratio actual / forecast, per transformer and block of hours (config: intervals):

    P10 = forecast x q10(ratio)          P90 = forecast x q90(ratio)

A ratio (not a difference) keeps the band proportional to the load, so it scales with the stress scenarios.

Two ways to choose the ratios:
  static   the validation period's ratios from the TRAIN-ONLY model. The published model was refit on
           train+val, so its own validation errors would be optimistic; the train-only model never saw that
           data (and differs from the published one by about 0.07 kW on average).
  rolling  (default) for a target hour t, the ratios of the last window_days whose actual value was known by
           the time the forecast was issued (t - 24 h): validation-period ratios at first, then the published
           model's own test-period ratios as they become known. Past data only, and it follows seasonal drift
           (the test period, Feb-Mar, warms up after the Dec-Feb validation period).

The test period only CHECKS the bands: about 80% of actual loads should fall inside P10-P90.

    python ml/intervals.py   ->  data/forecast_intervals.csv, docs/interval_calibration.csv, docs/metrics_intervals.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import config, ids  # noqa: E402

ISSUE_LEAD = pd.Timedelta(hours=24)          # a forecast for t is issued at t - 24 h


def _cfg():
    return config.forecasting()["intervals"]


def hour_block(ts, blocks):
    return (pd.DatetimeIndex(ts).hour // (24 // blocks)).astype(int)


def _ratios(fa, blocks):
    """Ratio history: validation period from the train-only model, test period from the published model."""
    v = fa[(fa["split"] == "val") & (fa["predicted_kw_raw"] > 0)].assign(
        ratio=lambda d: d["actual_kw_raw"] / d["predicted_kw_raw"])
    t = fa[(fa["split"] == "test") & (fa["predicted_refit_raw"] > 0)].assign(
        ratio=lambda d: d["actual_kw_raw"] / d["predicted_refit_raw"])
    h = pd.concat([v, t])[["timestamp", "transformer_id", "split", "ratio"]].dropna()
    h["block"] = hour_block(h["timestamp"], blocks)
    return h


def calibrate(fa, lower=None, upper=None, blocks=None):
    """Static table: quantiles of the validation ratios per transformer and hour block."""
    c = _cfg()
    lower, upper, blocks = lower or c["lower_quantile"], upper or c["upper_quantile"], blocks or c["hour_blocks"]
    v = _ratios(fa, blocks)
    v = v[v["split"] == "val"]
    t = (v.groupby(["transformer_id", "block"])["ratio"]
         .agg(q_lower=lambda r: r.quantile(lower), q_upper=lambda r: r.quantile(upper), n="size").reset_index())
    t.attrs.update(lower=lower, upper=upper, blocks=blocks)
    return t


def apply(df, table, pred_col="predicted_kw"):
    """Static bands: add p_lower_kw / p_upper_kw to a frame with timestamp, transformer_id and a forecast."""
    out = df.copy()
    out["block"] = hour_block(out["timestamp"], table.attrs.get("blocks", _cfg()["hour_blocks"]))
    out = out.merge(table[["transformer_id", "block", "q_lower", "q_upper"]], on=["transformer_id", "block"], how="left")
    out["p_lower_kw"] = out[pred_col] * out["q_lower"]
    out["p_upper_kw"] = out[pred_col] * out["q_upper"]
    return out.drop(columns=["block", "q_lower", "q_upper"])


def apply_rolling(fa, df, window_days=None, lower=None, upper=None, blocks=None, pred_col="predicted_kw"):
    """Rolling bands for the rows of df (timestamp, transformer_id, forecast): for target t, the ratio quantiles
    over (t - 24 h - window, t - 24 h], i.e. only actuals known when the forecast was issued."""
    c = _cfg()
    window_days, blocks = window_days or c["window_days"], blocks or c["hour_blocks"]
    lower, upper = lower or c["lower_quantile"], upper or c["upper_quantile"]
    hist = _ratios(fa, blocks)
    out = df.copy()
    out["block"] = hour_block(out["timestamp"], blocks)
    out["issued"] = out["timestamp"] - ISSUE_LEAD
    parts = []
    for (tx, b), g in out.groupby(["transformer_id", "block"]):
        h = hist[(hist["transformer_id"] == tx) & (hist["block"] == b)].sort_values("timestamp").set_index("timestamp")
        roll = h["ratio"].rolling(f"{window_days}D")
        q = pd.DataFrame({"known_at": h.index, "q_lower": roll.quantile(lower).values,
                          "q_upper": roll.quantile(upper).values, "n": roll.count().values})
        g = pd.merge_asof(g.sort_values("issued"), q, left_on="issued", right_on="known_at", direction="backward")
        parts.append(g)
    out = pd.concat(parts).sort_values(["transformer_id", "timestamp"]).reset_index(drop=True)
    out["p_lower_kw"] = out[pred_col] * out["q_lower"]
    out["p_upper_kw"] = out[pred_col] * out["q_upper"]
    return out.drop(columns=["block", "issued", "known_at", "q_lower", "q_upper", "n"])


def pinball(y, q, alpha):
    d = y - q
    return np.mean(np.maximum(alpha * d, (alpha - 1) * d))


def coverage_table(test, lower, upper, method=""):
    """Share of actual loads inside the band, below P10 and above P90, band width, and pinball loss."""
    rows = []
    for tx, g in list(test.groupby("transformer_id")) + [("ALL", test)]:
        g = g.dropna(subset=["actual_kw", "p_lower_kw", "p_upper_kw"])
        y = g["actual_kw"]
        rows.append({"method": method, "transformer": tx, "hours": len(g),
                     "coverage_%": 100 * ((y >= g["p_lower_kw"]) & (y <= g["p_upper_kw"])).mean(),
                     "below_lower_%": 100 * (y < g["p_lower_kw"]).mean(),
                     "above_upper_%": 100 * (y > g["p_upper_kw"]).mean(),
                     "mean_width_%_of_forecast": 100 * ((g["p_upper_kw"] - g["p_lower_kw"]) / g["predicted_kw"]).mean(),
                     "pinball_lower": pinball(y, g["p_lower_kw"], lower),
                     "pinball_upper": pinball(y, g["p_upper_kw"], upper)})
    return pd.DataFrame(rows)


def build(fa=None):
    """(static calibration table, test rows with the configured method's bands, coverage of both methods)."""
    c = _cfg()
    fa = pd.read_csv(ROOT / "data" / "forecast_all.csv", parse_dates=["timestamp"]) if fa is None else fa
    table = calibrate(fa)
    test = fa[fa["split"] == "test"][["timestamp", "transformer_id", "predicted_kw", "actual_kw"]]
    static = apply(test, table)
    rolling = apply_rolling(fa, test)
    cov = pd.concat([coverage_table(static, c["lower_quantile"], c["upper_quantile"], "static"),
                     coverage_table(rolling, c["lower_quantile"], c["upper_quantile"],
                                    f"rolling {c['window_days']} d")], ignore_index=True)
    chosen = rolling if c["method"] == "rolling" else static
    return table, chosen.sort_values(["transformer_id", "timestamp"]).reset_index(drop=True), cov


def main():
    c = _cfg()
    table, test, cov = build()
    lo, hi = int(round(100 * c["lower_quantile"])), int(round(100 * c["upper_quantile"]))
    out = test[["timestamp", "transformer_id", "p_lower_kw", "p_upper_kw"]].rename(
        columns={"p_lower_kw": f"p{lo}_kw", "p_upper_kw": f"p{hi}_kw"})
    out[[f"p{lo}_kw", f"p{hi}_kw"]] = out[[f"p{lo}_kw", f"p{hi}_kw"]].round(3)
    ids.add_transformer_ids(out).to_csv(ROOT / "data" / "forecast_intervals.csv", index=False)
    table.round(4).to_csv(ROOT / "docs" / "interval_calibration.csv", index=False)
    cov.round(3).to_csv(ROOT / "docs" / "metrics_intervals.csv", index=False)
    print(f"P{lo}-P{hi} bands on the test period (target coverage {hi - lo}%), method used: {c['method']}")
    print(cov.round(1).to_string(index=False))
    print("\nSaved data/forecast_intervals.csv, docs/interval_calibration.csv and docs/metrics_intervals.csv")


if __name__ == "__main__":
    main()
