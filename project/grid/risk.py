"""Risk classes (gap item F5), voltage and line alerts (F8) and recommendation confidence (F6).

Five transformer risk classes, from least to most severe:
    normal              below risk.elevated_from_pct
    elevated            at or above elevated_from_pct
    approaching         within risk.approaching_margin_pct below the overload threshold
    predicted_overload  the FORECAST is above the threshold
    actual_overload     the MEASURED load is above the threshold

Forecast risk and measured risk are classified separately and reported side by side; the overall class
is the more severe of the two. Settings come from config/operations.yaml.
"""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project/ -> import core
from core import config  # noqa: E402
from core.paths import DOCS_DIR  # noqa: E402

CLASSES = ("normal", "elevated", "approaching", "predicted_overload", "actual_overload")
SEVERITY = {c: i for i, c in enumerate(CLASSES)}
LABELS = {"normal": "Normal", "elevated": "Elevated", "approaching": "Approaching limit",
          "predicted_overload": "Predicted overload", "actual_overload": "Actual overload"}


class RiskModel:
    def __init__(self, ops_cfg=None):
        o = ops_cfg if ops_cfg is not None else config.operations()
        self.threshold = o["limits"]["overload_threshold_pct"]
        self.v_min, self.v_max = o["limits"]["voltage_min_pu"], o["limits"]["voltage_max_pu"]
        self.line_limit = o["limits"]["line_limit_pct"]
        r = o["risk"]
        self.elevated_from = r["elevated_from_pct"]
        self.approach_margin = r["approaching_margin_pct"]
        self.v_margin = r["voltage_margin_pu"]
        self.line_approaching = r["line_approaching_pct"]
        d = o["decisions"]
        self.conf_high, self.conf_medium = d["confidence_high"], d["confidence_medium"]

    # ------------------------------------------------------------------ transformers
    def band(self, pct, threshold=None):
        """normal / elevated / approaching / overload for one loading value (None -> None)."""
        if pct is None or (isinstance(pct, float) and math.isnan(pct)):
            return None
        thr = self.threshold if threshold is None else threshold
        if pct > thr:
            return "overload"
        if pct >= thr - self.approach_margin:
            return "approaching"
        if pct >= self.elevated_from:
            return "elevated"
        return "normal"

    def transformer(self, forecast_pct, measured_pct=None, threshold=None, forecast_upper_pct=None):
        """Risk of one transformer. forecast_upper_pct (an upper forecast bound, from Phase 4's interval
        forecasts) is used for the forecast class when given, so uncertainty can raise the risk."""
        f = self.band(forecast_upper_pct if forecast_upper_pct is not None else forecast_pct, threshold)
        m = self.band(measured_pct, threshold)
        f_cls = "predicted_overload" if f == "overload" else f
        m_cls = "actual_overload" if m == "overload" else m
        overall = max((c for c in (f_cls, m_cls) if c), key=SEVERITY.get, default="normal")
        return {"forecast_pct": forecast_pct, "measured_pct": measured_pct,
                "forecast_class": f_cls, "measured_class": m_cls, "risk_class": overall}

    def transformers(self, forecast_by_tx, measured_by_tx=None, threshold=None):
        measured_by_tx = measured_by_tx or {}
        return {t: self.transformer(f, measured_by_tx.get(t), threshold) for t, f in forecast_by_tx.items()}

    # ------------------------------------------------------------------ voltage and lines
    def voltage_alerts(self, bus_voltages):
        out = []
        for bus, v in sorted(bus_voltages.items(), key=lambda kv: kv[1]):
            if bus.lower() == "sub":
                continue
            if v < self.v_min:
                out.append(self._alert("voltage_low", "violation", bus, v, self.v_min, f"bus {bus.upper()} at {v:.4f} pu (< {self.v_min:g})"))
            elif v < self.v_min + self.v_margin:
                out.append(self._alert("voltage_low", "approaching", bus, v, self.v_min, f"bus {bus.upper()} at {v:.4f} pu, near {self.v_min:g}"))
            elif v > self.v_max:
                out.append(self._alert("voltage_high", "violation", bus, v, self.v_max, f"bus {bus.upper()} at {v:.4f} pu (> {self.v_max:g})"))
            elif v > self.v_max - self.v_margin:
                out.append(self._alert("voltage_high", "approaching", bus, v, self.v_max, f"bus {bus.upper()} at {v:.4f} pu, near {self.v_max:g}"))
        return out

    def line_alerts(self, line_loading):
        out = []
        for line, pct in sorted(line_loading.items(), key=lambda kv: -kv[1]):
            if pct > self.line_limit:
                out.append(self._alert("line", "violation", line, pct, self.line_limit, f"line {line} at {pct:.1f}% of rating"))
            elif pct >= self.line_approaching:
                out.append(self._alert("line", "approaching", line, pct, self.line_limit, f"line {line} at {pct:.1f}% of rating"))
        return out

    def network_alerts(self, metrics):
        """Voltage and line alerts from an engine result pack (needs bus_voltages_pu and line_loading_pct)."""
        return self.voltage_alerts(metrics.get("bus_voltages_pu", {})) + self.line_alerts(metrics.get("line_loading_pct", {}))

    @staticmethod
    def _alert(kind, level, element, value, limit, text):
        return {"type": kind, "level": level, "element": element.upper(), "value": round(value, 4), "limit": limit, "text": text}

    # ------------------------------------------------------------------ confidence
    def confidence(self, loading_by_tx, rel_error_by_tx, threshold=None):
        """Probability that every transformer stays at or below the threshold, if each one's true loading
        differs from the planned value by a normal error with standard deviation rel_error x loading.

        rel_error_by_tx comes from the forecast's normalised RMSE on the test period (forecast_errors()),
        so the confidence reflects how much the forecast that triggered the action can be trusted.
        Returns None when no error estimate is available."""
        if not rel_error_by_tx or not loading_by_tx:
            return None
        thr = self.threshold if threshold is None else threshold
        p = 1.0
        for t, load in loading_by_tx.items():
            sigma = rel_error_by_tx.get(t, max(rel_error_by_tx.values())) * load
            if sigma <= 0:
                p *= 1.0 if load <= thr else 0.0
            else:
                p *= 0.5 * (1 + math.erf((thr - load) / (sigma * math.sqrt(2))))
        label = "high" if p >= self.conf_high else "medium" if p >= self.conf_medium else "low"
        return {"probability": round(p, 3), "label": label,
                "basis": "probability that every transformer stays at or below the threshold, given the "
                         "forecast's test-period error for each transformer"}


def forecast_errors(path=DOCS_DIR / "metrics.csv", model="XGBoost (train+val refit)"):
    """{transformer: relative forecast error (nRMSE / 100)} from the published metrics, or {} if unavailable."""
    try:
        m = pd.read_csv(path)
        m = m[(m["model"] == model) & (m["transformer"] != "ALL")]
        return {r["transformer"]: r["nRMSE_%"] / 100 for _, r in m.iterrows()}
    except (OSError, KeyError, ValueError):
        return {}
