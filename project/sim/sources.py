"""Where the closed loop gets its loads from.

A demand source answers "what did the meters measure at time t?" and a forecast source "what is the
24 h-ahead forecast for time t (and its upper bound)?". Phase 5 has the classic source: the real eMARC test
period (six transformers, hourly) from the database. The synthetic city of Phase 7 plugs in as another
demand source with the same interface, so the orchestrator, database and dashboard do not change.

Forecast sources:
    PrecomputedForecasts  the committed 24 h-ahead forecasts (data/forecast.csv, imported into the database)
                          and their P10/P90 band: reproducible on any machine
    LiveForecasts         ml/forecaster.py forecast_at(t) at run time, from data/processed/clusters.parquet.
                          At step t it issues the forecast for t + 24 h (stored as pending, scored when its hour
                          passes); the upper bound still comes from the committed band
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import ids  # noqa: E402

HORIZON = pd.Timedelta(hours=24)


def _qcols(df):
    return sorted((c for c in df.columns if c[0] == "p" and c.endswith("_kw") and c[1:-3].isdigit()),
                  key=lambda c: int(c[1:-3]))


class ClassicSource:
    """Measured loads of the eMARC test period (scaled to grid kW), from the database."""
    name = "classic"
    resolution = pd.Timedelta(hours=1)

    def __init__(self, store, frame=None):
        self.frame = store.forecast_frame() if frame is None else frame
        if self.frame.empty:
            raise RuntimeError("the database holds no forecasts or measurements: import data/forecast.csv first")
        self.actual = self.frame.pivot(index="timestamp", columns="transformer_id", values="actual_kw")
        self.transformers = list(self.actual.columns)

    def times(self):
        return self.actual.index

    def measured(self, t):
        """{transformer: kW} at t, or None if any reading is missing."""
        t = pd.Timestamp(t)
        if t not in self.actual.index or self.actual.loc[t].isna().any():
            return None
        return {tx: float(v) for tx, v in self.actual.loc[t].items()}


class PrecomputedForecasts:
    name = "precomputed"

    def __init__(self, store, frame=None):
        frame = store.forecast_frame() if frame is None else frame
        self.model_version = store.import_cfg["model_version"]
        self.point_ = frame.pivot(index="timestamp", columns="transformer_id", values="predicted_kw")
        q = _qcols(frame)
        self.upper_col = q[-1] if len(q) >= 2 else None
        self.upper_ = frame.pivot(index="timestamp", columns="transformer_id", values=self.upper_col) if self.upper_col else None

    @staticmethod
    def _row(table, t):
        t = pd.Timestamp(t)
        if table is None or t not in table.index or table.loc[t].isna().any():
            return None
        return {tx: float(v) for tx, v in table.loc[t].items()}

    def point(self, t):
        return self._row(self.point_, t)

    def upper(self, t):
        return self._row(self.upper_, t)

    def issue(self, now):
        """Precomputed forecasts were all issued in advance: nothing to do."""
        return 0


class LiveForecasts(PrecomputedForecasts):
    """forecast_at(t) from the committed model at run time."""
    name = "live"

    def __init__(self, store, frame=None, clusters=None, forecaster=None):
        super().__init__(store, frame)
        sys.path.insert(0, str(ROOT / "ml"))
        from forecaster import Forecaster
        path = ROOT / "data" / "processed" / "clusters.parquet"
        if clusters is None:
            if not path.exists():
                raise FileNotFoundError(f"live forecasting needs {path} (the eMARC clusters); "
                                        "use forecast_source: precomputed without it")
            clusters = pd.read_parquet(path)
        self.history = clusters
        self.f = forecaster or Forecaster()
        self.store = store
        self.model_version = f"{self.f.version}-live"
        self._cache = {}

    def _compute(self, t):
        t = pd.Timestamp(t)
        if t not in self._cache:
            out = self.f.forecast_at(t, self.history)
            self._cache[t] = {r.transformer_id: float(r.predicted_kw) for r in out.itertuples()}
            self.store.write_forecasts([(t, ids.transformer_id(tx), kw) for tx, kw in self._cache[t].items()],
                                       self.model_version)
        return self._cache[t]

    def point(self, t):
        """The forecast for t (issued at t - 24 h: by an earlier step, or now if the loop started later)."""
        try:
            return self._compute(t)
        except ValueError:                      # no history row for t (outside the data)
            return None

    def issue(self, now):
        """At time `now` the forecast for now + 24 h can be made; store it as pending."""
        try:
            self._compute(pd.Timestamp(now) + HORIZON)
            return 1
        except ValueError:
            return 0


def forecast_source(kind, store, frame=None, **kw):
    if kind == "precomputed":
        return PrecomputedForecasts(store, frame)
    if kind == "live":
        return LiveForecasts(store, frame, **kw)
    raise ValueError(f"unknown forecast source {kind!r}")
