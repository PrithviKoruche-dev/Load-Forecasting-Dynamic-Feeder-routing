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


class CitySource:
    """Measured loads of a synthetic city (Phase 7): cities/<name>/, 15-minute steps over a year. The
    scenario's events (heatwave, EV charging, ...) change the demand itself; each step's measurements are
    written to the database as they arrive, so forecasts can be scored against them."""
    name = "city"

    def __init__(self, store, city, scenario=None):
        sys.path.insert(0, str(ROOT / "city"))
        from demand import CityDemand
        from core import config
        self.store, self.city = store, city
        self.demand = CityDemand(city)
        self.resolution = self.demand.times[1] - self.demand.times[0]
        self.transformers = list(config.grid()["transformers"])
        missing = set(self.transformers) - set(self.demand.transformers)
        if missing:
            raise RuntimeError(f"cities/{city}/consumers.csv has no consumers for {sorted(missing)}: generate again")
        self.events = ()
        self.source = store.import_cfg["measurement_source"]
        self.set_events(scenario.events if scenario is not None else ())

    def set_events(self, events):
        self.events = tuple(events)
        self.table = self.demand.matrix(self.events)[self.transformers]

    def times(self):
        from core import config
        c = config.simulation()["clock"]
        t = self.table.index
        lo = t.min() if c["start"] == "test_start" else pd.Timestamp(c["start"])
        hi = t.max() if c["end"] == "test_end" else pd.Timestamp(c["end"])
        return t[(t >= lo) & (t <= hi)]

    def measured(self, t):
        t = pd.Timestamp(t)
        if t not in self.table.index:
            return None
        return {tx: float(v) for tx, v in self.table.loc[t].items()}

    def record(self, t, loads):
        """The meters' readings for t arrive: store them."""
        if loads is not None:
            self.store.write_measurements([{"time": pd.Timestamp(t).to_pydatetime(), "asset_id": ids.transformer_id(tx),
                                            "source": self.source, "kw": round(kw, 3), "quality": "good"}
                                           for tx, kw in loads.items()])


class PersistenceForecasts:
    """24 h-ahead forecast for the synthetic city: the load at the same time the day before, with a P10-P90 band
    by rolling conformal calibration (config forecasting.yaml intervals: the ratio actual / forecast over the
    last window_days, per transformer and block of the day, using only loads known at issue time t - 24 h).
    Phase 9 replaces it with a model trained on the simulator's history."""
    name = "persistence"

    def __init__(self, store, demand):
        from core import config
        self.store, self.demand = store, demand
        self.model_version = store.import_cfg["model_version"]
        self.cfg = config.forecasting()["intervals"]
        self._key = None

    def _refresh(self):
        if self._key == self.demand.events:
            return
        a = self.demand.table
        lag = int(HORIZON / self.demand.resolution)
        self.point_ = a.shift(lag)
        ratio = (a / self.point_).replace([float("inf")], float("nan"))
        hours = 24 // self.cfg["hour_blocks"]
        block = a.index.hour // hours
        lo_q, hi_q = self.cfg["lower_quantile"], self.cfg["upper_quantile"]
        win, min_n = f"{self.cfg['window_days']}D", 7 * (hours * int(pd.Timedelta(hours=1) / self.demand.resolution))
        lo, hi = pd.DataFrame(index=a.index, columns=a.columns, dtype=float), pd.DataFrame(index=a.index, columns=a.columns, dtype=float)
        for b in sorted(set(block)):
            r = ratio[block == b]
            ql = r.rolling(win, min_periods=min_n).quantile(lo_q)
            qh = r.rolling(win, min_periods=min_n).quantile(hi_q)
            targets = a.index[block == b]
            issued = targets - HORIZON                           # known at issue time: ratios up to t - 24 h
            pos = ql.index.searchsorted(issued, side="right") - 1
            ok = pos >= 0
            lo.loc[targets[ok]] = ql.iloc[pos[ok]].to_numpy()
            hi.loc[targets[ok]] = qh.iloc[pos[ok]].to_numpy()
        self.lower_ = self.point_ * lo
        self.upper_ = self.point_ * hi
        self._key = self.demand.events

    @staticmethod
    def _row(table, t):
        t = pd.Timestamp(t)
        if t not in table.index or table.loc[t].isna().any():
            return None
        return {tx: float(v) for tx, v in table.loc[t].items()}

    def point(self, t):
        self._refresh()
        return self._row(self.point_, t)

    def upper(self, t):
        self._refresh()
        return self._row(self.upper_, t)

    def issue(self, now):
        """At time `now` the forecast for now + 24 h is known (it is the load measured now): store it, with its band."""
        self._refresh()
        target = pd.Timestamp(now) + HORIZON
        p = self._row(self.point_, target)
        if p is None:
            return 0
        aid = {tx: ids.transformer_id(tx) for tx in p}
        self.store.write_forecasts([(target, aid[tx], kw) for tx, kw in p.items()], self.model_version)
        rows = []
        for q, table in ((self.cfg["lower_quantile"], self.lower_), (self.cfg["upper_quantile"], self.upper_)):
            b = self._row(table, target)
            if b:
                rows += [(target, aid[tx], q, kw) for tx, kw in b.items()]
        self.store.write_quantiles(rows, self.model_version, "rolling conformal")
        return 1


def forecast_source(kind, store, frame=None, demand=None, **kw):
    if kind == "precomputed":
        return PrecomputedForecasts(store, frame)
    if kind == "live":
        return LiveForecasts(store, frame, **kw)
    if kind == "persistence":
        return PersistenceForecasts(store, demand)
    raise ValueError(f"unknown forecast source {kind!r}")
