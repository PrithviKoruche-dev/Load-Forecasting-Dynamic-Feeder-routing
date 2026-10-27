"""Measurement streams of the synthetic city (Phase 8, gap item C5): transformer meters and smart meters, kept
separate, with delays, losses, outages, faults and quality flags.

    model = MeterModel(city_demand)          # settings: city.yaml `meters`
    head = HeadEnd(model)
    batch = head.step(i, now, events)        # interval i has just ended (now = its end)

Transformer meters (SCADA): one per transformer, available at once, true load x (1 + accuracy noise); a
sensor is out of service during its failure periods (no value).

Smart meters (AMI): one per consumer (meter IDs from consumers.csv). A consumer's 15-minute load is its share
of the transformer's load (size x its type's profile, plus a car if it has one and the EV event is on),
varied individually; together the meters see the transformer load minus the LV losses. Each reading arrives
after a lognormal delay; some are lost; while a transformer's data concentrator is down, its readings wait
and arrive when it comes back; stuck meters repeat one value from their fault time.

The head-end collects what has arrived, flags a meter after `stuck_after` identical readings in a row, and
estimates every transformer's load per interval from the readings received so far:
    estimate = (sum of received readings / expected share of those consumers) / (1 - lv_losses)
    quality  = good (>= good_share of the expected load has reported), estimated, poor, missing
An estimate is revised whenever late readings for its interval arrive.

Everything is drawn from (seed, interval), so the streams are the same on every run, and the head-end's state at
any time can be rebuilt by replaying the intervals of its window (used after a restart or a time jump).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demand import TYPES, ev_profile  # noqa: E402

NS = 10 ** 9
DEFAULTS = {"enabled": False,
            "transformer": {"accuracy": 0.01, "failures": {"per_year": 2, "mean_hours": 12}},
            "smart": {"individual_variation": 0.2, "accuracy": 0.01, "delay_s": {"median": 120, "sigma": 1.2},
                      "lost": 0.01, "concentrator_outages": {"per_year": 4, "mean_hours": 6}, "stuck_meters": 0.005,
                      "stuck_after": 8},
            "lv_losses": 0.04,
            "head_end": {"good_share": 0.98, "estimated_share": 0.7, "window_hours": 48},
            "store_readings": "exceptions"}


def _merge(base, over, where="meters"):
    out = {}
    for k, v in base.items():
        if isinstance(v, dict):
            o = over.get(k, {}) if over else {}
            if not isinstance(o, dict):
                raise ValueError(f"city.yaml: {where}.{k} must be a mapping")
            out[k] = _merge(v, o, f"{where}.{k}")
        else:
            out[k] = over.get(k, v) if over else v
    unknown = set(over or {}) - set(base)
    if unknown:
        raise ValueError(f"city.yaml: unknown settings {where}.{sorted(unknown)}")
    return out


def settings(spec):
    """The `meters` section of city.yaml with defaults filled in, checked. Without the section (or with
    enabled: false) the operator sees the true loads: perfect measurement, as before Phase 8."""
    m = _merge(DEFAULTS, spec.get("meters") or {})
    shares = [("transformer.accuracy", m["transformer"]["accuracy"]), ("smart.accuracy", m["smart"]["accuracy"]),
              ("smart.individual_variation", m["smart"]["individual_variation"]), ("smart.lost", m["smart"]["lost"]),
              ("smart.stuck_meters", m["smart"]["stuck_meters"]), ("lv_losses", m["lv_losses"]),
              ("head_end.good_share", m["head_end"]["good_share"]),
              ("head_end.estimated_share", m["head_end"]["estimated_share"])]
    for k, v in shares:
        if not isinstance(v, (int, float)) or not 0 <= v < 1 + (k.startswith("head_end")):
            raise ValueError(f"city.yaml: meters.{k} must be a number from 0 to 1, got {v!r}")
    if m["head_end"]["estimated_share"] > m["head_end"]["good_share"]:
        raise ValueError("city.yaml: meters.head_end.estimated_share must not be above good_share")
    for k, v in (("transformer.failures.per_year", m["transformer"]["failures"]["per_year"]),
                 ("transformer.failures.mean_hours", m["transformer"]["failures"]["mean_hours"]),
                 ("smart.concentrator_outages.per_year", m["smart"]["concentrator_outages"]["per_year"]),
                 ("smart.concentrator_outages.mean_hours", m["smart"]["concentrator_outages"]["mean_hours"]),
                 ("smart.delay_s.median", m["smart"]["delay_s"]["median"]), ("smart.delay_s.sigma", m["smart"]["delay_s"]["sigma"]),
                 ("head_end.window_hours", m["head_end"]["window_hours"])):
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"city.yaml: meters.{k} must be a number of at least 0, got {v!r}")
    if m["smart"]["delay_s"]["median"] <= 0 or m["head_end"]["window_hours"] <= 0:
        raise ValueError("city.yaml: meters.smart.delay_s.median and meters.head_end.window_hours must be above 0")
    if m["store_readings"] not in ("all", "exceptions", "none"):
        raise ValueError("city.yaml: meters.store_readings must be all, exceptions or none")
    if not isinstance(m["smart"]["stuck_after"], int) or m["smart"]["stuck_after"] < 2:
        raise ValueError("city.yaml: meters.smart.stuck_after must be a whole number of at least 2")
    return m


def _windows(rng, n_tx, start, end, per_year, mean_hours):
    """Random outage windows per transformer (Poisson starts, exponential durations): list of (start, end) ns."""
    span = (end - start).total_seconds()
    out = []
    for _ in range(n_tx):
        k = rng.poisson(per_year * span / (365 * 86400))
        s = np.sort(rng.uniform(0, span, k))
        d = rng.exponential(mean_hours * 3600, k)
        out.append([(int(start.value + a * NS), int(start.value + (a + b) * NS)) for a, b in zip(s, d)])
    return out


def _covering(windows, t_ns):
    """End of the window containing each time (0 where none). windows: list of (start, end)."""
    end = np.zeros(len(t_ns), dtype=np.int64)
    for a, b in windows:
        inside = (t_ns >= a) & (t_ns < b)
        end[inside] = b
    return end


class MeterModel:
    def __init__(self, demand):
        self.d = demand
        self.cfg = settings(demand.spec)
        self.seed = int(demand.spec["seed"])
        c = demand.consumers.reset_index(drop=True)
        self.consumers = c
        self.meter_ids = c["meter_id"].to_numpy()
        self.tx_names = list(demand.transformers)
        self.tx_index = c["transformer"].map({t: k for k, t in enumerate(self.tx_names)}).to_numpy()
        self.type_index = c["type"].map({t: k for k, t in enumerate(TYPES)}).to_numpy()
        self.size = c["size"].to_numpy(dtype=float)
        self.ev_rank = c["ev_rank"].to_numpy(dtype=float)
        self.n = len(c)
        self.step = demand.times[1] - demand.times[0]
        rng = np.random.default_rng([self.seed, 301])
        t0, t1 = demand.times[0], demand.times[-1] + self.step
        sm, tm = self.cfg["smart"], self.cfg["transformer"]
        self.dt_failures = _windows(rng, len(self.tx_names), t0, t1, tm["failures"]["per_year"], tm["failures"]["mean_hours"])
        self.outages = _windows(rng, len(self.tx_names), t0, t1, sm["concentrator_outages"]["per_year"],
                                sm["concentrator_outages"]["mean_hours"])
        self.dt_noise = rng.normal(0, tm["accuracy"], (len(demand.times), len(self.tx_names)))
        self.dt_missing = np.zeros(self.dt_noise.shape, dtype=bool)
        t = demand.times.as_unit("ns").asi8
        for k, wins in enumerate(self.dt_failures):
            for a, b in wins:
                self.dt_missing[(t >= a) & (t < b), k] = True
        stuck = rng.random(self.n) < sm["stuck_meters"]
        self.stuck_from = np.full(self.n, np.iinfo(np.int64).max)
        self.stuck_from[stuck] = rng.integers(0, len(demand.times), stuck.sum())
        self._units = {}
        self._stuck_value = {}
        self.loss = self.cfg["lv_losses"]

    # ------------------------------------------------------------------ consumer loads
    def _unit(self, events):
        key = tuple(sorted(events))
        if key not in self._units:
            units, _, names = self.d.units(events)
            U = np.vstack([units[k] for k in TYPES])                         # types x steps
            ev = []
            for name in names:
                e = self.d.spec["events"][name]
                if e["kind"] == "ev_charging":
                    owners = np.isin(self.consumers["type"].to_numpy(), e["consumers"]) & (self.ev_rank < e["share"])
                    ev.append((owners, ev_profile(e, self.d.cal)))
            self._units[key] = (U, ev)
        return self._units[key]

    def expected(self, i, events):
        """Each consumer's typical load (kW) in interval i: size x its type's profile (+ its car)."""
        U, ev = self._unit(events)
        w = self.size * U[self.type_index, i]
        for owners, prof in ev:
            w = w + owners * prof[i]
        return w

    def readings(self, i, events):
        """Interval i: every meter's reading (kW) and the time it arrives (ns; -1 = lost)."""
        sm = self.cfg["smart"]
        rng = np.random.default_rng([self.seed, 303, i])
        w = self.expected(i, events)
        eps = self.d.noise[i, self.tx_index]                                  # the transformer's own variation
        vary = rng.normal(0, sm["individual_variation"], self.n) + rng.normal(0, sm["accuracy"], self.n)
        kw = np.clip(w * (1 + eps) * (1 - self.loss) * (1 + vary), 0, None)
        stuck = self.stuck_from <= i
        if stuck.any():
            for m in np.flatnonzero(stuck):
                key = (m, tuple(sorted(events)))
                if key not in self._stuck_value:
                    j = int(self.stuck_from[m])
                    r = np.random.default_rng([self.seed, 303, j])
                    v = r.normal(0, sm["individual_variation"], self.n) + r.normal(0, sm["accuracy"], self.n)
                    w_j = self.expected(j, events)[m]
                    self._stuck_value[key] = max(0.0, w_j * (1 + self.d.noise[j, self.tx_index[m]]) * (1 - self.loss)
                                                 * (1 + v[m]))
                kw[m] = self._stuck_value[key]
        end = (self.d.times[i] + self.step).value
        med, sig = sm["delay_s"]["median"], sm["delay_s"]["sigma"]
        delay = rng.lognormal(math.log(med), sig, self.n)
        arrive = end + (delay * NS).astype(np.int64)
        for k, wins in enumerate(self.outages):                               # concentrator down: wait for it
            if not wins:
                continue
            sel = self.tx_index == k
            back = _covering(wins, arrive[sel])
            late = back > 0
            if late.any():
                a = arrive[sel]
                a[late] = back[late] + (delay[sel][late] * NS).astype(np.int64)
                arrive[sel] = a
        arrive[rng.random(self.n) < sm["lost"]] = -1
        return np.round(kw, 4), arrive

    # ------------------------------------------------------------------ transformer meters
    def transformer_readings(self, i, true_kw):
        """Transformer meters for interval i: true load x (1 + accuracy noise); NaN while a sensor is out."""
        kw = np.asarray(true_kw, dtype=float) * (1 + self.dt_noise[i])
        kw[self.dt_missing[i]] = np.nan
        return kw

    def transformer_series(self, table):
        """Transformer-meter readings for a table of true loads (the year x transformers, in tx_names order)."""
        out = table.to_numpy(dtype=float) * (1 + self.dt_noise[:len(table)])
        out[self.dt_missing[:len(table)]] = np.nan
        return pd.DataFrame(out, index=table.index, columns=table.columns)


def observed_history(model, table):
    """The loads an operator has recorded for a table of true loads (times x transformers): the transformer
    meters, a gap (sensor out) filled with the reading of the same time the day before (repeatedly, for long
    gaps). Every value uses only readings from its own time or earlier, except the first day of the year."""
    h = model.transformer_series(table)
    day = int(pd.Timedelta(days=1) / model.step)
    for _ in range(7):
        if not h.isna().any().any():
            break
        h = h.fillna(h.shift(day))
    return h.ffill().bfill()


class HeadEnd:
    """Collects smart-meter readings as they arrive and estimates each transformer's load per interval.

    Per interval of the window it keeps every meter's reading (NaN until received). A meter flagged as stuck is
    left out of every estimate of the window, also those made before it was flagged (they are revised): so the
    state after replaying a window is the same as after running through it."""

    def __init__(self, model):
        self.m = model
        hc = model.cfg["head_end"]
        self.good, self.est = hc["good_share"], hc["estimated_share"]
        self.window = int(pd.Timedelta(hours=hc["window_hours"]) / model.step)
        self.stuck_after = model.cfg["smart"]["stuck_after"]
        self.n_tx = len(model.tx_names)
        self.pending = {}                      # interval -> (kw, arrive) of readings not yet delivered
        self.agg = {}                          # interval -> {"kw": received readings (NaN = not yet), "w": expected}
        self.last_val = np.full(model.n, np.nan)
        self.run = np.zeros(model.n, dtype=int)
        self.last_interval = np.full(model.n, -10)
        self.suspect = np.zeros(model.n, dtype=bool)
        self.position = None                   # the last interval processed

    def step(self, i, events, now_ns=None):
        """Interval i has ended: generate its readings, deliver everything that has arrived by now (the end of
        interval i), update the estimates. Returns the readings delivered and the estimates that changed."""
        if self.position is None or i != self.position + 1:
            self.rebuild(i - 1, events)
        now = now_ns if now_ns is not None else (self.m.d.times[i] + self.m.step).value
        kw, arrive = self.m.readings(i, events)
        self.pending[i] = (kw, arrive)
        self.agg[i] = {"kw": np.full(self.m.n, np.nan), "w": self.m.expected(i, events)}
        delivered, changed = [], set()
        was = self.suspect.copy()
        for j in sorted(self.pending):
            kw_j, arr_j = self.pending[j]
            got = (arr_j >= 0) & (arr_j <= now)
            if got.any():
                idx = np.flatnonzero(got)
                self._deliver(j, idx, kw_j[idx], arr_j[idx], delivered)
                changed.add(j)
                arr_j = arr_j.copy()
                arr_j[idx] = -2                                               # delivered
                self.pending[j] = (kw_j, arr_j)
            if not (self.pending[j][1] >= 0).any():                           # nothing more to come
                self.pending.pop(j)
        new = np.flatnonzero(self.suspect & ~was)
        if len(new):                                                          # newly flagged: revise their intervals
            changed |= {j for j, a in self.agg.items() if not np.isnan(a["kw"][new]).all()}
        for j in [j for j in self.pending if j <= i - self.window]:           # too late: counted as lost
            self.pending.pop(j)
        for j in [j for j in self.agg if j <= i - self.window]:
            self.agg.pop(j)
        self.position = i
        return delivered, {j: self.estimate(j) for j in sorted(changed) if j in self.agg}

    def _deliver(self, j, idx, kw, arrive, delivered):
        self.agg[j]["kw"][idx] = kw
        in_order = self.last_interval[idx] == j - 1                           # stuck detection on consecutive readings
        same = in_order & np.isclose(self.last_val[idx], kw) & (kw > 0)       # zero use is not a fault
        self.run[idx] = np.where(same, self.run[idx] + 1, np.where(in_order, 0, self.run[idx]))
        newer = j > self.last_interval[idx]
        self.last_val[idx] = np.where(newer, kw, self.last_val[idx])
        self.last_interval[idx] = np.where(newer, j, self.last_interval[idx])
        self.suspect[idx] |= self.run[idx] >= self.stuck_after - 1
        delivered.append((j, idx, kw, arrive, self.suspect[idx].copy()))

    def estimate(self, j):
        """{tx: (kW, quality, reported share)} for interval j from the readings received so far (suspect meters
        left out)."""
        a = self.agg[j]
        use = ~np.isnan(a["kw"]) & ~self.suspect
        tx = self.m.tx_index
        got = np.bincount(tx[use], weights=a["kw"][use], minlength=self.n_tx)
        share_w = np.bincount(tx[use], weights=a["w"][use], minlength=self.n_tx)
        total_w = np.bincount(tx, weights=a["w"], minlength=self.n_tx)
        out = {}
        for k, name in enumerate(self.m.tx_names):
            share = share_w[k] / total_w[k] if total_w[k] > 0 else 0.0
            if share <= 0:
                out[name] = (None, "missing", 0.0)
                continue
            kw = got[k] / share / (1 - self.m.loss)
            q = "good" if share >= self.good else "estimated" if share >= self.est else "poor"
            out[name] = (float(kw), q, float(share))
        return out

    def rebuild(self, upto, events):
        """State as if the head-end had run up to interval `upto` (after a restart or a jump in time)."""
        self.pending, self.agg = {}, {}
        self.last_val[:] = np.nan
        self.run[:] = 0
        self.last_interval[:] = -10
        self.suspect[:] = False
        start = max(0, upto - self.window + 1)
        self.position = start - 1
        for j in range(start, upto + 1):
            self.step(j, events)
        self.position = upto
