"""Demand of the synthetic city (gap items A2, C1, C4): every transformer's load at 15-minute resolution for a
whole year, from its consumers, the calendar, the seasons, the weather and events.

    d = CityDemand("demo")                 # cities/demo/city.yaml and the generated consumers.csv
    m = d.matrix(events=["ev_adoption"])   # DataFrame: 15-minute times x transformers, kW

Consumer types (per typical consumer; each consumer has a size factor around 1):
    residential  the eMARC households' fitted profile (city/profiles/residential_emarc.csv: season, weekday or
                 weekend, hour of day, plus cooling per degree above 24 °C) times demand.residential.scale
    commercial   an hourly weekday profile, scaled on Saturdays and Sundays/holidays, plus cooling while open
    industrial   two shifts, scaled at weekends, a little cooling
    mixed        a share of a household plus the rest of a shop
Calendar: weekday / Saturday / Sunday / public holiday; season from the month (config/simulation.yaml).
Weather: monthly means, a daily cycle by season and day-to-day variation (all seeded).
Events: demand_change (a factor on some consumer types at some hours of some days), heatwave (extra degrees,
which act through the cooling terms) and ev_charging (a share of households charge a car in the evening).

The same seed gives the same city: the noise and the weather do not depend on which events are switched on,
so scenarios differ only by their events.
"""
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import config  # noqa: E402

CITIES = ROOT / "cities"
PROFILES = Path(__file__).resolve().parent / "profiles" / "residential_emarc.csv"
COOLING_BASE_C = 24.0
TYPES = ("residential", "commercial", "industrial", "mixed")
STEP = pd.Timedelta(minutes=15)


def city_dir(name):
    p = Path(name)
    return p if p.is_dir() and (p / "city.yaml").exists() else CITIES / str(name)


def load_spec(name):
    path = city_dir(name) / "city.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no city spec at {path}")
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    events = spec.get("events") or {}
    for ev, e in events.items():
        if e.get("kind") not in ("demand_change", "heatwave", "ev_charging"):
            raise ValueError(f"city.yaml: event {ev}: kind must be demand_change, heatwave or ev_charging")
    for sc in spec.get("scenarios", []):
        unknown = set(sc.get("events", [])) - set(events)
        if unknown:
            raise ValueError(f"city.yaml: scenario {sc['name']!r} uses unknown events {sorted(unknown)}")
    return spec


def year_times(spec):
    y = int(spec["year"])
    return pd.date_range(f"{y}-01-01 00:00", f"{y}-12-31 23:45", freq=STEP)


def _hourly_to_steps(values24, minutes):
    """Hourly values (index = hour) to any minute of the day, linearly interpolated, wrapping at midnight."""
    v = np.asarray(values24, dtype=float)
    return np.interp(minutes / 60.0, np.arange(25), np.append(v, v[0]))


class Calendar:
    def __init__(self, spec, times):
        self.times = times
        hol = {pd.Timestamp(d).date() for d in spec.get("holidays", [])}
        day = times.normalize()
        is_hol = pd.Series(day.date).isin(hol).to_numpy()
        dow = times.day_of_week.to_numpy()
        self.daytype = np.where(is_hol | (dow == 6), "sunday", np.where(dow == 5, "saturday", "weekday"))
        self.weekend = self.daytype != "weekday"                # households: Saturday, Sunday and holidays
        by_month = {m: s for s, ms in config.simulation()["seasons"].items() for m in ms}   # the shared calendar
        self.season = np.array([by_month[m] for m in times.month])
        self.minutes = (times.hour * 60 + times.minute).to_numpy().astype(float)
        self.hour = times.hour.to_numpy()


def temperature(spec, cal, extra=None):
    """Air temperature (°C) at every step; extra: a per-step array of added degrees (heatwaves)."""
    w = spec["weather"]
    rng = np.random.default_rng(int(spec["seed"]) + 101)
    days = cal.times.normalize().unique()
    doy = days.day_of_year.to_numpy()
    mid = np.array([15 + 30.4 * i for i in range(12)])                       # mid-month day of year
    means = np.asarray(w["monthly_mean_c"], dtype=float)
    daily = np.interp(doy, np.concatenate([mid - 365, mid, mid + 365]), np.tile(means, 3))
    phi = w["day_to_day_persistence"]
    anomaly = lfilter([w["day_to_day_c"] * math.sqrt(1 - phi ** 2)], [1, -phi], rng.standard_normal(len(days)))
    per_day = pd.Series(daily + anomaly, index=days)
    base = per_day.reindex(cal.times.normalize()).to_numpy()
    amp = np.array([w["daily_amplitude_c"][s] for s in cal.season])
    t = base + amp * np.cos(2 * np.pi * (cal.minutes / 60.0 - 15) / 24)
    return t + (0 if extra is None else extra)


@lru_cache(maxsize=1)
def residential_table():
    p = pd.read_csv(PROFILES)
    b = p[p["kind"] == "base"].sort_values("hour")
    base = {(s, d): g["value"].to_numpy() for (s, d), g in b.groupby(["season", "daytype"])}
    cool = p[p["kind"] == "cooling_kw_per_c"].sort_values("hour")["value"].to_numpy()
    return base, cool


def unit_profiles(spec, cal, temp, factors=None):
    """kW of one typical consumer of each type at every step. factors: {type: per-step multiplier array}."""
    d = spec["demand"]
    cdd = np.clip(temp - COOLING_BASE_C, 0, None)
    out = {}
    base, cool = residential_table()
    res = np.empty(len(cal.times))
    for (s, dt_), vals in base.items():
        sel = (cal.season == s) & (cal.weekend == (dt_ == "weekend"))
        res[sel] = _hourly_to_steps(vals, cal.minutes[sel])
    res = d["residential"]["scale"] * (res + _hourly_to_steps(cool, cal.minutes) * cdd)
    out["residential"] = res
    for kind in ("commercial", "industrial"):
        c = d[kind]
        shape = _hourly_to_steps(c["weekday"], cal.minutes)
        factor = np.where(cal.daytype == "saturday", c["saturday_factor"],
                          np.where(cal.daytype == "sunday", c["sunday_factor"], 1.0))
        open_ = (cal.minutes >= 8 * 60) & (cal.minutes < 21 * 60) if kind == "commercial" else np.ones(len(cal.times), bool)
        out[kind] = shape * factor + c["cooling_kw_per_c"] * cdd * open_ * factor
    share = d["mixed"]["residential_share"]
    out["mixed"] = share * out["residential"] + (1 - share) * out["commercial"]
    for kind, f in (factors or {}).items():
        out[kind] = out[kind] * f
    return out


def ev_profile(ev, cal):
    """Expected kW of one car charging, at every step: arrivals ~ normal(mean, sd) hours, charging for
    energy / charger power hours at full power."""
    dur = ev["energy_kwh"] / ev["charger_kw"]
    mu, sd = ev["arrival_h"]["mean"], ev["arrival_h"]["sd"]
    h = cal.minutes / 60.0
    phi = lambda x: 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))  # noqa: E731
    share = np.zeros_like(h)
    for shift in (-24, 0, 24):                                               # charging that runs past midnight
        share += phi((h + shift - mu) / sd) - phi((h + shift - dur - mu) / sd)
    return ev["charger_kw"] * np.clip(share, 0, 1)


def _in_dates(cal, e):
    d = cal.times.normalize()
    sel = (d >= pd.Timestamp(e["from"])) & (d <= pd.Timestamp(e["to"]))
    if "hours" in e:
        h0, h1 = e["hours"]
        sel &= (cal.minutes >= h0 * 60) & (cal.minutes < h1 * 60)
    return np.asarray(sel)


class CityDemand:
    def __init__(self, name, consumers="file"):
        """consumers: "file" (the generated consumers.csv), a DataFrame, or None (profiles only, for the generator)."""
        self.dir = city_dir(name)
        self.spec = load_spec(self.dir)
        self.times = year_times(self.spec)
        self.cal = Calendar(self.spec, self.times)
        if isinstance(consumers, str):
            path = self.dir / "consumers.csv"
            consumers = pd.read_csv(path) if path.exists() else None
        self.consumers = consumers
        self._cache = {}
        if consumers is not None:
            self.transformers = sorted(consumers["transformer"].unique())
            sizes = consumers.groupby(["transformer", "type"])["size"].sum().unstack(fill_value=0.0)
            self.sizes = sizes.reindex(index=self.transformers, columns=list(TYPES), fill_value=0.0)
            rng = np.random.default_rng(int(self.spec["seed"]) + 202)
            d = self.spec["demand"]
            phi = d["noise_persistence"]
            z = rng.standard_normal((len(self.times), len(self.transformers)))
            self.noise = lfilter([d["noise"] * math.sqrt(1 - phi ** 2)], [1, -phi], z, axis=0)

    def always_events(self):
        return [n for n, e in (self.spec.get("events") or {}).items() if e.get("always")]

    def _event_parts(self, events):
        evs = self.spec.get("events") or {}
        names = list(dict.fromkeys(self.always_events() + list(events)))
        extra = np.zeros(len(self.times))
        factors = {}
        for n in names:
            e = evs[n]
            if e["kind"] == "heatwave":
                extra = extra + e["extra_c"] * _in_dates(self.cal, e)
            elif e["kind"] == "demand_change":
                sel = _in_dates(self.cal, e)
                for kind in e["consumers"]:
                    factors[kind] = factors.get(kind, np.ones(len(self.times))) * np.where(sel, e["factor"], 1.0)
        return names, extra, factors

    def units(self, events=()):
        """Typical-consumer kW per type, and the temperature, with these events (plus the always-on ones)."""
        names, extra, factors = self._event_parts(events)
        temp = temperature(self.spec, self.cal, extra)
        return unit_profiles(self.spec, self.cal, temp, factors), temp, names

    def matrix(self, events=()):
        """Load of every transformer (kW) at every 15 minutes of the year, with these events."""
        key = tuple(sorted(events))
        if key in self._cache:
            return self._cache[key][0]
        if self.consumers is None:
            raise RuntimeError("no consumers.csv yet: run python city/generate.py first")
        units, temp, names = self.units(events)
        U = np.vstack([units[k] for k in TYPES])                              # types x steps
        load = (self.sizes.to_numpy() @ U).T                                  # steps x transformers
        for n in names:
            e = self.spec["events"][n]
            if e["kind"] == "ev_charging":
                c = self.consumers
                cars = c[c["type"].isin(e["consumers"]) & (c["ev_rank"] < e["share"])].groupby("transformer").size()
                cars = cars.reindex(self.transformers, fill_value=0).to_numpy()
                load = load + np.outer(ev_profile(e, self.cal), cars)
        load = np.clip(load * (1 + self.noise), 0, None)
        out = pd.DataFrame(load, index=self.times, columns=self.transformers)
        self._cache[key] = (out, pd.Series(temp, index=self.times))
        return out

    def temperature(self, events=()):
        """Air temperature (°C) at every step, with these events."""
        self.matrix(events)
        return self._cache[tuple(sorted(events))][1]
