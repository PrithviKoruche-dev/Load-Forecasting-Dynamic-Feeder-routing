"""Stress scenarios: the one implementation used by the benchmark, the quick script and the dashboard.

Definitions live in config/scenarios.yaml. A scenario is a list of rules applied in order:
    scale: load * factor        add: load + kw
each optionally limited to some transformers and/or to clock hours [start, end).
"""
from dataclasses import dataclass, field

from core import config


@dataclass(frozen=True)
class Rule:
    type: str                                   # "scale" or "add"
    factor: float = 1.0
    kw: float = 0.0
    transformers: tuple = None                  # None = every transformer
    hours: tuple = None                         # None = all day, else (start, end)

    @classmethod
    def from_dict(cls, d):
        return cls(type=d["type"], factor=d.get("factor", 1.0), kw=d.get("kw", 0.0),
                   transformers=tuple(d["transformers"]) if d.get("transformers") else None,
                   hours=tuple(d["hours"]) if d.get("hours") else None)

    def applies(self, tx, ts=None):
        if self.transformers is not None and tx not in self.transformers:
            return False
        if self.hours is not None:
            if ts is None:
                raise ValueError("this rule depends on the hour of day, so a timestamp is required")
            return self.hours[0] <= ts.hour < self.hours[1]
        return True

    def apply(self, kw):
        return kw * self.factor if self.type == "scale" else kw + self.kw

    def describe(self):
        what = f"x{self.factor:g}" if self.type == "scale" else f"+{self.kw:g} kW"
        where = ", ".join(self.transformers) if self.transformers else "all transformers"
        when = f", {self.hours[0]:02d}:00-{self.hours[1]:02d}:00" if self.hours else ""
        return f"{what} on {where}{when}"


@dataclass(frozen=True)
class Scenario:
    name: str
    rules: tuple = field(default_factory=tuple)

    def adjust(self, tx, ts, kw):
        """Load of transformer `tx` at time `ts` after every rule that applies."""
        for r in self.rules:
            if r.applies(tx, ts):
                kw = r.apply(kw)
        return kw

    def loads(self, base_kw_by_tx, ts=None):
        """{tx: kW} -> {tx: adjusted kW} as plain floats."""
        return {tx: float(self.adjust(tx, ts, kw)) for tx, kw in base_kw_by_tx.items()}

    def describe(self):
        return "; ".join(r.describe() for r in self.rules) or "no change"


def _scenario(d):
    return Scenario(d["name"], tuple(Rule.from_dict(r) for r in d.get("rules", [])))


def benchmark():
    """The published S1-S5 scenarios (grid/run_results.py)."""
    return [_scenario(d) for d in config.scenarios()["benchmark"]]


def quick():
    """(base_kw, scenarios) for the flat-load quick check (grid/scenarios.py)."""
    q = config.scenarios()["quick"]
    return q["base_kw"], [_scenario(d) for d in q["scenarios"]]


@dataclass(frozen=True)
class Preset:
    """A dashboard scenario: one rule whose size (and optionally transformer) the operator picks."""
    label: str
    rule: dict = None
    slider: dict = None
    choose_transformer: dict = None

    def build(self, value=None, transformer=None):
        if self.rule is None:
            return Scenario(self.label)
        r = dict(self.rule)
        r[self.slider["field"]] = self.slider["default"] if value is None else value
        if self.choose_transformer is not None:
            r["transformers"] = [transformer or self.choose_transformer["default"]]
        return Scenario(self.label, (Rule.from_dict(r),))


def dashboard_presets():
    return [Preset(p["label"], p.get("rule"), p.get("slider"), p.get("choose_transformer"))
            for p in config.scenarios()["dashboard"]]
