"""Load and validate the YAML files in project/config/.

    from core import config
    limits = config.operations()["limits"]

Each file is read once and checked; an invalid value raises ConfigError naming the file and the key,
so a typo stops the program instead of silently changing a result.

Set the environment variable LFDR_CONFIG_DIR to use a different config folder (tests and, later,
alternative city configurations do this).
"""
import datetime as _dt
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import yaml

from core.paths import PROJECT_ROOT, project_path

ID_PATTERNS = {                                 # kind: (pattern, example)
    "substation": (r"SUB-\d{3}", "SUB-001"),
    "feeder": (r"FD-\d{3}", "FD-001"),
    "transformer": (r"DT-\d{3}", "DT-001"),
    "switch": (r"SW-\d{3}", "SW-001"),
}


class ConfigError(ValueError):
    pass


def config_dir():
    return Path(os.environ.get("LFDR_CONFIG_DIR", PROJECT_ROOT / "config"))


def use_city(name):
    """Point the whole project at a synthetic city's generated configuration (cities/<name>/config), for this
    process and the processes it starts. None or "" keeps the classic eMARC configuration."""
    if not name:
        return None
    folder = PROJECT_ROOT / "cities" / name / "config"
    if not (folder / "grid.yaml").exists():
        raise ConfigError(f"no generated city at {folder}: run python city/generate.py {name}")
    os.environ["LFDR_CONFIG_DIR"] = str(folder)
    reload()
    return folder


def city_arg(argv=None):
    """Read `--city NAME` from the command line (also after Streamlit's `--`) and apply it."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--city" in argv:
        i = argv.index("--city")
        if i + 1 < len(argv):
            return use_city(argv[i + 1])
    return None


def reload():
    """Forget cached files (after changing LFDR_CONFIG_DIR or editing a file at runtime)."""
    _read.cache_clear()
    for fn in (grid, operations, scenarios, forecasting, storage, simulation):
        fn.cache_clear()


@lru_cache(maxsize=None)
def _read(name):
    path = config_dir() / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: expected a mapping at the top level")
    return data


# ------------------------------------------------------------------ small checking helpers
def _need(cond, file, msg):
    if not cond:
        raise ConfigError(f"{file}.yaml: {msg}")


def _get(d, file, *keys):
    cur = d
    for i, k in enumerate(keys):
        _need(isinstance(cur, dict) and k in cur, file, f"missing key '{'.'.join(map(str, keys[:i + 1]))}'")
        cur = cur[k]
    return cur


def _number(d, file, *keys, low=None, high=None, low_open=False):
    v = _get(d, file, *keys)
    name = ".".join(keys)
    _need(isinstance(v, (int, float)) and not isinstance(v, bool), file, f"'{name}' must be a number, got {v!r}")
    if low is not None:
        _need(v > low if low_open else v >= low, file, f"'{name}' = {v} is below the allowed minimum {low}")
    if high is not None:
        _need(v <= high, file, f"'{name}' = {v} is above the allowed maximum {high}")
    return v


def _id(value, kind, file, where):
    pattern, example = ID_PATTERNS[kind]
    _need(isinstance(value, str) and re.fullmatch(pattern, value), file,
          f"{where}: '{value}' is not a valid {kind} ID (expected like {example})")


def move_options(g):
    """[(transformer, {home, alt, to}), ...] from grid.yaml 'moves', where each entry may be one option
    (a mapping) or several (a list). Order follows the file."""
    out = []
    for tx, m in g.get("moves", {}).items():
        for opt in (m if isinstance(m, list) else [m]):
            out.append((tx, opt))
    return out


def substations(g):
    """{name: {id, bus, ...}}: the classic file's single `substation`, or a generated city's `substations`.
    Each substation is an 11 kV source in the OpenDSS model."""
    if "substations" in g:
        return {n: {"bus": n, **s} for n, s in g["substations"].items()}
    s = g["substation"]
    return {s["name"]: {"bus": s.get("bus", s["name"]), **s}}


def feeder_substation(g, feeder):
    """Name of the substation a feeder leaves from."""
    subs = substations(g)
    return g["feeders"][feeder].get("substation") or next(iter(subs))


# ------------------------------------------------------------------ files
@lru_cache(maxsize=None)
def grid():
    f = "grid"
    g = _read(f)
    _need(project_path(_get(g, f, "network_file")).exists(), f, f"network_file {g['network_file']} not found")
    _need(("substation" in g) != ("substations" in g), f, "give either 'substation' (one) or 'substations' (several)")
    subs = substations(g)
    for name, s in subs.items():
        _id(s.get("id"), "substation", f, f"substation {name}")

    txs = _get(g, f, "transformers")
    _need(isinstance(txs, dict) and txs, f, "'transformers' must be a non-empty mapping")
    feeders = _get(g, f, "feeders")
    categories = _get(g, f, "load_categories")
    for cat, c in categories.items():
        _number(c, f, "power_factor", low=0, high=1, low_open=True)
    for name, fd in feeders.items():
        _id(fd.get("id"), "feeder", f, f"feeder {name}")
        _need(isinstance(fd.get("head_line"), str), f, f"feeder {name}: 'head_line' is required")
        if len(subs) > 1 or "substation" in fd:
            _need(fd.get("substation") in subs, f, f"feeder {name}: unknown substation {fd.get('substation')!r}")
        for t in fd.get("transformers", []):
            _need(t in txs, f, f"feeder {name} lists unknown transformer {t}")
    for t, tx in txs.items():
        _id(tx.get("id"), "transformer", f, f"transformer {t}")
        _need(tx.get("feeder") in feeders, f, f"transformer {t}: unknown feeder {tx.get('feeder')}")
        _need(t in feeders[tx["feeder"]]["transformers"], f, f"transformer {t} missing from feeder {tx['feeder']}")
        _number(tx, f, "rating_kva", low=0, low_open=True)
        _need(tx.get("category") in categories, f, f"transformer {t}: unknown load category {tx.get('category')!r}")
        for k in ("mv_bus", "fixed_load", "movable_load"):
            _need(isinstance(tx.get(k), str), f, f"transformer {t}: '{k}' is required")

    frac = _number(g, f, "load_split", "movable_fraction", low=0, high=1, low_open=True)
    _need(frac < 1, f, "load_split.movable_fraction must be below 1")

    switches = _get(g, f, "switches")
    for s, sw in switches.items():
        _id(sw.get("id"), "switch", f, f"switch {s}")
        _need(sw.get("kind") in ("lv_home", "lv_alt", "mv_tie", "mv_sectionaliser"), f,
              f"switch {s}: unknown kind {sw.get('kind')}")
        _need(sw.get("normal") in ("open", "closed"), f, f"switch {s}: normal must be open or closed")

    _get(g, f, "moves")
    seen = set()
    for t, m in move_options(g):
        _need(t in txs, f, f"move for unknown transformer {t}")
        _need(isinstance(m, dict), f, f"move {t}: each option needs home, alt and to")
        _need((t, m.get("to")) not in seen, f, f"move {t}: two options lead to {m.get('to')}")
        seen.add((t, m.get("to")))
        _need(m.get("to") in txs and m["to"] != t, f, f"move {t}: 'to' must be another transformer")
        _need(switches.get(m.get("home"), {}).get("kind") == "lv_home", f, f"move {t}: home must be an lv_home switch")
        _need(switches.get(m.get("alt"), {}).get("kind") == "lv_alt", f, f"move {t}: alt must be an lv_alt switch")

    for name, mv in (g.get("mv_transfers") or {}).items():
        _need(switches.get(mv.get("open"), {}).get("kind") == "mv_sectionaliser", f,
              f"mv_transfers {name}: 'open' must be an mv_sectionaliser switch")
        _need(switches.get(mv.get("close"), {}).get("kind") == "mv_tie", f,
              f"mv_transfers {name}: 'close' must be an mv_tie switch")

    all_ids = ([s["id"] for s in subs.values()] + [fd["id"] for fd in feeders.values()]
               + [tx["id"] for tx in txs.values()] + [sw["id"] for sw in switches.values()])
    dup = {i for i in all_ids if all_ids.count(i) > 1}
    _need(not dup, f, f"duplicate IDs: {sorted(dup)}")
    return g


@lru_cache(maxsize=None)
def operations():
    f = "operations"
    o = _read(f)
    thr = _number(o, f, "limits", "overload_threshold_pct", low=0, low_open=True)
    hard = _number(o, f, "limits", "transformer_hard_limit_pct", low=0, low_open=True)
    _need(thr <= hard, f, "overload_threshold_pct must not exceed transformer_hard_limit_pct")
    vmin = _number(o, f, "limits", "voltage_min_pu", low=0, low_open=True)
    vmax = _number(o, f, "limits", "voltage_max_pu", low=0, low_open=True)
    _need(vmin < vmax, f, "voltage_min_pu must be below voltage_max_pu")
    _number(o, f, "limits", "line_limit_pct", low=0, low_open=True)
    _number(o, f, "rerouting", "min_improvement_pct", low=0)
    _number(o, f, "rerouting", "tie_tolerance_pct", low=0)
    _need(o["rerouting"].get("candidate_scope", "all") in ("all", "relevant"), f,
          "rerouting.candidate_scope must be all or relevant")
    _need(isinstance(_get(o, f, "rerouting", "use_mv_transfers"), bool), f, "rerouting.use_mv_transfers must be true or false")
    elev = _number(o, f, "risk", "elevated_from_pct", low=0)
    _need(elev < thr, f, "risk.elevated_from_pct must be below the overload threshold")
    _number(o, f, "risk", "approaching_margin_pct", low=0)
    _number(o, f, "risk", "voltage_margin_pu", low=0, high=0.1)
    _number(o, f, "risk", "line_approaching_pct", low=0, high=_get(o, f, "limits", "line_limit_pct"))
    _need(_get(o, f, "decisions", "default_mode") in ("monitor", "recommend", "auto"), f,
          "decisions.default_mode must be monitor, recommend or auto")
    med = _number(o, f, "decisions", "confidence_medium", low=0, high=1)
    high = _number(o, f, "decisions", "confidence_high", low=0, high=1)
    _need(med < high, f, "decisions.confidence_medium must be below confidence_high")
    lo = _number(o, f, "dashboard", "threshold_slider", "min", low=0)
    hi = _number(o, f, "dashboard", "threshold_slider", "max", low=0)
    _need(lo <= thr <= hi, f, "overload_threshold_pct must lie inside dashboard.threshold_slider")
    _number(o, f, "dashboard", "amber_from_pct", low=0)
    return o


@lru_cache(maxsize=None)
def scenarios():
    f = "scenarios"
    s = _read(f)
    txs = set(grid()["transformers"])

    def check_rule(r, where, partial=False):
        _need(r.get("type") in ("scale", "add"), f, f"{where}: type must be scale or add")
        if not partial:
            field = "factor" if r["type"] == "scale" else "kw"
            _need(isinstance(r.get(field), (int, float)), f, f"{where}: '{field}' must be a number")
            if r["type"] == "scale":
                _need(r["factor"] > 0, f, f"{where}: factor must be positive")
        for t in r.get("transformers", []) or []:
            _need(t in txs, f, f"{where}: unknown transformer {t}")
        if "hours" in r:
            h = r["hours"]
            _need(isinstance(h, list) and len(h) == 2 and 0 <= h[0] < h[1] <= 24, f,
                  f"{where}: hours must be [start, end) with 0 <= start < end <= 24")

    for group in ("benchmark",):
        names = [sc["name"] for sc in _get(s, f, group)]
        _need(len(names) == len(set(names)), f, f"{group}: scenario names must be unique")
        for sc in s[group]:
            for i, r in enumerate(sc.get("rules", [])):
                check_rule(r, f"{group} '{sc['name']}' rule {i + 1}")
            ev = sc.get("events", [])
            _need(isinstance(ev, list) and all(isinstance(e, str) for e in ev), f,
                  f"{group} '{sc['name']}': events must be a list of event names (synthetic city)")
    _number(s, f, "quick", "base_kw", low=0, low_open=True)
    for sc in _get(s, f, "quick", "scenarios"):
        for i, r in enumerate(sc.get("rules", [])):
            check_rule(r, f"quick '{sc['name']}' rule {i + 1}")
    for p in _get(s, f, "dashboard"):
        _need("label" in p, f, "every dashboard preset needs a label")
        if "rule" in p:
            check_rule(p["rule"], f"dashboard '{p['label']}'", partial=True)
            sl = p.get("slider") or {}
            _need({"key", "label", "field", "min", "max", "step", "default"} <= sl.keys(), f,
                  f"dashboard '{p['label']}': slider needs key, label, field, min, max, step, default")
            _need(sl["min"] <= sl["default"] <= sl["max"], f, f"dashboard '{p['label']}': default outside slider range")
            if "choose_transformer" in p:
                _need(p["choose_transformer"].get("default") in txs, f,
                      f"dashboard '{p['label']}': choose_transformer.default must be a transformer")
    return s


def _check_intervals(c, f):
    lo = _number(c, f, "intervals", "lower_quantile", low=0, high=1, low_open=True)
    hi = _number(c, f, "intervals", "upper_quantile", low=0, high=1)
    _need(lo < hi < 1, f, "intervals: lower_quantile < upper_quantile < 1")
    _need(_get(c, f, "intervals", "method") in ("rolling", "static"), f, "intervals.method must be rolling or static")
    _number(c, f, "intervals", "window_days", low=1)
    hb = _number(c, f, "intervals", "hour_blocks", low=1, high=24)
    _need(24 % hb == 0, f, "intervals.hour_blocks must divide 24")
    _need(isinstance(_get(c, f, "intervals", "use_upper_for_risk"), bool), f, "intervals.use_upper_for_risk must be true/false")


@lru_cache(maxsize=None)
def forecasting():
    f = "forecasting"
    c = _read(f)
    if c.get("source") == "city":                       # a synthetic city: forecasts are made at run time
        _check_intervals(c, f)
        return c
    for k in ("start", "end"):
        v = _get(c, f, "data", k)
        if isinstance(v, _dt.date):                     # unquoted YAML dates arrive as date objects
            c["data"][k] = v = v.isoformat()
        try:
            _dt.date.fromisoformat(str(v))
        except ValueError:
            raise ConfigError(f"{f}.yaml: data.{k} must be a date like 2019-06-01, got {v!r}") from None
    _need(c["data"]["start"] < c["data"]["end"], f, "data.start must be before data.end")
    _number(c, f, "data", "min_blocks_per_day", low=1, high=96)
    _number(c, f, "data", "min_complete_share", low=0, high=1)
    for k in ("max_gap_steps", "cluster_gap_hours", "weather_gap_hours"):
        _number(c, f, "data", k, low=0)

    txs = set(grid()["transformers"])
    clusters = _get(c, f, "clusters")
    _need(set(clusters) == txs, f, f"clusters must cover exactly the grid transformers {sorted(txs)}")
    regions = _get(c, f, "regions")
    for t, cl in clusters.items():
        _need(cl.get("region") in regions, f, f"cluster {t}: region {cl.get('region')!r} has no coordinates")
        _need(cl.get("half") in (None, 0, 1), f, f"cluster {t}: half must be 0, 1 or omitted")
    for r, ll in regions.items():
        _need(isinstance(ll, list) and len(ll) == 2 and -90 <= ll[0] <= 90 and -180 <= ll[1] <= 180, f,
              f"region {r}: expected [latitude, longitude]")

    _number(c, f, "features", "cdd_base_c")
    tr = _number(c, f, "split", "train", low=0, high=1, low_open=True)
    va = _number(c, f, "split", "val", low=0, high=1, low_open=True)
    _need(tr + va < 1, f, "split.train + split.val must leave room for a test split")
    _get(c, f, "model", "xgboost")
    _number(c, f, "scaling", "target_peak_kw", low=0, low_open=True)
    _check_intervals(c, f)
    _need(isinstance(_get(c, f, "v2", "model_version"), str), f, "v2.model_version is required")
    for k in ("n_estimators", "min_samples_leaf"):
        _number(c, f, "v2", "random_forest", k, low=1)
    _number(c, f, "v2", "random_forest", "max_features", low=0, high=1, low_open=True)
    qs = _get(c, f, "v2", "quantiles")
    _need(isinstance(qs, list) and len(qs) >= 2 and qs == sorted(qs) and all(0 < q < 1 for q in qs), f,
          "v2.quantiles must be an increasing list of numbers between 0 and 1")
    _number(c, f, "v2", "retrain_every_days", low=1)
    return c


@lru_cache(maxsize=None)
def storage():
    f = "storage"
    c = _read(f)
    _need(isinstance(_get(c, f, "url"), str) and "://" in c["url"], f, "url must be a database URL like sqlite:///data/twin.db")
    _number(c, f, "batch_size", low=1)
    for k in ("measurements_days", "alerts_days"):
        _number(c, f, "retention", k, low=0)
    for k in ("forecasts_days", "meter_readings_days"):
        c["retention"].setdefault(k, 0)
        _number(c, f, "retention", k, low=0)
    _need(isinstance(_get(c, f, "forecast_import", "model_version"), str), f, "forecast_import.model_version is required")
    _need(isinstance(_get(c, f, "forecast_import", "measurement_source"), str), f,
          "forecast_import.measurement_source is required")
    return c


TIME_RE = r"\d{4}-\d{2}-\d{2}( \d{2}:\d{2}(:\d{2})?)?"


@lru_cache(maxsize=None)
def simulation():
    f = "simulation"
    c = _read(f)
    for k in ("start", "end"):
        v = _get(c, f, "clock", k)
        _need(v in (f"test_{k}",) or (isinstance(v, str) and re.fullmatch(TIME_RE, v)), f,
              f"clock.{k} must be test_{k} or a time like 2020-02-21 06:00, got {v!r}")
    step = _number(c, f, "clock", "step_minutes", low=1)
    _need(float(step).is_integer() and 1440 % int(step) == 0, f,
          f"clock.step_minutes must divide a day into whole steps (e.g. 15, 30, 60), got {step}")
    speed = _get(c, f, "clock", "speed")
    _need(speed in ("max", "accelerated") or (isinstance(speed, (int, float)) and not isinstance(speed, bool)
                                               and speed > 0), f,
          f"clock.speed must be a positive number, 'accelerated' or 'max', got {speed!r}")
    _number(c, f, "clock", "accelerated_factor", low=0, low_open=True)
    _need(isinstance(_get(c, f, "clock", "catch_up"), bool), f, "clock.catch_up must be true or false")
    _number(c, f, "clock", "behind_warning_s", low=0)
    o = _get(c, f, "orchestrator")
    _need(isinstance(o.get("controller"), str) and o["controller"], f, "orchestrator.controller is required")
    names = [s["name"] for s in scenarios()["benchmark"]]
    _need(o.get("scenario") in names, f, f"orchestrator.scenario must be one of {names}, got {o.get('scenario')!r}")
    _need(o.get("mode") in ("monitor", "recommend", "auto"), f,
          f"orchestrator.mode must be monitor, recommend or auto, got {o.get('mode')!r}")
    _need(o.get("forecast_source") in ("precomputed", "live", "persistence"), f,
          f"orchestrator.forecast_source must be precomputed, live or persistence, got {o.get('forecast_source')!r}")
    _need(c.get("city") is None or isinstance(c["city"], str), f, "city must be the name of a folder in cities/")
    _need((c.get("city") is None) == (o["forecast_source"] != "persistence"), f,
          "forecast_source persistence is for a synthetic city (and a city uses it)")
    _need(isinstance(o.get("pause_on_recommendation"), bool), f, "orchestrator.pause_on_recommendation must be true or false")
    for k in ("poll_interval_s", "heartbeat_s", "stale_after_s"):
        _number(c, f, "service", k, low=0, low_open=True)
    _need(c["service"]["stale_after_s"] > c["service"]["heartbeat_s"], f,
          "service.stale_after_s must be longer than service.heartbeat_s")
    _need(isinstance(_get(c, f, "service", "resume_on_start"), bool), f, "service.resume_on_start must be true or false")
    months = [m for ms in _get(c, f, "seasons").values() for m in (ms or [])]
    _need(sorted(months) == list(range(1, 13)), f, "seasons must list every month 1-12 exactly once")
    return c


def database_url():
    """The database URL: LFDR_DB_URL if set (read on every call, so tests can switch it), else storage.yaml.
    A relative SQLite path is resolved inside the project/ folder."""
    url = os.environ.get("LFDR_DB_URL") or storage()["url"]
    prefix = "sqlite:///"
    path = url[len(prefix):]
    if url.startswith(prefix) and path != ":memory:" and not Path(path).is_absolute() and not path.startswith("/"):
        # forward slashes on every platform (SQLAlchemy accepts sqlite:///C:/... on Windows)
        url = prefix + (PROJECT_ROOT / path).as_posix()
    return url
