"""Asset IDs in the research specification's scheme, mapped to the short names used in OpenDSS.

    ids.transformer_id("T1")      -> "DT-001"
    ids.transformer_name("DT-001") -> "T1"
    ids.registry()                -> one row per asset (substation, feeders, transformers, switches)
    ids.meter_id(1)               -> "MTR-00001"

Short names stay on the dashboard and inside network.dss; the IDs go into data files and the database.
"""
import pandas as pd

from core import config

METER_FORMAT = "MTR-{:05d}"


def _grid():
    return config.grid()


def transformer_id(name):
    return _grid()["transformers"][name]["id"]


def transformer_name(asset_id):
    for name, tx in _grid()["transformers"].items():
        if tx["id"] == asset_id:
            return name
    raise KeyError(asset_id)


def transformer_ids():
    """{short name: asset ID} in network order."""
    return {name: tx["id"] for name, tx in _grid()["transformers"].items()}


def switch_id(name):
    return _grid()["switches"][name]["id"]


def feeder_id(name):
    return _grid()["feeders"][name]["id"]


def meter_id(n):
    """n is 1-based."""
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"meter number must be a positive integer, got {n!r}")
    return METER_FORMAT.format(n)


def registry():
    """All network assets: asset_id, kind, name, parent_id, rating_kva (transformers), normal (switches)."""
    g = _grid()
    sub = g["substation"]
    rows = [{"asset_id": sub["id"], "kind": "substation", "name": sub["name"], "parent_id": None}]
    for name, fd in g["feeders"].items():
        rows.append({"asset_id": fd["id"], "kind": "feeder", "name": name, "parent_id": sub["id"]})
    for name, tx in g["transformers"].items():
        rows.append({"asset_id": tx["id"], "kind": "transformer", "name": name,
                     "parent_id": g["feeders"][tx["feeder"]]["id"], "rating_kva": tx["rating_kva"]})
    owner = {}                                          # LV switch -> transformer whose block b it carries
    for tx, m in config.move_options(g):
        owner[m["home"]] = owner[m["alt"]] = g["transformers"][tx]["id"]
    for name, sw in g["switches"].items():
        rows.append({"asset_id": sw["id"], "kind": "switch", "name": name, "parent_id": owner.get(name),
                     "switch_kind": sw["kind"], "normal": sw["normal"]})
    return pd.DataFrame(rows)


def add_transformer_ids(df, name_col="transformer_id", id_col="asset_id"):
    """Insert an asset-ID column right after the transformer short-name column."""
    out = df.copy()
    pos = out.columns.get_loc(name_col) + 1
    out.insert(pos, id_col, out[name_col].map(transformer_ids()))
    if out[id_col].isna().any():
        bad = sorted(out.loc[out[id_col].isna(), name_col].unique())
        raise KeyError(f"no asset ID configured for {bad}")
    return out
