"""Asset IDs follow the specification's scheme and map one-to-one onto the network's short names."""
import re

import pandas as pd
import pytest

from core import ids

PATTERN = {"substation": r"SUB-\d{3}", "feeder": r"FD-\d{3}", "transformer": r"DT-\d{3}", "switch": r"SW-\d{3}"}


@pytest.fixture(scope="module")
def reg():
    return ids.registry()


def test_registry_covers_every_asset(reg):
    assert reg["kind"].value_counts().to_dict() == {"switch": 13, "transformer": 6, "feeder": 2, "substation": 1}


def test_ids_are_unique_and_well_formed(reg):
    assert reg["asset_id"].is_unique
    for _, r in reg.iterrows():
        assert re.fullmatch(PATTERN[r["kind"]], r["asset_id"]), r["asset_id"]


def test_parents_exist(reg):
    known = set(reg["asset_id"])
    for _, r in reg.dropna(subset=["parent_id"]).iterrows():
        assert r["parent_id"] in known, r["asset_id"]
    tx = reg[reg["kind"] == "transformer"].set_index("name")
    assert tx.loc["T1", "parent_id"] == "FD-001" and tx.loc["T6", "parent_id"] == "FD-002"
    sw = reg[reg["kind"] == "switch"].set_index("name")
    assert sw.loc["SW3A", "parent_id"] == "DT-003"          # the alt switch carries T3's block b
    assert pd.isna(sw.loc["TIE1", "parent_id"])


def test_transformer_round_trip():
    for name, asset in ids.transformer_ids().items():
        assert ids.transformer_id(name) == asset
        assert ids.transformer_name(asset) == name
    assert ids.transformer_id("T1") == "DT-001"
    with pytest.raises(KeyError):
        ids.transformer_name("DT-999")


def test_switch_and_feeder_ids():
    assert ids.switch_id("SW1H") == "SW-001"
    assert ids.switch_id("TIE1") == "SW-013"
    assert ids.feeder_id("B") == "FD-002"


def test_meter_ids():
    assert ids.meter_id(1) == "MTR-00001"
    assert ids.meter_id(97) == "MTR-00097"
    for bad in (0, -1, 1.5, "3"):
        with pytest.raises(ValueError):
            ids.meter_id(bad)


def test_add_transformer_ids_inserts_after_name():
    df = pd.DataFrame({"timestamp": [1, 2], "transformer_id": ["T1", "T6"], "kw": [1.0, 2.0]})
    out = ids.add_transformer_ids(df)
    assert list(out.columns) == ["timestamp", "transformer_id", "asset_id", "kw"]
    assert out["asset_id"].tolist() == ["DT-001", "DT-006"]
    with pytest.raises(KeyError, match="T9"):
        ids.add_transformer_ids(pd.DataFrame({"transformer_id": ["T9"]}))
