"""Configuration files: they load, invalid values are rejected, and grid.yaml agrees with network.dss."""
import re
import shutil

import opendssdirect as dss
import pytest
import yaml

import engine as e
from core import config
from core.config import ConfigError


def test_all_config_files_are_valid():
    config.reload()
    for load in (config.grid, config.operations, config.scenarios, config.forecasting):
        assert isinstance(load(), dict)


# ------------------------------------------------------------------ rejection of bad values
@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    """A writable copy of config/; edit(name, fn) changes one file. The real config is restored after."""
    shutil.copytree(config.config_dir(), tmp_path / "config")
    monkeypatch.setenv("LFDR_CONFIG_DIR", str(tmp_path / "config"))
    config.reload()

    def edit(name, fn):
        path = tmp_path / "config" / f"{name}.yaml"
        data = yaml.safe_load(path.read_text())
        fn(data)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        config.reload()

    yield edit
    monkeypatch.delenv("LFDR_CONFIG_DIR")
    config.reload()


BAD_EDITS = [
    ("operations", lambda d: d["limits"].update(overload_threshold_pct=105), config.operations,
     "must not exceed transformer_hard_limit_pct"),
    ("operations", lambda d: d["limits"].update(voltage_min_pu=1.10), config.operations,
     "voltage_min_pu must be below voltage_max_pu"),
    ("operations", lambda d: d["limits"].update(line_limit_pct="high"), config.operations, "must be a number"),
    ("grid", lambda d: d["transformers"]["T1"].update(id="T-1"), config.grid, "not a valid transformer ID"),
    ("grid", lambda d: d["transformers"]["T2"].update(id="DT-001"), config.grid, "duplicate IDs"),
    ("grid", lambda d: d["moves"]["T1"].update(to="T1"), config.grid, "'to' must be another transformer"),
    ("grid", lambda d: d["moves"]["T1"].update(home="SW1A"), config.grid, "home must be an lv_home switch"),
    ("grid", lambda d: d["load_split"].update(movable_fraction=1.5), config.grid, "above the allowed maximum"),
    ("scenarios", lambda d: d["benchmark"][1]["rules"][0].update(transformers=["T9"]), config.scenarios,
     "unknown transformer T9"),
    ("scenarios", lambda d: d["benchmark"][2]["rules"][0].update(hours=[22, 18]), config.scenarios,
     "hours must be [start, end)"),
    ("scenarios", lambda d: d["benchmark"][1].update(name=d["benchmark"][0]["name"]), config.scenarios,
     "names must be unique"),
    ("forecasting", lambda d: d["split"].update(train=0.9, val=0.2), config.forecasting,
     "must leave room for a test split"),
    ("forecasting", lambda d: d["data"].update(start="1 June 2019"), config.forecasting, "must be a date"),
    ("forecasting", lambda d: d["clusters"].pop("T6"), config.forecasting, "cover exactly the grid transformers"),
]


@pytest.mark.parametrize("name, change, loader, message", BAD_EDITS, ids=[m for *_, m in BAD_EDITS])
def test_invalid_values_are_rejected(temp_config, name, change, loader, message):
    temp_config(name, change)
    with pytest.raises(ConfigError, match=re.escape(message)):
        loader()


def test_unquoted_dates_are_accepted(temp_config, tmp_path):
    path = tmp_path / "config" / "forecasting.yaml"
    path.write_text(path.read_text().replace('start: "2019-06-01"', "start: 2019-06-01"))
    config.reload()
    assert config.forecasting()["data"]["start"] == "2019-06-01"


# ------------------------------------------------------------------ grid.yaml <-> network.dss
@pytest.fixture(scope="module")
def network():
    e.load_network()
    return config.grid()


def test_transformers_match_network(network):
    assert sorted(n.upper() for n in dss.Transformers.AllNames()) == sorted(network["transformers"])
    for name, tx in network["transformers"].items():
        dss.Transformers.Name(name)
        assert dss.Transformers.kVA() == pytest.approx(tx["rating_kva"]), name
        dss.Circuit.SetActiveElement(f"Transformer.{name}")
        assert dss.CktElement.BusNames()[0].split(".")[0].upper() == tx["mv_bus"], name


def test_loads_exist_in_network(network):
    loads = {n.upper() for n in dss.Loads.AllNames()}
    for name, tx in network["transformers"].items():
        assert tx["fixed_load"].upper() in loads and tx["movable_load"].upper() in loads, name


def test_switches_match_network(network):
    for name, sw in network["switches"].items():
        dss.Lines.Name(name)
        assert dss.Lines.Name().upper() == name, f"{name} missing from network.dss"
        assert dss.Lines.IsSwitch(), f"{name} is not a switch in network.dss"
        dss.Circuit.SetActiveElement(f"Line.{name}")
        assert dss.CktElement.IsOpen(1, 0) == (sw["normal"] == "open"), name


def test_moves_connect_the_right_buses(network):
    """Closing a move's alt switch must join the block's bus to the target transformer's LV bus."""
    lv_bus = {}
    for name in network["transformers"]:
        dss.Circuit.SetActiveElement(f"Transformer.{name}")
        lv_bus[name] = dss.CktElement.BusNames()[1].split(".")[0].upper()
    for tx, m in network["moves"].items():
        dss.Circuit.SetActiveElement(f"Line.{m['home']}")
        home_buses = {b.split(".")[0].upper() for b in dss.CktElement.BusNames()}
        dss.Circuit.SetActiveElement(f"Line.{m['alt']}")
        alt_buses = {b.split(".")[0].upper() for b in dss.CktElement.BusNames()}
        assert lv_bus[tx] in home_buses, f"{m['home']} does not reach {tx}'s LV bus"
        assert lv_bus[m["to"]] in alt_buses, f"{m['alt']} does not reach {m['to']}'s LV bus"
        assert home_buses & alt_buses, f"{m['home']} and {m['alt']} do not share the block's bus"


def test_feeders_partition_the_transformers(network):
    listed = [t for fd in network["feeders"].values() for t in fd["transformers"]]
    assert sorted(listed) == sorted(network["transformers"])
