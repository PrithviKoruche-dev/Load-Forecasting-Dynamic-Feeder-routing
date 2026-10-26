"""Build data/processed/clusters.parquet from the eMARC raw files.

Output: one row per (hourly timestamp, transformer) with
    timestamp, transformer_id, region, n_households, load_kw_raw, temp_c
load_kw_raw is the cluster's total hourly mean demand in kW BEFORE scaling to the grid's kVA rating.

Also writes data/processed/meter_registry.csv: one row per household kept, with its meter ID
(MTR-00001, ...) and the transformer (short name and DT-xxx ID) whose load it is part of.

The date window, cleaning rules, household-to-transformer clusters and weather coordinates come from
config/forecasting.yaml.
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # project/ -> import core
from core import config, ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

CFG = config.forecasting()
START, END = CFG["data"]["start"], CFG["data"]["end"]
MIN_BLOCKS_PER_DAY = CFG["data"]["min_blocks_per_day"]      # a day is "complete" if >= this many of 96 blocks
MIN_COMPLETE_SHARE = CFG["data"]["min_complete_share"]      # household kept if this share of days is complete
MAX_GAP_STEPS = CFG["data"]["max_gap_steps"]                # interpolate household gaps up to this many 15-min steps
CLUSTER_GAP_HOURS = CFG["data"]["cluster_gap_hours"]
WEATHER_GAP_HOURS = CFG["data"]["weather_gap_hours"]

# transformer -> (region, which half of the region's households; None = all of them)
TX_MAP = {tx: (c["region"], c.get("half")) for tx, c in CFG["clusters"].items()}
COORDS = {region: tuple(ll) for region, ll in CFG["regions"].items()}   # approximate city centres for weather


def select_households():
    info = pd.read_excel(RAW / "Household-Deployment basic info.xlsx")
    main = info[info["Deployment type"].str.lower() == "mainline"]
    return main[["deployment_id", "household_id", "Region"]].drop_duplicates("deployment_id")


def read_blocks(deployment_ids):
    parts = []
    for ch in pd.read_csv(RAW / "eMARC load blocks.csv", chunksize=2_000_000):
        ch = ch[ch["deployment_id"].isin(deployment_ids)]
        if ch.empty:
            continue
        ch["date"] = pd.to_datetime(ch["date"], format="%m/%d/%Y")
        ch = ch[(ch["date"] >= START) & (ch["date"] <= END)]
        ch["timestamp"] = ch["date"] + pd.to_timedelta(ch["block"] * 15, unit="min")
        parts.append(ch[["deployment_id", "date", "timestamp", "Load (kW)"]])
    return pd.concat(parts, ignore_index=True)


def fetch_weather(region):
    """Hourly 2 m temperature from Open-Meteo (free, no key). Cached in data/raw/."""
    cache = RAW / f"weather_{region.replace(' ', '_')}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["timestamp"])
    lat, lon = COORDS[region]
    q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": START,
                                "end_date": END, "hourly": "temperature_2m",
                                "timezone": CFG["data"]["weather_timezone"]})
    url = "https://archive-api.open-meteo.com/v1/archive?" + q
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            js = json.load(r)
    except Exception as e:
        raise SystemExit(f"Weather download failed for {region}: {e}\n"
                         f"Check your internet connection and run again, or open this URL in a browser:\n{url}")
    w = pd.DataFrame({"timestamp": pd.to_datetime(js["hourly"]["time"]),
                      "temp_c": js["hourly"]["temperature_2m"]})
    w.to_csv(cache, index=False)
    return w


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dep = select_households()
    print("Mainline deployments:", len(dep))
    blocks = read_blocks(set(dep["deployment_id"]))
    print("rows in window:", len(blocks))

    # 1) keep households with >=95% complete days
    n_days = (pd.Timestamp(END) - pd.Timestamp(START)).days + 1
    per_day = blocks.groupby(["deployment_id", "date"]).size().rename("n").reset_index()
    complete = per_day[per_day["n"] >= MIN_BLOCKS_PER_DAY].groupby("deployment_id").size()
    keep = complete[complete >= MIN_COMPLETE_SHARE * n_days].index
    dep = dep[dep["deployment_id"].isin(keep)].sort_values("household_id")
    print("households kept:", len(dep), "\n", dep.groupby("Region").size().to_string())

    # 2) wide 15-min grid, short gaps interpolated
    grid = pd.date_range(START, pd.Timestamp(END) + pd.Timedelta(days=1), freq="15min", inclusive="left")
    wide = (blocks[blocks["deployment_id"].isin(keep)]
            .groupby(["timestamp", "deployment_id"])["Load (kW)"].mean().unstack()
            .reindex(grid))
    wide = wide.interpolate(limit=MAX_GAP_STEPS, limit_area="inside")

    # 3) hourly mean per household
    hourly = wide.resample("1h").mean()

    # 4) clusters -> transformers
    rows, meters = [], []
    for tx, (region, half) in TX_MAP.items():
        d = dep[dep["Region"] == region]
        dep_ids = d["deployment_id"].tolist()
        if half is not None:                      # split the region's households into two halves
            mid = len(dep_ids) // 2
            dep_ids = dep_ids[:mid] if half == 0 else dep_ids[mid:]
        mean_kw = hourly[dep_ids].mean(axis=1)    # mean over households with data that hour
        total = mean_kw * len(dep_ids)            # = sum, with missing households filled by the mean
        total = total.interpolate(limit=CLUSTER_GAP_HOURS, limit_area="inside")
        weather = fetch_weather(region).set_index("timestamp")["temp_c"].reindex(hourly.index)
        weather = weather.interpolate(limit=WEATHER_GAP_HOURS)
        meters.append(d[d["deployment_id"].isin(dep_ids)].assign(transformer=tx))
        rows.append(pd.DataFrame({"timestamp": hourly.index, "transformer_id": tx,
                                  "region": region, "n_households": len(dep_ids),
                                  "load_kw_raw": total.values, "temp_c": weather.values}))
    out = pd.concat(rows, ignore_index=True)

    # 5) report + save
    print("\nper transformer:")
    g = out.groupby("transformer_id").agg(region=("region", "first"), households=("n_households", "first"),
                                          mean_kw=("load_kw_raw", "mean"), peak_kw=("load_kw_raw", "max"),
                                          missing_load=("load_kw_raw", lambda s: int(s.isna().sum())),
                                          missing_temp=("temp_c", lambda s: int(s.isna().sum())))
    print(g.round(2).to_string())
    out.to_parquet(OUT / "clusters.parquet", index=False)
    print(f"\nSaved {len(out)} rows -> {OUT / 'clusters.parquet'}")

    reg = pd.concat(meters, ignore_index=True)
    reg.insert(0, "meter_id", [ids.meter_id(i + 1) for i in range(len(reg))])
    reg["transformer_asset_id"] = reg["transformer"].map(ids.transformer_ids())
    reg = reg[["meter_id", "deployment_id", "household_id", "Region", "transformer", "transformer_asset_id"]]
    reg.rename(columns={"Region": "region"}).to_csv(OUT / "meter_registry.csv", index=False)
    print(f"Saved {len(reg)} meters -> {OUT / 'meter_registry.csv'}")


if __name__ == "__main__":
    main()