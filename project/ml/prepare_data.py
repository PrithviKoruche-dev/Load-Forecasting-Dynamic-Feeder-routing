"""Build data/processed/clusters.parquet from the eMARC raw files.

Output: one row per (hourly timestamp, transformer) with
    timestamp, transformer_id, region, n_households, load_kw_raw, temp_c
load_kw_raw is the cluster's total hourly mean demand in kW BEFORE scaling to the grid's kVA rating.
"""
import json
import urllib.request
import urllib.parse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

START, END = "2019-06-01", "2020-03-24"          # ends before the COVID lockdown (25 Mar 2020)
MIN_BLOCKS_PER_DAY = 90                          # a day is "complete" if >= 90 of 96 blocks exist
MIN_COMPLETE_SHARE = 0.95                        # household kept if >= 95% of days complete
MAX_GAP_STEPS = 4                                # interpolate gaps up to 4 x 15 min = 1 h

# transformer -> (region, which half of the region's households)
TX_MAP = {
    "T1": ("Pune city", 0), "T2": ("Pune city", 1),
    "T3": ("Pune", None), "T4": ("Aurangabad", None),
    "T5": ("Kanpur rural", None), "T6": ("Gonda", None),
}
COORDS = {                                       # approximate city centres for weather
    "Pune city": (18.52, 73.86), "Pune": (18.52, 73.86), "Aurangabad": (19.88, 75.34),
    "Kanpur rural": (26.45, 80.33), "Gonda": (27.13, 81.96),
}


def select_households():
    info = pd.read_excel(RAW / "Household-Deployment basic info.xlsx")
    main = info[info["Deployment type"].str.lower() == "mainline"]
    return main[["deployment_id", "household_id", "Region"]].drop_duplicates("deployment_id")


def read_blocks(ids):
    parts = []
    for ch in pd.read_csv(RAW / "eMARC load blocks.csv", chunksize=2_000_000):
        ch = ch[ch["deployment_id"].isin(ids)]
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
                                "timezone": "Asia/Kolkata"})
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
    rows = []
    for tx, (region, half) in TX_MAP.items():
        d = dep[dep["Region"] == region]
        ids = d["deployment_id"].tolist()
        if half is not None:                      # split Pune city into two halves
            mid = len(ids) // 2
            ids = ids[:mid] if half == 0 else ids[mid:]
        mean_kw = hourly[ids].mean(axis=1)        # mean over households with data that hour
        total = mean_kw * len(ids)                # = sum, with missing households filled by the mean
        total = total.interpolate(limit=6, limit_area="inside")
        weather = fetch_weather(region).set_index("timestamp")["temp_c"].reindex(hourly.index)
        weather = weather.interpolate(limit=3)
        rows.append(pd.DataFrame({"timestamp": hourly.index, "transformer_id": tx,
                                  "region": region, "n_households": len(ids),
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


if __name__ == "__main__":
    main()