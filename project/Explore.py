"""Explore the eMARC raw files. Prints the facts needed to design prepare_data.py."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)

DAILY = RAW / "eMAR daily consumption.csv"
BLOCKS = RAW / "eMARC load blocks.csv"
INFO = RAW / "Household-Deployment basic info.xlsx"

# ---------------------------------------------------------------- daily file
print("\n" + "=" * 70 + "\nDAILY CONSUMPTION FILE\n" + "=" * 70)
d = pd.read_csv(DAILY)
d["Date"] = pd.to_datetime(d["Date"], format="%m/%d/%Y", errors="coerce")
print("rows:", len(d), "| dates:", d["Date"].min().date(), "to", d["Date"].max().date())
print("unique households:", d["household_id"].nunique(), "| unique deployments:", d["deployment_id"].nunique())
print("\nunique households per Region:\n", d.groupby("Region")["household_id"].nunique().to_string())
print("\nunique households per Household type:\n", d.groupby("Household type")["household_id"].nunique().to_string())
print("\nunique deployments per Deployment type:\n", d.groupby("Deployment type")["deployment_id"].nunique().to_string())
print("\ndeployments per household (describe):\n", d.groupby("household_id")["deployment_id"].nunique().describe().to_string())
dep_map = d.drop_duplicates("deployment_id")[["deployment_id", "household_id", "Region", "Household type", "Deployment type"]]

# ---------------------------------------------------------------- info file
print("\n" + "=" * 70 + "\nHOUSEHOLD-DEPLOYMENT INFO (xlsx)\n" + "=" * 70)
try:
    sheets = pd.read_excel(INFO, sheet_name=None)
    for name, df in sheets.items():
        print(f"\n-- sheet '{name}': {df.shape}")
        print("columns:", list(df.columns))
        print(df.head(5).to_string())
except ImportError:
    print("openpyxl missing: run  pip install openpyxl  then re-run this script.")
except Exception as e:
    print("Could not read xlsx:", e)

# ---------------------------------------------------------------- load blocks (chunked)
print("\n" + "=" * 70 + "\nLOAD BLOCKS FILE (reading in chunks, may take a minute)\n" + "=" * 70)
blocks_seen, pairs, nrows = set(), [], 0
stats = {"min": float("inf"), "max": float("-inf"), "sum": 0.0, "n": 0, "nulls": 0, "neg": 0}
for ch in pd.read_csv(BLOCKS, chunksize=2_000_000):
    nrows += len(ch)
    blocks_seen.update(ch["block"].unique().tolist())
    v = ch["Load (kW)"]
    stats["min"] = min(stats["min"], v.min()); stats["max"] = max(stats["max"], v.max())
    stats["sum"] += v.sum(); stats["n"] += v.notna().sum()
    stats["nulls"] += v.isna().sum(); stats["neg"] += (v < 0).sum()
    ch["date"] = pd.to_datetime(ch["date"], format="%m/%d/%Y", errors="coerce")
    pairs.append(ch.drop_duplicates(["deployment_id", "date"])[["deployment_id", "date"]])
pairs = pd.concat(pairs).drop_duplicates()

print("rows:", nrows)
print("blocks: min", min(blocks_seen), "max", max(blocks_seen), "unique", len(blocks_seen))
print(f"Load (kW): min {stats['min']:.4f}  max {stats['max']:.3f}  mean {stats['sum']/stats['n']:.4f}  "
      f"nulls {stats['nulls']}  negatives {stats['neg']}")
print("date range:", pairs["date"].min().date(), "to", pairs["date"].max().date())
print("deployments in load blocks:", pairs["deployment_id"].nunique())

cov = pairs.groupby("deployment_id")["date"].agg(days="nunique", first="min", last="max").reset_index()
cov = cov.merge(dep_map, on="deployment_id", how="left")
print("\ndays of data per deployment (describe):\n", cov["days"].describe().to_string())
print("\nload-block deployments by Deployment type:\n", cov["Deployment type"].value_counts(dropna=False).to_string())
print("load-block deployments by Region:\n", cov["Region"].value_counts(dropna=False).to_string())

# month-by-month: how many deployments have data (find the best continuous window)
pairs["month"] = pairs["date"].dt.to_period("M")
m = pairs.groupby("month").agg(deployments=("deployment_id", "nunique"), days=("date", "nunique"))
print("\nmonthly coverage (deployments with data, distinct days):\n", m.to_string())

cov.to_csv(OUT / "deployment_coverage.csv", index=False)
m.to_csv(OUT / "monthly_coverage.csv")
print("\nSaved data/processed/deployment_coverage.csv and monthly_coverage.csv")