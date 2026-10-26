"""Second exploration: Mainline households only, window choice, demand by month."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the project/ folder
RAW = ROOT / "data" / "raw"
pd.set_option("display.width", 200)

info = pd.read_excel(RAW / "Household-Deployment basic info.xlsx")
main = info[info["Deployment type"].str.lower() == "mainline"].copy()
print("Mainline deployments in info file:", len(main), "| households:", main["household_id"].nunique())
print("households with >1 Mainline deployment:",
      int((main.groupby("household_id")["deployment_id"].nunique() > 1).sum()))
print("\nMainline households per Region:\n", main.groupby("Region")["household_id"].nunique().to_string())
ids = set(main["deployment_id"])

# daily completeness + kWh per (deployment, date), Mainline only
parts = []
for ch in pd.read_csv(RAW / "eMARC load blocks.csv", chunksize=2_000_000):
    ch = ch[ch["deployment_id"].isin(ids)]
    if ch.empty:
        continue
    ch["date"] = pd.to_datetime(ch["date"], format="%m/%d/%Y")
    parts.append(ch.groupby(["deployment_id", "date"]).agg(n=("block", "size"), kw=("Load (kW)", "sum")).reset_index())
daily = pd.concat(parts).groupby(["deployment_id", "date"], as_index=False).sum()
daily["kwh"] = daily["kw"] * 0.25
print("\nblocks per day: max", daily["n"].max(), "| share of days with all 96 blocks:",
      round((daily["n"] == 96).mean(), 3))
daily = daily.merge(main[["deployment_id", "household_id", "Region"]], on="deployment_id")

# monthly mean daily kWh per household, by region (shows seasonality + lockdown)
full = daily[daily["n"] >= 90].copy()
full["month"] = full["date"].dt.to_period("M")
tab = full.groupby(["month", "Region"])["kwh"].mean().unstack().round(1)
print("\nMean daily kWh per Mainline meter, by month and region:\n",
      tab.loc["2019-03":"2020-06"].to_string())

# candidate windows: households with >=95% of days complete
windows = {
    "W1 2019-03-01..2020-03-24": ("2019-03-01", "2020-03-24"),
    "W2 2019-06-01..2020-03-24": ("2019-06-01", "2020-03-24"),
    "W3 2019-06-01..2020-06-30": ("2019-06-01", "2020-06-30"),
}
rows = []
for name, (a, b) in windows.items():
    ndays = (pd.Timestamp(b) - pd.Timestamp(a)).days + 1
    w = daily[(daily["date"] >= a) & (daily["date"] <= b) & (daily["n"] >= 90)]
    ok = w.groupby(["Region", "household_id", "deployment_id"]).size().reset_index(name="days")
    ok = ok[ok["days"] >= 0.95 * ndays]
    r = ok.groupby("Region")["household_id"].nunique()
    r["TOTAL"] = ok["household_id"].nunique()
    r.name = name + f" ({ndays} d)"
    rows.append(r)
print("\nMainline households with >=95% complete days, per window:\n", pd.DataFrame(rows).T.fillna(0).astype(int).to_string())