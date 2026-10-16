import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # the project folder, wherever you run from
RAW = ROOT / "data" / "raw"
print("Looking in:", RAW, "| exists:", RAW.exists())

# Everything in raw (any type), so we see what you actually have
all_files = [f for f in RAW.rglob("*") if f.is_file()] if RAW.exists() else []
print(f"{len(all_files)} files found")
for f in all_files[:30]:
    print("  ", f.relative_to(RAW), f"({f.stat().st_size / 1e6:.1f} MB)")

csvs = [f for f in all_files if f.suffix.lower() == ".csv"]
for f in csvs[:10]:
    print("\n" + "=" * 70)
    print(f.relative_to(RAW))
    df = pd.read_csv(f, nrows=2000)
    print("columns:", list(df.columns))
    print(df.dtypes.to_string())
    print(df.head(4).to_string())