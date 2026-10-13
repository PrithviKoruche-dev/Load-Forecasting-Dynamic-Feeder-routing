import pandas as pd
from pathlib import Path

for f in sorted(Path("data/raw").rglob("*.csv"))[:5]:
    print("\n==", f.name)
    df = pd.read_csv(f, nrows=5)
    print(df.to_string())