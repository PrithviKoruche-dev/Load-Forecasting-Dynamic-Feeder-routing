"""Model metadata (gap-analysis item E5): what a saved model is, how it was trained and how well it did.

Every model file gets a <name>.meta.json next to it with: version, kind, feature list, training settings,
the data windows (train / validation / test), the number of trees, the library version, the file's SHA-256
(so a metadata file can never silently describe a different model) and its test metrics.

    python ml/metadata.py        # (re)writes ml/model_xgb.meta.json for the published v1 model
"""
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import config  # noqa: E402

V1_VERSION = "xgb-24h-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def meta_path(model_path):
    p = Path(model_path)
    return p.with_name(p.stem + ".meta.json")


def build(version, kind, model_path, features, params, windows, metrics, **extra):
    import xgboost
    return {"version": version, "kind": kind, "model_file": Path(model_path).name, "sha256": sha256(model_path),
            "created_at": dt.datetime.now().isoformat(timespec="seconds"), "features": list(features),
            "params": params, "windows": windows, "metrics": metrics,
            "libraries": {"xgboost": xgboost.__version__, "pandas": pd.__version__}, **extra}


def write(meta, model_path):
    path = meta_path(model_path)
    path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return path


def read(model_path, verify=True):
    """Load the metadata of a model file. With verify, refuse if the file's hash does not match."""
    meta = json.loads(meta_path(model_path).read_text())
    if verify and meta["sha256"] != sha256(model_path):
        raise ValueError(f"{meta_path(model_path).name} describes a different file than {Path(model_path).name} "
                         "(SHA-256 mismatch): retrain or regenerate the metadata")
    return meta


def windows_from(df, time_col="timestamp", split_col="split"):
    w = df.groupby(split_col)[time_col].agg(["min", "max", "count"])
    return {s: {"from": str(w.loc[s, "min"]), "to": str(w.loc[s, "max"]), "rows": int(w.loc[s, "count"])}
            for s in ("train", "val", "test") if s in w.index}


def v1_metadata():
    """Reconstruct the metadata of the published model from the committed files and config."""
    import xgboost as xgb
    from features import FEATURES
    model_path = ROOT / "ml" / "model_xgb.json"
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    file_version = ".".join(str(x) for x in json.loads(model_path.read_text())["version"])
    cfg = config.forecasting()
    params = {**cfg["model"]["xgboost"], "n_estimators_used": booster.num_boosted_rounds(),
              "enable_categorical": True, "refit": "train+val with the tree count fixed by early stopping"}
    fa = pd.read_csv(ROOT / "data" / "forecast_all.csv", parse_dates=["timestamp"])
    m = pd.read_csv(ROOT / "docs" / "metrics.csv")
    m = m[m["model"] == "XGBoost (train+val refit)"].set_index("transformer")
    metrics = {"test": {"pooled": m.loc["ALL", ["MAE_kW", "RMSE_kW", "nRMSE_%", "MAPE_%"]].to_dict(),
                        "nRMSE_%_by_transformer": m.drop("ALL")["nRMSE_%"].to_dict(),
                        "units": "MAE/RMSE in raw cluster kW"}}
    return build(V1_VERSION, "xgboost-24h-direct", model_path, FEATURES, params, windows_from(fa), metrics,
                 horizon_h=24, target="load_kw_raw / train mean of the transformer",
                 scaling={"target_peak_kw": cfg["scaling"]["target_peak_kw"]},
                 trained_with={"xgboost": file_version}, outputs=["data/forecast.csv", "data/forecast_all.csv"],
                 note="written by ml/metadata.py from the committed model, forecasts and metrics; ml/evaluate.py "
                      "rewrites it after every retraining")


def register(store, model_path):
    """Record a model (from its metadata file) in the database's model registry."""
    meta = read(model_path)
    store.register_model(meta["version"], meta["kind"], meta["model_file"], features=meta["features"],
                         params=meta["params"], metrics=meta["metrics"],
                         notes=json.dumps({"windows": meta["windows"], "sha256": meta["sha256"]}))
    return meta


if __name__ == "__main__":
    meta = v1_metadata()
    print("wrote", write(meta, ROOT / "ml" / "model_xgb.json"))
    print(json.dumps({k: meta[k] for k in ("version", "sha256", "windows")}, indent=2))
