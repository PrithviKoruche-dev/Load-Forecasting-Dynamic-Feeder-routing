"""Cold start (gap-analysis item E6): make sure a forecasting model and its forecasts exist, building whatever
is missing in the right order:

    raw eMARC files  --prepare_data.py-->  clusters.parquet  --train_xgb.py-->  model + forecasts
                     --evaluate.py-->  metrics + model metadata  --intervals.py-->  P10/P90 bands

    python ml/cold_start.py            # does nothing if everything is present and the metadata matches
    python ml/cold_start.py --force    # retrain from clusters.parquet (or from the raw files if needed)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = ["eMARC load blocks.csv", "Household-Deployment basic info.xlsx"]


def status(root=ROOT):
    """What is present: {raw, clusters, model, forecasts, metadata_ok}."""
    root = Path(root)
    sys.path.insert(0, str(root / "ml"))
    import metadata
    model = root / "ml" / "model_xgb.json"
    ok = False
    if model.exists() and metadata.meta_path(model).exists():
        try:
            metadata.read(model)
            ok = True
        except ValueError:
            ok = False
    return {"raw": all((root / "data" / "raw" / f).exists() for f in RAW_FILES),
            "clusters": (root / "data" / "processed" / "clusters.parquet").exists(),
            "model": model.exists(), "forecasts": (root / "data" / "forecast.csv").exists(), "metadata_ok": ok}


def _run(script, root):
    print(f"--> python ml/{script}")
    subprocess.run([sys.executable, str(Path(root) / "ml" / script)], cwd=str(Path(root) / "ml"), check=True)


def ensure(root=ROOT, force=False):
    """Build what is missing. Returns the list of scripts that were run."""
    root = Path(root)
    st = status(root)
    if not force and st["model"] and st["forecasts"] and st["metadata_ok"]:
        print("Model, forecasts and metadata are present: nothing to do (use --force to retrain).")
        return []
    steps = []
    if not st["clusters"]:
        if not st["raw"]:
            raise SystemExit("No data/processed/clusters.parquet and no eMARC raw files in data/raw/ "
                             f"({', '.join(RAW_FILES)}). Obtain the eMARC data first; see the README.")
        steps.append("prepare_data.py")
    steps += ["train_xgb.py", "evaluate.py", "intervals.py"]
    for s in steps:
        _run(s, root)
    return steps


def main():
    ap = argparse.ArgumentParser(description="Make sure a trained model and its forecasts exist.")
    ap.add_argument("--force", action="store_true", help="retrain even if a model exists")
    args = ap.parse_args()
    print("Ran:", ensure(force=args.force) or "nothing")


if __name__ == "__main__":
    main()
