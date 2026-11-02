"""Step 2: artifacts/store -> artifacts/features/biomarkers.csv (one row per recording, raw values).

Normative z-scoring is NOT applied here: it is fitted on the train split at use time
(src/features/normative.py), so this file contains no split information at all.
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from src.common.config import add_common_args, features_fingerprint, load_config, store_fingerprint
from src.common.logging_utils import log, write_json
from src.data.splits import all_task_serials, load_splits
from src.data.store import load_manifest
from src.features.biomarkers import recording_features


def _one(args):
    serial, store, fs, fcfg = args
    return recording_features(serial, store, fs, fcfg)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    manifest = load_manifest(cfg["paths"]["store"])
    if manifest["fingerprint"] != store_fingerprint(cfg):
        raise SystemExit("store was built with a different `data` config - re-run 01_build_store.py")
    splits = load_splits(cfg["paths"]["dataset_root"])
    serials = all_task_serials(splits)
    missing = [s for s in serials if s not in manifest["recordings"]]
    if missing:
        raise SystemExit(f"{len(missing)} recordings missing from the store - re-run 01_build_store.py")

    out_dir = Path(cfg["paths"]["features"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"features: computing biomarkers for {len(serials)} recordings ({cfg['data']['workers']} workers)")
    rows, t0 = [], time.time()
    jobs = [(s, cfg["paths"]["store"], manifest["fs"], cfg["features"]) for s in serials]
    with ProcessPoolExecutor(max_workers=cfg["data"]["workers"]) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 200 == 0 or i == len(futs):
                log(f"features: {i}/{len(futs)} ({time.time() - t0:.0f} s)")

    df = pd.DataFrame(rows).sort_values("serial").reset_index(drop=True)
    df.to_csv(out_dir / "biomarkers.csv", index=False)
    feat_cols = [c for c in df.columns if c != "serial" and not c.startswith("_")]
    nan_rate = df[feat_cols].isna().mean()
    write_json(out_dir / "biomarkers_meta.json", {
        "fingerprint": features_fingerprint(cfg),
        "n_recordings": len(df),
        "n_features": len(feat_cols),
        "features": feat_cols,
        "nan_rate": {k: float(v) for k, v in nan_rate.items() if v > 0},
        "ec_source_counts": df["_ec_source"].value_counts().to_dict(),
    })
    log(f"features: done. {len(feat_cols)} biomarkers x {len(df)} recordings. "
        f"EC source: {df['_ec_source'].value_counts().to_dict()} | has EO: {int(df['has_eo'].sum())} "
        f"| has photic: {int(df['has_photic'].sum())}")


if __name__ == "__main__":
    main()
