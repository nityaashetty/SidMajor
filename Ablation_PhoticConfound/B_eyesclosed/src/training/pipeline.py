"""Assembles splits + store + biomarkers into datasets (shared by train, evaluate and verify)."""
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.config import features_fingerprint, store_fingerprint
from ..common.logging_utils import read_json
from ..data.dataset import N_STATES, RecordingBags
from ..data.splits import EVAL_SPLITS, load_splits
from ..data.store import load_manifest
from ..features.normative import build_feature_matrices


def fold_splits(splits, n_folds, k, seed=0):
    """Pool train+val, stratified k-fold by class; fold k's held-out part becomes 'val'.
    test / test_no_overlap are untouched (the leakage guard already separated them from train+val)."""
    from sklearn.model_selection import StratifiedKFold
    pool = splits["train"] + splits["val"]
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(pool, [r["label"] for r in pool]))
    tr, va = folds[k]
    out = {k_: v for k_, v in splits.items() if k_ not in ("val_no_overlap",)}
    out["train"] = [pool[i] for i in tr]
    out["val"] = [pool[i] for i in va]
    return out


def load_bundle(cfg, fold=None):
    """fold=None: official split. fold=k: CV fold k of cfg['cv'] (normative model + age scaling refit on it)."""
    p = cfg["paths"]
    manifest = load_manifest(p["store"])
    if manifest["fingerprint"] != store_fingerprint(cfg):
        raise SystemExit("store was built with a different `data` config - re-run scripts/01_build_store.py")
    meta = read_json(Path(p["features"]) / "biomarkers_meta.json")
    if meta["fingerprint"] != features_fingerprint(cfg):
        raise SystemExit("biomarkers are stale for this config - re-run scripts/02_build_features.py")
    splits = load_splits(p["dataset_root"])
    if fold is not None:
        splits = fold_splits(splits, cfg["cv"]["n_folds"], fold, cfg["cv"]["seed"])
    df = pd.read_csv(Path(p["features"]) / "biomarkers.csv", dtype={"serial": str})
    mats, names, normative = build_feature_matrices(df, splits, cfg["features"])
    age_tr = mats["train"]["age"]
    age_mu, age_sd = float(age_tr.mean()), float(age_tr.std())
    return {"cfg": cfg, "manifest": manifest, "splits": splits, "mats": mats, "bio_names": names,
            "normative": normative, "age_mu": age_mu, "age_sd": age_sd, "n_states": N_STATES}


def make_dataset(bundle, split, train, seed=0):
    cfg, m = bundle["cfg"], bundle["mats"][split]
    age_z = (m["age"] - bundle["age_mu"]) / bundle["age_sd"]
    return RecordingBags(bundle["splits"][split], m["Z"], age_z[:, None], cfg["paths"]["store"],
                         bundle["manifest"], cfg["segments"], cfg["train"]["augment"], train=train, seed=seed)


def eval_datasets(bundle):
    return {sp: make_dataset(bundle, sp, train=False) for sp in EVAL_SPLITS}


def labels(bundle, split):
    return np.asarray(bundle["mats"][split]["y"])
