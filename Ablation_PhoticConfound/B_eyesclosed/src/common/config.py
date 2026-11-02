"""Config loading and path resolution. No path in this approach is hardcoded.

Dataset root priority: $CAUEEG_ROOT > paths.dataset_root in config.yaml.
Relative paths are resolved against the approach folder, never the CWD.
Everything generated lives under paths.artifacts (default: <approach>/artifacts).
"""
import hashlib
import json
import os
from pathlib import Path

import yaml

APPROACH_ROOT = Path(__file__).resolve().parents[2]


def _set_dotted(d, dotted, raw_value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(raw_value)


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else (APPROACH_ROOT / p).resolve()


def load_config(path=None, overrides=()):
    """Load config.yaml, apply `a.b.c=value` overrides, resolve paths."""
    path = Path(path) if path else APPROACH_ROOT / "config.yaml"
    if not path.is_absolute():
        path = path.resolve() if path.exists() else APPROACH_ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for o in overrides or ():
        key, value = o.split("=", 1)
        _set_dotted(cfg, key.strip(), value.strip())

    p = cfg.setdefault("paths", {})
    dataset_root = _abs(os.environ.get("CAUEEG_ROOT") or p.get("dataset_root", "../caueeg-dataset"))
    artifacts = _abs(p.get("artifacts", "artifacts"))
    cfg["paths"] = {
        "dataset_root": str(dataset_root),
        "artifacts": str(artifacts),
        "store": str(_abs(p["store"]) if p.get("store") else artifacts / "store"),  # ablation: read A11 store
        "features": str(artifacts / "features"),
        "runs": str(artifacts / "runs"),
    }
    cfg["_config_path"] = str(path)
    return cfg


def fingerprint(cfg, sections):
    """Short hash of the given config sections; used to detect stale artifacts."""
    blob = json.dumps({k: cfg.get(k) for k in sections}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


STORE_IGNORED = ("workers",)                                   # do not change the stored signals
EXTRACTION_KEYS = ("window_sec", "step_sec", "max_windows")    # the rest of `features` is applied at load time


def store_fingerprint(cfg):
    return fingerprint({"data": {k: v for k, v in cfg["data"].items() if k not in STORE_IGNORED}}, ["data"])


def features_fingerprint(cfg):
    fx = {k: cfg["features"][k] for k in EXTRACTION_KEYS}
    return fingerprint({"store": store_fingerprint(cfg), "fx": fx}, ["store", "fx"])


def add_common_args(parser):
    parser.add_argument("--config", default=None, help="config file (default: <approach>/config.yaml)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set train.epochs=50 model.use_age=false")
    return parser
