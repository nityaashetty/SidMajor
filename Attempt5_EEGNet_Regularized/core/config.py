"""Config loading + path resolution.

Phase-1 defect this fixes:
    Every script in ProjectAttempt1/src hardcoded r"D:\\Major\\..." (train_npz.py:7-15,
    evaluate.py:8-17, precompute_multicrop.py:6-11, ...). Nothing ran after the project
    moved. Here, NO path is ever hardcoded: the dataset root is resolved at runtime from
    config -> $CAUEEG_ROOT -> upward directory search, so the same folder runs unchanged
    on all three devices.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

ENV_VAR = "CAUEEG_ROOT"
REQUIRED_IN_ROOT = ("dementia.json", "signal/edf")


def _looks_like_dataset(p: Path) -> bool:
    return all((p / r).exists() for r in REQUIRED_IN_ROOT)


def _search_upward(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        cand = parent / "caueeg-dataset"
        if _looks_like_dataset(cand):
            return cand
    return None


def resolve_dataset_root(cfg_value, config_dir: Path) -> Path:
    """config value -> $CAUEEG_ROOT -> search upward from the config file."""
    tried = []
    if cfg_value:
        p = Path(cfg_value).expanduser()
        if not p.is_absolute():
            p = (config_dir / p).resolve()
        tried.append(p)
        if _looks_like_dataset(p):
            return p
    env = os.environ.get(ENV_VAR)
    if env:
        p = Path(env).expanduser()
        tried.append(p)
        if _looks_like_dataset(p):
            return p
    found = _search_upward(config_dir)
    if found is not None:
        return found
    raise FileNotFoundError(
        "Could not locate the CAUEEG dataset root.\n"
        f"  Tried: {[str(t) for t in tried]}\n"
        f"  A valid root contains: {', '.join(REQUIRED_IN_ROOT)}\n"
        f"  Fix by setting paths.dataset_root in your config.yaml, "
        f"or by exporting {ENV_VAR}."
    )


def config_fingerprint(cfg: dict) -> str:
    """Stable hash of the *scientific* config, so a checkpoint can be matched to the
    settings that produced it. Paths and worker counts are excluded on purpose -- they
    differ per device and must not invalidate a cross-device comparison."""
    scientific = {k: v for k, v in cfg.items() if k not in ("paths", "runtime")}
    blob = json.dumps(scientific, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg_dir = path.parent

    paths = cfg.setdefault("paths", {})
    root = resolve_dataset_root(paths.get("dataset_root"), cfg_dir)
    paths["dataset_root"] = str(root)

    # Cache defaults to a project-level sibling so two approaches on the SAME machine
    # share one 7-8 GB cache instead of building it twice.
    cache = paths.get("cache_dir") or "../_cache/eeg_fp16"
    cache = Path(cache).expanduser()
    if not cache.is_absolute():
        cache = (cfg_dir / cache).resolve()
    paths["cache_dir"] = str(cache)

    out = paths.get("output_dir") or "outputs"
    out = Path(out).expanduser()
    if not out.is_absolute():
        out = (cfg_dir / out).resolve()
    paths["output_dir"] = str(out)

    cfg["_config_path"] = str(path)
    cfg["_fingerprint"] = config_fingerprint(cfg)
    return cfg


def describe(cfg: dict) -> str:
    p = cfg["paths"]
    return (
        f"approach      : {cfg.get('approach')}\n"
        f"fingerprint   : {cfg['_fingerprint']}\n"
        f"dataset_root  : {p['dataset_root']}\n"
        f"cache_dir     : {p['cache_dir']}\n"
        f"output_dir    : {p['output_dir']}"
    )
