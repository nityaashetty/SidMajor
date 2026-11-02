"""Signal store built from the original EDFs.

artifacts/store/
  signals/<serial>.npy     float16 [19, n] in uV, band-passed and resampled to data.fs
  timeline/<serial>.npz    state [n_sec] int8, clean [n_sec] bool, photic [k, 3] (start_s, end_s, Hz)
  manifest.json            per-recording metadata, channel scale (train-only), config fingerprint
"""
import json
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from .edf_io import EEG_CHANNELS, read_eeg
from .events import build_timeline


def store_dirs(store_root):
    root = Path(store_root)
    return root / "signals", root / "timeline", root / "manifest.json"


def preprocess(x, fs_in, cfg):
    lo, hi = cfg["bandpass"]
    sos = butter(cfg["filter_order"], [lo, hi], btype="band", fs=fs_in, output="sos")
    x = x - np.median(x, axis=1, keepdims=True)
    y = sosfiltfilt(sos, x, axis=1)
    fs_out = cfg["fs"]
    if fs_out != fs_in:
        g = gcd(int(fs_in), int(fs_out))
        y = resample_poly(y, int(fs_out) // g, int(fs_in) // g, axis=1)
    clip = cfg["clip_uv"]
    return np.clip(y, -clip, clip).astype(np.float32)


def build_one(serial, dataset_root, store_root, cfg):
    """Process one recording. Returns metadata incl. per-channel std over clean seconds."""
    sig_dir, tl_dir, _ = store_dirs(store_root)
    root = Path(dataset_root)
    x, fs_in = read_eeg(root / "signal" / "edf" / f"{serial}.edf")
    if fs_in != cfg["fs_in"]:
        raise ValueError(f"{serial}: sampling rate {fs_in}, expected {cfg['fs_in']}")
    events = json.loads((root / "event" / f"{serial}.json").read_text(encoding="utf-8"))
    state, clean, photic = build_timeline(events, x.shape[1], fs_in, cfg)

    y = preprocess(x, fs_in, cfg)
    fs = cfg["fs"]
    n_sec = min(len(state), y.shape[1] // fs)
    state, clean = state[:n_sec], clean[:n_sec]

    sec_mask = np.repeat(clean, fs)
    seg = y[:, : n_sec * fs][:, sec_mask] if sec_mask.any() else y
    ch_std = seg.std(axis=1)

    np.save(sig_dir / f"{serial}.npy", y.astype(np.float16))
    np.savez(tl_dir / f"{serial}.npz", state=state, clean=clean, photic=photic)
    counts = np.bincount(state[clean], minlength=5)
    return {
        "serial": serial,
        "n_samples": int(y.shape[1]),
        "n_sec": int(n_sec),
        "clean_sec": int(clean.sum()),
        "clean_sec_by_state": counts.tolist(),
        "n_photic_blocks": int(len(photic)),
        "ch_std": ch_std.astype(float).tolist(),
    }


def load_manifest(store_root):
    _, _, mf = store_dirs(store_root)
    if not mf.exists():
        raise FileNotFoundError(f"{mf} missing - run scripts/01_build_store.py first")
    return json.loads(mf.read_text(encoding="utf-8"))


def load_signal(store_root, serial, mmap=True):
    sig_dir, _, _ = store_dirs(store_root)
    return np.load(sig_dir / f"{serial}.npy", mmap_mode="r" if mmap else None)


def load_timeline(store_root, serial):
    _, tl_dir, _ = store_dirs(store_root)
    with np.load(tl_dir / f"{serial}.npz") as z:
        return z["state"], z["clean"], z["photic"]


CHANNELS = EEG_CHANNELS
