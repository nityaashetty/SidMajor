"""EDF -> single fp16 memmap cache, plus train-set normalisation statistics.

Why this exists (measured on this machine):
    pyedflib full EDF read .............. 230.0 ms / patient
    random 10 s crop from this cache .... 0.091 ms   (~2500x faster)
    full-recording read from this cache . 5.7   ms

    Phase 1 worked around slow EDF reads by freezing 8 crops per patient into .npz
    (precompute_multicrop.py). That replayed the SAME 8 feature vectors every epoch,
    which is memorisation, not augmentation -- the direct cause of the 77% train /
    53% val gap. With this cache, a fresh random crop costs 0.091 ms, so crops are
    sampled on the fly and the frozen-crop problem disappears.

Layout
    signals.dat   one flat fp16 array; each patient is a contiguous [n_samples, 19]
                  TIME-MAJOR block, so a 10 s crop is one contiguous read.
    manifest.json {serial: {offset, n_samples}} + build metadata
    norm_stats.npz per-channel mean/std computed over the TRAIN split only

Build is resumable and crash-safe: the manifest is flushed periodically and the data
file is truncated back to the last committed offset on resume.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

N_EEG = 19          # channels 0..18; ch 19 = EKG, ch 20 = Photic -> always dropped
N_RAW = 21
SFREQ = 200.0
FLUSH_EVERY = 25

DATA_FILE = "signals.dat"
MANIFEST = "manifest.json"
NORM_FILE = "norm_stats.npz"


# ----------------------------------------------------------------------------- build
def _read_edf(path: Path) -> np.ndarray:
    """Return [n_samples, 19] float32, time-major. Validates header."""
    import pyedflib

    r = pyedflib.EdfReader(str(path))
    try:
        if r.signals_in_file != N_RAW:
            raise ValueError(f"{path.name}: expected {N_RAW} channels, got {r.signals_in_file}")
        sf = r.getSampleFrequency(0)
        if abs(sf - SFREQ) > 1e-6:
            raise ValueError(f"{path.name}: expected {SFREQ} Hz, got {sf}")
        n = r.getNSamples()[0]
        out = np.empty((n, N_EEG), dtype=np.float32)
        for i in range(N_EEG):
            out[:, i] = r.readSignal(i)
    finally:
        r.close()
    return out


def _load_manifest(cache: Path) -> dict:
    f = cache / MANIFEST
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"entries": {}, "meta": {}}


def _save_manifest(cache: Path, man: dict) -> None:
    tmp = cache / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    os.replace(tmp, cache / MANIFEST)


def build_cache(dataset_root: Path, cache_dir: Path, serials: list[str],
                verbose: bool = True) -> dict:
    """Idempotent + resumable. Returns the manifest."""
    from tqdm import tqdm

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    edf_dir = Path(dataset_root) / "signal" / "edf"
    dat = cache_dir / DATA_FILE

    man = _load_manifest(cache_dir)
    entries: dict = man["entries"]

    # Resume: trim the data file back to the end of the last committed entry so a
    # half-written patient from a previous crash cannot corrupt later reads.
    committed_end = 0
    for e in entries.values():
        committed_end = max(committed_end, e["offset"] + e["n_samples"] * N_EEG)
    if dat.exists():
        want_bytes = committed_end * 2            # fp16
        if dat.stat().st_size != want_bytes:
            with open(dat, "r+b") as fh:
                fh.truncate(want_bytes)
            if verbose:
                print(f"[cache] truncated {DATA_FILE} to last committed offset "
                      f"({want_bytes/1e9:.2f} GB)")
    else:
        dat.touch()

    todo = [s for s in serials if s not in entries]
    if verbose:
        print(f"[cache] {len(entries)} already cached, {len(todo)} to build")
    if not todo:
        man["meta"].update(_meta(entries))
        _save_manifest(cache_dir, man)
        return man

    offset = committed_end
    with open(dat, "ab") as fh:
        for i, s in enumerate(tqdm(todo, desc="caching EDF", disable=not verbose)):
            p = edf_dir / f"{s}.edf"
            if not p.exists():
                raise FileNotFoundError(f"missing EDF for serial {s}: {p}")
            arr = _read_edf(p)
            fh.write(np.ascontiguousarray(arr, dtype=np.float16).tobytes())
            entries[s] = {"offset": offset, "n_samples": int(arr.shape[0])}
            offset += arr.shape[0] * N_EEG
            if (i + 1) % FLUSH_EVERY == 0:
                fh.flush()
                os.fsync(fh.fileno())
                _save_manifest(cache_dir, man)
        fh.flush()
        os.fsync(fh.fileno())

    man["meta"].update(_meta(entries))
    _save_manifest(cache_dir, man)
    if verbose:
        print(f"[cache] done: {len(entries)} patients, {dat.stat().st_size/1e9:.2f} GB")
    return man


def _meta(entries: dict) -> dict:
    lens = np.array([e["n_samples"] for e in entries.values()]) if entries else np.array([0])
    return {
        "n_patients": len(entries),
        "n_channels": N_EEG,
        "sfreq": SFREQ,
        "dtype": "float16",
        "layout": "time_major[n_samples, 19] per patient, concatenated",
        "samples_min": int(lens.min()),
        "samples_max": int(lens.max()),
        "samples_median": int(np.median(lens)),
    }


# ------------------------------------------------------------------------ norm stats
def compute_norm_stats(cache_dir: Path, train_serials: list[str],
                       max_patients: int = 400, seed: int = 0) -> dict:
    """Per-channel mean/std over the TRAIN split only.

    Phase-1 defect this fixes:
        eeg_dataset.py:_normalize z-scored each crop independently. That throws away
        absolute amplitude entirely and makes every crop's per-channel variance exactly
        1, so between-patient amplitude differences became invisible. The CAUEEG paper
        (Sec 3.2, "Data normalization") uses training-set statistics instead. We follow
        the paper; `data.normalize: percrop` is kept as an ablation switch.

    Statistics come from TRAIN serials only -- never val/test -- so there is no
    normalisation leakage.
    """
    cache_dir = Path(cache_dir)
    man = _load_manifest(cache_dir)
    entries = man["entries"]
    missing = [s for s in train_serials if s not in entries]
    if missing:
        raise RuntimeError(f"{len(missing)} train serials not in cache, e.g. {missing[:3]}")

    rng = np.random.RandomState(seed)
    pick = list(train_serials)
    if len(pick) > max_patients:
        pick = [pick[i] for i in rng.permutation(len(pick))[:max_patients]]

    mm = open_memmap(cache_dir)
    # Welford over patients (each patient contributes its own full-recording moments).
    n_tot = 0
    mean = np.zeros(N_EEG, np.float64)
    m2 = np.zeros(N_EEG, np.float64)
    for s in pick:
        blk = read_block(mm, entries[s]).astype(np.float64)   # [n, 19]
        n = blk.shape[0]
        mu = blk.mean(0)
        var = blk.var(0)
        delta = mu - mean
        tot = n_tot + n
        mean += delta * (n / tot)
        m2 += var * n + (delta ** 2) * (n_tot * n / tot)
        n_tot = tot
    std = np.sqrt(m2 / max(n_tot, 1))
    std = np.maximum(std, 1e-6)

    np.savez(cache_dir / NORM_FILE, mean=mean.astype(np.float32),
             std=std.astype(np.float32), n_patients=len(pick), n_samples=n_tot)
    return {"mean": mean, "std": std, "n_patients": len(pick), "n_samples": int(n_tot)}


def load_norm_stats(cache_dir: Path):
    f = Path(cache_dir) / NORM_FILE
    if not f.exists():
        raise FileNotFoundError(
            f"{NORM_FILE} missing. Run scripts/01_build_cache.py first.")
    z = np.load(f)
    return z["mean"].astype(np.float32), z["std"].astype(np.float32)


# ------------------------------------------------------------------------------ read
def open_memmap(cache_dir: Path) -> np.memmap:
    return np.memmap(Path(cache_dir) / DATA_FILE, dtype=np.float16, mode="r")


def read_block(mm: np.memmap, entry: dict) -> np.ndarray:
    n = entry["n_samples"]
    off = entry["offset"]
    return mm[off: off + n * N_EEG].reshape(n, N_EEG)


def read_crop(mm: np.memmap, entry: dict, start: int, length: int) -> np.ndarray:
    """-> [19, length] float32. One contiguous read thanks to the time-major layout."""
    off = entry["offset"] + start * N_EEG
    chunk = mm[off: off + length * N_EEG].reshape(length, N_EEG)
    return np.ascontiguousarray(chunk.T, dtype=np.float32)


def load_manifest(cache_dir: Path) -> dict:
    man = _load_manifest(Path(cache_dir))
    if not man["entries"]:
        raise FileNotFoundError(
            f"Cache at {cache_dir} is empty. Run scripts/01_build_cache.py first.")
    return man
