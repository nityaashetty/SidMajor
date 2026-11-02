"""Dataset over the fp16 cache.

Three Phase-1 defects fixed here:

 1. FROZEN CROPS. precompute_multicrop.py wrote 8 fixed .npz feature vectors per
    patient and npz_dataset.py replayed those same 8 every epoch. The model memorised
    8 points per patient (train 77% / val 53%). Here crops are drawn fresh every epoch
    (0.091 ms each), so `train_repeats: 8` means 8 DIFFERENT crops per epoch.

 2. SINGLE-CROP EVALUATION. precompute_multicrop.py gave val/test exactly one centre
    crop, i.e. 10 s judged out of a 301-1757 s recording. Measured on the Phase-1
    features: 63.5% of patients have 8 crops that disagree, within-patient variance is
    52.5% of total variance, and majority-voting 8 crops was worth +5.05 points.
    Here eval uses `eval_views` DETERMINISTIC evenly-spaced crops, aggregated in the
    model. Deterministic => the three devices produce identical eval crops.

 3. AGE DISCARDED. age was loaded into every batch (npz_dataset.py:36) and never passed
    to QSFENet.forward (qsfe_net.py:79). Measured: logistic regression on age ALONE
    scores 55.08% test / F1 0.523 vs QSFE-Net's 53.39% / 0.5226; age + S1 reaches
    61.86%. Age is returned here and consumed by every model.

Variable recording length is handled explicitly: lengths run 301-1757 s and 456 of 1379
recordings hold fewer than 72 non-overlapping 10 s windows, so segment COUNT is fixed
and segment SPACING adapts, never the other way round.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import N_EEG, SFREQ, load_manifest, load_norm_stats, open_memmap


def deterministic_starts(n_samples: int, crop_len: int, n_views: int, t0: int) -> np.ndarray:
    """Evenly spaced, reproducible across machines. Used for ALL evaluation."""
    lo = min(t0, max(0, n_samples - crop_len))
    hi = max(lo, n_samples - crop_len)
    if n_views == 1:
        return np.array([(lo + hi) // 2], dtype=np.int64)
    return np.linspace(lo, hi, n_views).astype(np.int64)


def stratified_starts(rng: np.random.Generator, n_samples: int, crop_len: int,
                      n_views: int, t0: int) -> np.ndarray:
    """One random start inside each of n_views equal bins: full coverage + augmentation."""
    lo = min(t0, max(0, n_samples - crop_len))
    hi = max(lo, n_samples - crop_len)
    if n_views == 1:
        return np.array([rng.integers(lo, hi + 1)], dtype=np.int64)
    edges = np.linspace(lo, hi + 1, n_views + 1)
    out = np.empty(n_views, dtype=np.int64)
    for i in range(n_views):
        a, b = int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1]))
        out[i] = rng.integers(a, b)
    return np.clip(out, lo, hi)


class CAUEEGCrops(Dataset):
    """Yields x:[V, 19, T] float32, age:[1], label:[], index:[].

    mode='train' -> random/stratified starts, len = n_patients * train_repeats
    mode='eval'  -> deterministic starts,    len = n_patients  (1 item == 1 patient)
    """

    def __init__(self, records, cache_dir, *, crop_seconds=10.0, t0_seconds=10.0,
                 n_views=1, mode="train", train_repeats=1, normalize="trainset",
                 age_mean=0.0, age_std=1.0, sampling="auto",
                 noise_awgn=0.0, noise_mwgn=0.0, seed=0):
        assert mode in ("train", "eval")
        assert normalize in ("trainset", "percrop")
        self.records = list(records)
        self.cache_dir = str(cache_dir)
        self.crop_len = int(round(crop_seconds * SFREQ))
        self.t0 = int(round(t0_seconds * SFREQ))
        self.n_views = int(n_views)
        self.mode = mode
        self.repeats = int(train_repeats) if mode == "train" else 1
        self.normalize = normalize
        self.age_mean, self.age_std = float(age_mean), float(age_std)
        self.noise_awgn, self.noise_mwgn = float(noise_awgn), float(noise_mwgn)
        self.seed = int(seed)
        self.sampling = ("stratified" if self.n_views > 1 else "random") \
            if sampling == "auto" else sampling

        man = load_manifest(cache_dir)
        self.entries = man["entries"]
        missing = [r["serial"] for r in self.records if r["serial"] not in self.entries]
        if missing:
            raise RuntimeError(
                f"{len(missing)} serials are not in the cache (e.g. {missing[:5]}). "
                f"Re-run scripts/01_build_cache.py.")

        too_short = [r["serial"] for r in self.records
                     if self.entries[r["serial"]]["n_samples"] < self.crop_len]
        if too_short:
            raise RuntimeError(f"recordings shorter than one crop: {too_short[:5]}")

        if normalize == "trainset":
            m, s = load_norm_stats(cache_dir)
            self._mean = m.reshape(N_EEG, 1)
            self._std = s.reshape(N_EEG, 1)
        else:
            self._mean = self._std = None

        self._mm = None          # opened lazily, once per DataLoader worker
        self._epoch = 0

    # DataLoader workers are separate processes on Windows (spawn); the memmap must be
    # opened inside the worker, not pickled from the parent.
    def _memmap(self):
        if self._mm is None:
            self._mm = open_memmap(self.cache_dir)
        return self._mm

    def set_epoch(self, epoch: int) -> None:
        """Makes train-time crop sampling vary per epoch AND stay reproducible."""
        self._epoch = int(epoch)

    def __len__(self):
        return len(self.records) * self.repeats

    def __getitem__(self, idx):
        rec = self.records[idx % len(self.records)]
        entry = self.entries[rec["serial"]]
        n = entry["n_samples"]

        if self.mode == "eval":
            starts = deterministic_starts(n, self.crop_len, self.n_views, self.t0)
        else:
            rng = np.random.default_rng(
                (self.seed * 1_000_003 + self._epoch * 10_007 + idx) % (2**63))
            if self.sampling == "stratified":
                starts = stratified_starts(rng, n, self.crop_len, self.n_views, self.t0)
            else:
                lo = min(self.t0, max(0, n - self.crop_len))
                hi = max(lo, n - self.crop_len)
                starts = rng.integers(lo, hi + 1, size=self.n_views)

        mm = self._memmap()
        off0 = entry["offset"]
        views = np.empty((self.n_views, N_EEG, self.crop_len), dtype=np.float32)
        for v, s in enumerate(starts):
            off = off0 + int(s) * N_EEG
            chunk = mm[off: off + self.crop_len * N_EEG].reshape(self.crop_len, N_EEG)
            views[v] = chunk.T.astype(np.float32)

        if self.normalize == "trainset":
            views = (views - self._mean) / self._std
        else:                                    # Phase-1 behaviour, ablation only
            mu = views.mean(axis=2, keepdims=True)
            sd = views.std(axis=2, keepdims=True) + 1e-8
            views = (views - mu) / sd

        if self.mode == "train" and (self.noise_awgn > 0 or self.noise_mwgn > 0):
            rng = np.random.default_rng(
                (self.seed * 7_919 + self._epoch * 104_729 + idx) % (2**63))
            if self.noise_mwgn > 0:
                views *= (1.0 + rng.normal(0, self.noise_mwgn, views.shape)).astype(np.float32)
            if self.noise_awgn > 0:
                views += rng.normal(0, self.noise_awgn, views.shape).astype(np.float32)

        age = (float(rec["age"]) - self.age_mean) / self.age_std
        return {
            "x": torch.from_numpy(views),                              # [V, 19, T]
            "age": torch.tensor([age], dtype=torch.float32),           # [1]
            "label": torch.tensor(rec["class_label"], dtype=torch.long),
            "index": torch.tensor(idx % len(self.records), dtype=torch.long),
        }


def make_loader(ds, batch_size, *, shuffle, num_workers, pin_memory=True, drop_last=False):
    from torch.utils.data import DataLoader
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=drop_last,
        persistent_workers=bool(num_workers),
        prefetch_factor=4 if num_workers else None,
    )
