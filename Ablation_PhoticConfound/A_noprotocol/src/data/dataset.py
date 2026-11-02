"""Recording-level dataset: a bag of K state-tagged 10 s segments + biomarker vector + age.

Training: fresh segments every epoch, drawn per state according to segments.state_mix
(eyes-closed favoured), only from clean stretches; amplitude/noise/channel-dropout
augmentation; and same-class bag mixing - part of a patient's bag is replaced with segments
of another patient with the SAME label (labels never mix, so no soft targets are needed).
Evaluation: deterministic, evenly spaced segments with the same state quotas.
use_events=false reproduces CEEDNet-style random crops with no state information (ablation).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .events import STATE_EC, STATE_EO, STATE_IDS, STATE_PHOTIC, STATE_UNKNOWN, window_starts
from .store import load_signal, load_timeline

SAMPLED_STATES = (STATE_EC, STATE_EO, STATE_PHOTIC)


class RecordingBags(Dataset):
    def __init__(self, records, bio, age_z, store_root, manifest, seg_cfg, aug_cfg, train, seed=0):
        self.records = records
        self.bio = np.asarray(bio, np.float32)
        self.age = np.asarray(age_z, np.float32)
        self.labels = np.array([r["label"] for r in records])
        self.store = store_root
        self.fs = int(manifest["fs"])
        self.scale = np.asarray(manifest["channel_scale_uv"], np.float32)[:, None]
        self.win = int(seg_cfg["seconds"])
        self.T = self.win * self.fs
        self.K = int(seg_cfg["train_per_recording"] if train else seg_cfg["eval_per_recording"])
        self.use_events = bool(seg_cfg["use_events"])
        self.clip = float(seg_cfg["signal_clip"])
        mix = seg_cfg["state_mix"]
        self.mix = np.array([mix["eyes_closed"], mix["eyes_open"], mix["photic"]], np.float64)
        self.aug = aug_cfg
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.cands = [self._candidates(r["serial"]) for r in records]
        self.by_class = {c: np.flatnonzero(self.labels == c) for c in np.unique(self.labels)}
        self.eval_plan = None if train else [self._eval_plan(i) for i in range(len(records))]

    # ---------- segment candidates ----------
    def _candidates(self, serial):
        state, clean, _ = load_timeline(self.store, serial)
        n_sec = len(state)
        if not self.use_events:
            starts = np.arange(10, max(n_sec - self.win, 11))  # anywhere after settling, no masks
            return {STATE_UNKNOWN: starts}
        c = {st: window_starts(state, clean, st, self.win) for st in SAMPLED_STATES}
        if sum(len(v) for v in c.values()) < 4:   # very few labelled windows: any clean resting EEG
            c = {STATE_UNKNOWN: window_starts(state, clean, None, self.win)}
        if sum(len(v) for v in c.values()) == 0:
            c = {STATE_UNKNOWN: np.arange(10, max(n_sec - self.win, 11))}
        c = {k: v for k, v in c.items() if len(v)}
        # ablation: states with a zero share are never sampled; fall back to unlabelled clean EEG if nothing is left
        keep = {k: v for k, v in c.items() if k == STATE_UNKNOWN or self.mix[SAMPLED_STATES.index(k)] > 0}
        if not keep:
            keep = {STATE_UNKNOWN: window_starts(state, clean, None, self.win)}
            if not len(keep[STATE_UNKNOWN]):
                keep = {STATE_UNKNOWN: np.arange(10, max(n_sec - self.win, 11))}
        return keep

    def _quota(self, cands, K, rng=None):
        """How many segments per state; states without windows hand their share to the others."""
        states = [s for s in SAMPLED_STATES if s in cands] or list(cands)
        if states == [STATE_UNKNOWN]:
            return {STATE_UNKNOWN: K}
        w = np.array([self.mix[SAMPLED_STATES.index(s)] for s in states])
        w = w / w.sum()
        if rng is not None:
            counts = rng.multinomial(K, w)
        else:
            counts = np.floor(w * K).astype(int)
            counts[np.argmax(w)] += K - counts.sum()
        return dict(zip(states, counts))

    def _eval_plan(self, i):
        cands = self.cands[i]
        plan = []
        for st, n in self._quota(cands, self.K).items():
            if n == 0:
                continue
            starts = cands[st]
            pick = starts[np.linspace(0, len(starts) - 1, n).round().astype(int)]
            plan += [(int(s) * self.fs, st) for s in pick]
        return plan

    def _train_plan(self, i, rng, K):
        cands = self.cands[i]
        plan = []
        for st, n in self._quota(cands, K, rng).items():
            starts = cands[st]
            s = rng.choice(starts, size=n, replace=len(starts) < n)
            jitter = rng.integers(0, self.fs, size=n)
            ok = np.isin(s + 1, starts) | (st == STATE_UNKNOWN)  # stay inside the clean stretch
            plan += [(int(a) * self.fs + int(j if o else 0), st) for a, j, o in zip(s, jitter, ok)]
        rng.shuffle(plan)
        return plan

    # ---------- loading ----------
    def _load(self, i, plan):
        x = load_signal(self.store, self.records[i]["serial"])
        n = x.shape[1]
        segs = []
        for start, _ in plan:
            start = min(start, n - self.T)
            segs.append(np.asarray(x[:, start:start + self.T], dtype=np.float32))
        segs = np.stack(segs) / self.scale
        return np.clip(segs, -self.clip, self.clip)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        if not self.train:
            plan = self.eval_plan[i]
            x = self._load(i, plan)
            return self._pack(x, [s for _, s in plan], self.bio[i], self.age[i], i)

        rng = np.random.default_rng((self.seed, self.epoch, i))
        a = self.aug
        bio, age = self.bio[i].copy(), self.age[i].copy()
        if rng.random() < a["bag_mix_prob"]:
            j = int(rng.choice(self.by_class[self.labels[i]]))
            frac = rng.uniform(*a["bag_mix_frac"])
            k_other = int(round(frac * self.K))
            plan = self._train_plan(i, rng, self.K - k_other)
            plan_j = self._train_plan(j, rng, k_other)
            x = np.concatenate([self._load(i, plan), self._load(j, plan_j)]) if k_other else self._load(i, plan)
            plan = plan + plan_j
            r = k_other / self.K
            bio = (1 - r) * bio + r * self.bio[j]
            age = (1 - r) * age + r * self.age[j]
        else:
            plan = self._train_plan(i, rng, self.K)
            x = self._load(i, plan)

        K = x.shape[0]
        x = x * rng.uniform(*a["amp_scale"], size=(K, 1, 1)).astype(np.float32)
        x = x + rng.normal(0, a["noise_std"], size=x.shape).astype(np.float32)
        drop = rng.random((K, x.shape[1], 1)) < a["channel_dropout"]
        x = np.where(drop, 0.0, x).astype(np.float32)
        bio = bio + rng.normal(0, a["bio_noise_std"], size=bio.shape).astype(np.float32)
        return self._pack(x, [s for _, s in plan], bio, age, i)

    def _pack(self, x, states, bio, age, i):
        if not self.use_events:
            states = [STATE_UNKNOWN] * len(states)
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            "state": torch.tensor(states, dtype=torch.long),
            "mask": torch.ones(len(states), dtype=torch.bool),
            "bio": torch.as_tensor(bio, dtype=torch.float32),
            "age": torch.as_tensor(np.atleast_1d(age), dtype=torch.float32),
            "label": torch.tensor(int(self.labels[i])),
            "index": torch.tensor(i),
        }


N_STATES = len(STATE_IDS)
