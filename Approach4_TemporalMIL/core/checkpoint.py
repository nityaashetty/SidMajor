"""Crash-safe, fully resumable checkpointing.

Phase-1 defect this fixes:
    train_npz.py:139 saved ONLY model.state_dict(). An interrupted run could not be
    resumed (optimiser moments, LR-schedule position and RNG state were all lost), and
    a .pth file carried no record of the config that produced it -- which is exactly
    why outputs/qsfe_run7_58.82.pth is now an orphan whose architecture (s1=19) no
    longer matches any code in the repo.

Here every checkpoint carries model + optimiser + scheduler + AMP scaler + epoch +
best-metric + full history + all four RNG states + the config fingerprint, and is
written atomically (temp file then os.replace) so a crash mid-save cannot corrupt it.
This matters because the three approaches run unattended on three different machines.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

LAST = "last.pt"
BEST = "best.pt"


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(s):
    if not s:
        return
    try:
        random.setstate(s["python"])
        np.random.set_state(s["numpy"])
        torch.set_rng_state(s["torch"].cpu() if torch.is_tensor(s["torch"]) else s["torch"])
        if s.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])
    except Exception as e:                       # never let RNG restore kill a resume
        print(f"[ckpt] warning: could not restore RNG state ({e})")


def save(path, *, model, optimizer=None, scheduler=None, scaler=None, epoch=0,
         best=-float("inf"), history=None, fingerprint="", extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best": float(best),
        "history": history or [],
        "fingerprint": fingerprint,
        "rng": _rng_state(),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)                        # atomic on NTFS and POSIX


def load(path, *, model=None, optimizer=None, scheduler=None, scaler=None,
         map_location="cpu", restore_rng=True, fingerprint=None, strict_fp=False):
    path = Path(path)
    ck = torch.load(path, map_location=map_location, weights_only=False)

    if fingerprint and ck.get("fingerprint") and ck["fingerprint"] != fingerprint:
        msg = (f"Checkpoint fingerprint {ck['fingerprint']} != current config "
               f"{fingerprint}. The scientific config changed since this checkpoint "
               f"was written.")
        if strict_fp:
            raise RuntimeError(msg + " Refusing to resume; delete the checkpoint or "
                                     "restore the old config.")
        print(f"[ckpt] WARNING: {msg}")

    if model is not None:
        model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    if restore_rng:
        _restore_rng(ck.get("rng"))
    return ck


def find_resume(run_dir) -> Path | None:
    p = Path(run_dir) / LAST
    return p if p.exists() else None


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
