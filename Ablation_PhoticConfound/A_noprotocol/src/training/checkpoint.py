"""Atomic checkpoints (a crash mid-save never corrupts the previous file)."""
import os
from pathlib import Path

import torch


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)
