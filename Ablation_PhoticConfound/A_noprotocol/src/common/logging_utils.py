"""Console + PROGRESS.log logging (the live-progress window tails PROGRESS.log) and seeding."""
import json
import random
import time
from pathlib import Path

import numpy as np

from .config import APPROACH_ROOT

PROGRESS_LOG = APPROACH_ROOT.parent / "PROGRESS.log"  # ablation: one shared log


def log(msg, progress=True):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if progress:
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
