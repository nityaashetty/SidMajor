"""Split loading with a hard leakage guard.

Why the guard is not optional:
    CAUEEG-Abnormal's train split was built by starting from the Dementia train split
    and topping up with random draws from the REST of the corpus -- including Dementia's
    own val and test recordings. Verified on this dataset:

        abnormal_train INTERSECT dementia_test       = 36   (30.5% of the test set)
        abnormal_train INTERSECT dementia_validation = 36   (30.3% of the val set)
        abnormal_train entirely new vs dementia      = 85

    So any multi-task / extra-data idea that trains on abnormal_train silently trains on
    a third of the test set. assert_no_contamination() below exists so that mistake
    fails loudly instead of producing a great-looking number.
"""
from __future__ import annotations

import json
from pathlib import Path

CLASS_NAMES = ["Normal", "MCI", "Dementia"]
TASK_FILES = {
    "dementia": "dementia.json",
    "dementia-no-overlap": "dementia-no-overlap.json",
}


class LeakageError(RuntimeError):
    pass


def load_splits(dataset_root: str | Path, task: str = "dementia", verbose: bool = True):
    if task not in TASK_FILES:
        raise ValueError(f"task must be one of {list(TASK_FILES)}, got {task!r}")
    path = Path(dataset_root) / TASK_FILES[task]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    d = json.loads(path.read_text(encoding="utf-8"))

    splits = {
        "train": d["train_split"],
        "val": d["validation_split"],
        "test": d["test_split"],
    }
    for name, recs in splits.items():
        for r in recs:
            if r.get("age") is None:
                raise ValueError(
                    f"record {r['serial']} in {name} has age=null; "
                    "the age feature cannot be used with this split file")

    assert_no_leakage(splits)
    if verbose:
        print(describe_splits(splits, task))
    return splits


def assert_no_leakage(splits: dict) -> None:
    s = {k: {r["serial"] for r in v} for k, v in splits.items()}
    problems = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = s[a] & s[b]
        if inter:
            problems.append(f"{a} INTERSECT {b} = {len(inter)} "
                            f"(e.g. {sorted(inter)[:5]})")
    if problems:
        raise LeakageError("Recording-level split leakage detected:\n  "
                           + "\n  ".join(problems))


def assert_no_contamination(extra_serials, splits: dict, what: str = "extra data") -> None:
    """Call this before adding ANY recordings outside train_split to training."""
    extra = set(extra_serials)
    bad = {}
    for name in ("val", "test"):
        inter = extra & {r["serial"] for r in splits[name]}
        if inter:
            bad[name] = sorted(inter)
    if bad:
        msg = "\n  ".join(
            f"{what} INTERSECT {k} = {len(v)} recordings (e.g. {v[:5]})"
            for k, v in bad.items())
        raise LeakageError(
            f"Refusing to train on {what}: it overlaps held-out data.\n  {msg}\n"
            f"  Remove these serials before training, or your reported "
            f"accuracy is invalid.")


def describe_splits(splits: dict, task: str) -> str:
    lines = [f"task: {task}"]
    for name, recs in splits.items():
        counts = [0, 0, 0]
        for r in recs:
            counts[r["class_label"]] += 1
        lines.append(f"  {name:5s} n={len(recs):4d}  " +
                     "  ".join(f"{CLASS_NAMES[i]}={counts[i]}" for i in range(3)))
    lines.append("  leakage check: PASS (no recording appears in two splits)")
    return "\n".join(lines)


def class_weights(train_records) -> list[float]:
    """Inverse-frequency weights, matching ProjectAttempt1/src/train/train_npz.py:24-30."""
    counts = [0, 0, 0]
    for r in train_records:
        counts[r["class_label"]] += 1
    total = sum(counts)
    return [total / (len(counts) * c) if c else 0.0 for c in counts]


def age_stats(train_records):
    import numpy as np
    a = np.array([r["age"] for r in train_records], dtype=np.float32)
    return float(a.mean()), float(a.std() + 1e-8)
