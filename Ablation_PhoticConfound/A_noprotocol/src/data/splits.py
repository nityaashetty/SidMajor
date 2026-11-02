"""CAUEEG-Dementia splits, read directly from the original dataset JSONs, with a hard leakage guard."""
import json
from pathlib import Path

CLASS_NAMES = ["Normal", "MCI", "Dementia"]
EVAL_SPLITS = ["val", "test", "test_no_overlap"]


def load_splits(dataset_root):
    root = Path(dataset_root)
    full = json.loads((root / "dementia.json").read_text(encoding="utf-8"))
    nov = json.loads((root / "dementia-no-overlap.json").read_text(encoding="utf-8"))

    def rows(items):
        return [{"serial": r["serial"], "age": float(r["age"]), "label": int(r["class_label"])} for r in items]

    splits = {
        "train": rows(full["train_split"]),
        "val": rows(full["validation_split"]),
        "test": rows(full["test_split"]),
        "val_no_overlap": rows(nov["validation_split"]),
        "test_no_overlap": rows(nov["test_split"]),
    }
    assert_no_leakage(splits, rows(nov["train_split"]))
    return splits


def _serials(rows):
    return {r["serial"] for r in rows}


def assert_no_leakage(splits, nov_train):
    train = _serials(splits["train"])
    for name in ("val", "test", "val_no_overlap", "test_no_overlap"):
        shared = train & _serials(splits[name])
        if shared:
            raise RuntimeError(f"LEAKAGE: {len(shared)} recordings in both train and {name}: {sorted(shared)[:5]}")
    if _serials(splits["val"]) & _serials(splits["test"]):
        raise RuntimeError("LEAKAGE: val and test share recordings")
    if _serials(nov_train) != train:
        raise RuntimeError("no-overlap train split differs from the full train split")
    if not _serials(splits["test_no_overlap"]) <= _serials(splits["test"]):
        raise RuntimeError("no-overlap test is not a subset of the full test split")


def all_task_serials(splits):
    return sorted(set().union(*(_serials(v) for v in splits.values())))
