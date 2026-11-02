"""Patient-level metrics in a fixed schema.

Phase-1 defect this fixes:
    train_npz.py selected the best epoch on raw val ACCURACY over 119 patients
    (line 136). With 119 samples the standard error is ~4.5 points, and the Phase-1
    run table (47.06% -> 58.82% across 8 runs) sits entirely inside that band -- those
    8 runs measured noise, not a ceiling. Here selection uses MACRO-F1 (class-balanced,
    the dataset is 367/334/249) and every run reports mean +/- std over seeds.

Because evaluation emits exactly one row per patient (views are aggregated inside the
model), everything below is already patient-level -- no epoch-level inflation is
possible, which is the failure mode the CAUEEG paper calls "epoch-based classification"
(their Table 6: 99%+ under that protocol vs 74.66% under the honest one).
"""
from __future__ import annotations

import numpy as np

CLASS_NAMES = ["Normal", "MCI", "Dementia"]


def confusion(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class(cm):
    out = {}
    for i, name in enumerate(CLASS_NAMES[:cm.shape[0]]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[name] = {"precision": float(prec), "recall": float(rec),
                     "f1": float(f1), "support": int(cm[i, :].sum())}
    return out


def roc_auc_ovr(y_true, probs):
    try:
        from sklearn.metrics import roc_auc_score
        present = np.unique(y_true)
        if len(present) < 2:
            return None
        if len(present) < probs.shape[1]:
            return None
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except Exception:
        return None


def compute(y_true, y_pred, probs=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion(y_true, y_pred)
    pc = per_class(cm)
    recalls = [v["recall"] for v in pc.values()]
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean([v["f1"] for v in pc.values()])),
        "per_class": pc,
        "confusion_matrix": cm.tolist(),
        "roc_auc_ovr_macro": roc_auc_ovr(y_true, probs) if probs is not None else None,
    }


def format_report(m: dict, title: str = "") -> str:
    L = []
    if title:
        L.append(title)
    L.append(f"  n={m['n']}  acc={m['accuracy']*100:.2f}%  "
             f"bal_acc={m['balanced_accuracy']*100:.2f}%  macro_f1={m['macro_f1']:.4f}"
             + (f"  auc={m['roc_auc_ovr_macro']:.4f}"
                if m.get("roc_auc_ovr_macro") else ""))
    L.append("  confusion (rows=true, cols=pred):")
    head = "           " + "".join(f"{n:>10}" for n in CLASS_NAMES)
    L.append("  " + head)
    for i, name in enumerate(CLASS_NAMES):
        L.append("  " + f"{name:>11}" +
                 "".join(f"{v:>10}" for v in m["confusion_matrix"][i]))
    L.append(f"  {'class':>11}{'prec':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for name, v in m["per_class"].items():
        L.append(f"  {name:>11}{v['precision']:>10.4f}{v['recall']:>10.4f}"
                 f"{v['f1']:>10.4f}{v['support']:>10}")
    return "\n".join(L)


def aggregate_seeds(runs: list[dict], key: str = "macro_f1") -> dict:
    """Mean +/- std across seeds. This, not a single best number, is what goes in the
    paper -- see the module docstring."""
    vals = {k: [] for k in ("accuracy", "balanced_accuracy", "macro_f1")}
    for r in runs:
        for k in vals:
            vals[k].append(r[k])
    return {
        "n_seeds": len(runs),
        **{f"{k}_mean": float(np.mean(v)) for k, v in vals.items()},
        **{f"{k}_std": float(np.std(v)) for k, v in vals.items()},
        "select_key": key,
    }
