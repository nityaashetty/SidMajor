"""Recording-level metrics in one fixed schema, plus mean/std aggregation over seeds."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from ..data.splits import CLASS_NAMES

HEADLINE = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr_macro")


def compute_metrics(y, probs):
    y = np.asarray(y)
    probs = np.asarray(probs, np.float64)
    pred = probs.argmax(1)
    labels = list(range(len(CLASS_NAMES)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    try:
        auc = float(roc_auc_score(y, probs / probs.sum(1, keepdims=True), multi_class="ovr",
                                  average="macro", labels=labels))
    except ValueError:
        auc = float("nan")
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "roc_auc_ovr_macro": auc,
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASS_NAMES)},
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def aggregate(metric_dicts):
    out = {"n_runs": len(metric_dicts)}
    for k in HEADLINE:
        v = np.array([m[k] for m in metric_dicts], np.float64)
        out[f"{k}_mean"], out[f"{k}_std"] = float(np.nanmean(v)), float(np.nanstd(v))
    return out


def fmt(m):
    return (f"acc {m['accuracy'] * 100:.2f}%  bal {m['balanced_accuracy'] * 100:.2f}%  "
            f"F1 {m['macro_f1']:.4f}  AUC {m['roc_auc_ovr_macro']:.4f}")


def fmt_agg(a):
    return (f"acc {a['accuracy_mean'] * 100:.2f}+/-{a['accuracy_std'] * 100:.2f}%  "
            f"F1 {a['macro_f1_mean']:.4f}+/-{a['macro_f1_std']:.4f}  AUC {a['roc_auc_ovr_macro_mean']:.4f}")
