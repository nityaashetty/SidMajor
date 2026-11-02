"""Step 3: biomarker + age models without deep learning (minutes, CPU only).

Answers three questions before any GPU time is spent:
  * how far age alone goes,
  * whether the event-aware biomarkers carry signal beyond age,
  * whether age-normative z-scoring beats plain standardisation.
Hyperparameters are picked by 5-fold CV on TRAIN only; val/test are scored once.
The variant with the best CV macro-F1 is saved for late fusion in 05_evaluate.py.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import add_common_args, load_config
from src.common.logging_utils import log, write_json
from src.data.splits import EVAL_SPLITS, load_splits
from src.evaluation.metrics import compute_metrics, fmt
from src.features.normative import build_feature_matrices


def design(mats, split, kind):
    m = mats[split]
    age = m["age"][:, None]
    if kind == "age":
        return age
    return np.hstack([m["Z"], age]) if kind.endswith("+age") else m["Z"]


def models():
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    lr = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
                      {"logisticregression__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}, scoring="f1_macro", cv=cv)
    gb = GridSearchCV(HistGradientBoostingClassifier(class_weight="balanced", random_state=0, max_iter=300,
                                                     early_stopping=False),
                      {"learning_rate": [0.03], "max_depth": [2, 3], "min_samples_leaf": [20, 40],
                       "l2_regularization": [1.0]}, scoring="f1_macro", cv=cv)
    return {"logreg": lr, "hgb": gb}


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    splits = load_splits(cfg["paths"]["dataset_root"])
    feat_dir = Path(cfg["paths"]["features"])
    df = pd.read_csv(feat_dir / "biomarkers.csv", dtype={"serial": str})

    results, best = {}, None
    for normative in (True, False):
        fcfg = dict(cfg["features"], normative=normative)
        mats, names, _ = build_feature_matrices(df, splits, fcfg)
        tag = "normative" if normative else "standardised"
        kinds = ["age", "bio", "bio+age"] if normative else ["bio", "bio+age"]
        for kind in kinds:
            for mname, est in models().items():
                key = f"{kind}|{tag}|{mname}" if kind != "age" else f"age|{mname}"
                if key in results:
                    continue
                est.fit(design(mats, "train", kind), mats["train"]["y"])
                res = {"cv_macro_f1": float(est.best_score_), "params": est.best_params_}
                probs = {}
                for sp in EVAL_SPLITS:
                    probs[sp] = est.predict_proba(design(mats, sp, kind))
                    res[sp] = compute_metrics(mats[sp]["y"], probs[sp])
                results[key] = res
                log(f"tabular {key:32s} cv-F1 {res['cv_macro_f1']:.4f} | test {fmt(res['test'])} | "
                    f"no-overlap acc {res['test_no_overlap']['accuracy'] * 100:.2f}%")
                if kind != "age" and (best is None or res["cv_macro_f1"] > best[1]):
                    best = (key, res["cv_macro_f1"], {sp: (probs[sp], mats[sp]["serials"]) for sp in EVAL_SPLITS})

    key, cvf1, probs = best
    np.savez(feat_dir / "tabular_probs.npz", key=key,
             **{f"{sp}_probs": p for sp, (p, _) in probs.items()},
             **{f"{sp}_serials": np.array(s) for sp, (_, s) in probs.items()})
    write_json(feat_dir / "tabular_results.json", {"selected_for_fusion": key, "results": results})
    log(f"tabular: selected for fusion (by train-CV F1 {cvf1:.4f}): {key}")


if __name__ == "__main__":
    main()
