"""Tabular baselines under the same 5-fold CV as SANF-Net (CPU only).

    python baselines_cv.py [name ...] [--cv-seed S]   # all baselines, or only the named ones;
                                                      # S > 0 = repeated-CV fold assignment, saved as <name>_rS

For each fold the normative model is refit on that fold's training Normals (exactly as in
SANF-Net's CV), a model is fit on the fold's training part, and it predicts the held-out part
(-> out-of-fold, n=1069) and the official test set (averaged over the 5 fold models).
Writes oof/<name>.csv with columns serial, age, y, p0, p1, p2, fold (fold=-1 for test rows).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A11 = HERE.parents[1] / "Approach11_StateAwareNormative"
sys.path.insert(0, str(A11))

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier                 # noqa: E402
from sklearn.linear_model import LogisticRegression                         # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold           # noqa: E402
from sklearn.pipeline import make_pipeline                                  # noqa: E402
from sklearn.preprocessing import StandardScaler                            # noqa: E402

from src.common.config import load_config                                   # noqa: E402
from src.data.splits import load_splits                                     # noqa: E402
from src.features.normative import build_feature_matrices                   # noqa: E402
from src.training.pipeline import fold_splits                               # noqa: E402

OUT = HERE / "oof"


def lr():
    return GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
                        {"logisticregression__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}, scoring="f1_macro",
                        cv=StratifiedKFold(5, shuffle=True, random_state=0))


def hgb():
    return GridSearchCV(HistGradientBoostingClassifier(class_weight="balanced", random_state=0, max_iter=300,
                                                       early_stopping=False),
                        {"learning_rate": [0.03], "max_depth": [2, 3], "min_samples_leaf": [20, 40],
                         "l2_regularization": [1.0]}, scoring="f1_macro",
                        cv=StratifiedKFold(5, shuffle=True, random_state=0))


# name -> (normative?, design, model factory)
BASELINES = {
    "base_age_lr": (True, "age", lr),
    "base_bio_raw_lr": (False, "bio", lr),
    "base_bio_norm_lr": (True, "bio", lr),
    "base_bioage_norm_lr": (True, "bio+age", lr),
    "base_bioage_norm_hgb": (True, "bio+age", hgb),
    "base_bioage_norm_lr_noproto": (True, "bio+age-proto", lr),
}
PROTOCOL = ("has_photic", "has_eo", "photic_driving_snr")   # recording-protocol features (confound check)


def design(m, kind, names):
    age = m["age"][:, None]
    keep = [i for i, n in enumerate(names) if n not in PROTOCOL]
    return {"age": age, "bio": m["Z"], "bio+age": np.hstack([m["Z"], age]),
            "bio+age-proto": np.hstack([m["Z"][:, keep], age])}[kind]


def main():
    cfg = load_config(None)
    splits = load_splits(cfg["paths"]["dataset_root"])
    df = pd.read_csv(Path(cfg["paths"]["features"]) / "biomarkers.csv", dtype={"serial": str})
    OUT.mkdir(exist_ok=True)
    args = sys.argv[1:]
    seed = int(args[args.index("--cv-seed") + 1]) if "--cv-seed" in args else 0
    only = [a for i, a in enumerate(args) if a != "--cv-seed" and (i == 0 or args[i - 1] != "--cv-seed")]
    for base, (normative, kind, factory) in BASELINES.items():
        if only and base not in only:
            continue
        name = base if seed == 0 else f"{base}_r{seed}"
        rows, test_p = [], []
        for k in range(5):
            sp = fold_splits(splits, 5, k, seed=seed)
            mats, names, _ = build_feature_matrices(df, sp, dict(cfg["features"], normative=normative))
            clf = factory().fit(design(mats["train"], kind, names), mats["train"]["y"])
            p = clf.predict_proba(design(mats["val"], kind, names))
            m = mats["val"]
            rows += [{"serial": s, "age": a, "y": int(y), "p0": q[0], "p1": q[1], "p2": q[2], "fold": k}
                     for s, a, y, q in zip(m["serials"], m["age"], m["y"], p)]
            test_p.append(clf.predict_proba(design(mats["test"], kind, names)))
        t = mats["test"]
        P = np.mean(test_p, 0)
        test = pd.DataFrame({"serial": t["serials"], "age": t["age"], "y": t["y"],
                             "p0": P[:, 0], "p1": P[:, 1], "p2": P[:, 2], "fold": -1})
        out = pd.concat([pd.DataFrame(rows), test])
        out.to_csv(OUT / f"{name}.csv", index=False)
        o = out[out.fold >= 0]
        print(f"{name:24s} OOF acc {(o[['p0', 'p1', 'p2']].to_numpy().argmax(1) == o.y).mean() * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
