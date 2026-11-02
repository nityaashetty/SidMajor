"""Per-recording out-of-fold predictions of a 5-fold CV run, broken down by clinical subgroup.

    python analyze_oof.py <run_dir> [<run_dir> ...]

Uses A_noprotocol's code copy (identical to Approach 11 apart from the ablation guard, which is a
no-op when the photic share is > 0). Checkpoints are only read. Writes oof_<run>.csv next to this file.
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "A_noprotocol"))

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402
import torch                                                                # noqa: E402
from sklearn.metrics import roc_auc_score                                   # noqa: E402

from src.common.config import load_config                                   # noqa: E402
from src.common.logging_utils import log                                    # noqa: E402
from src.training import checkpoint as ckpt                                 # noqa: E402
from src.training.engine import _loader, build_model, predict              # noqa: E402
from src.training.pipeline import labels, load_bundle, make_dataset        # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def oof(run_dir):
    cfg = load_config(run_dir / "config_used.yaml")
    cfg["train"]["num_workers"] = 0
    rows, test_p = [], []
    for k in range(cfg["cv"]["n_folds"]):
        b = load_bundle(cfg, fold=k)
        model = build_model(b).to(DEV)
        model.load_state_dict(ckpt.load(run_dir / f"fold{k}" / "final.pt", DEV)["model"])
        for sp in ("val", "test"):
            p, _ = predict(model, _loader(make_dataset(b, sp, train=False), cfg, shuffle=False), DEV)
            if sp == "val":
                rows += [{"serial": s, "age": a, "y": int(y), "p0": q[0], "p1": q[1], "p2": q[2], "fold": k}
                         for s, a, y, q in zip(b["mats"][sp]["serials"], b["mats"][sp]["age"], labels(b, sp), p)]
            else:
                test_p.append(p)
                test = pd.DataFrame({"serial": b["mats"][sp]["serials"], "age": b["mats"][sp]["age"], "y": labels(b, sp)})
    P = np.mean(test_p, 0)
    test["p0"], test["p1"], test["p2"], test["fold"] = P[:, 0], P[:, 1], P[:, 2], -1
    return pd.DataFrame(rows), test


def report(df, tag, sym):
    y, P = df["y"].to_numpy(), df[["p0", "p1", "p2"]].to_numpy()
    pr = P.argmax(1)
    rec = [(pr[y == k] == k).mean() for k in range(3)]
    cb = df["serial"].map(lambda s: "cb_normal" in sym[s]).to_numpy() & (y == 0)
    smi = df["serial"].map(lambda s: "smi" in sym[s]).to_numpy() & (y == 0)
    imp = 1 - P[:, 0]
    mci = y == 1

    def auc(neg, pos, score):
        m = neg | pos
        return roc_auc_score(pos[m], score[m]) if neg.any() and pos.any() else np.nan

    young, old = df["age"].to_numpy() < 70, df["age"].to_numpy() >= 70
    out = {"n": len(y), "acc": (pr == y).mean(), "bal": np.mean(rec),
           "auc_macro": roc_auc_score(y, P, multi_class="ovr", average="macro"),
           "rec_N": rec[0], "rec_MCI": rec[1], "rec_D": rec[2],
           "rec_cb_normal": (pr[cb] == 0).mean(), "rec_smi": (pr[smi] == 0).mean(),
           "auc_smi_vs_mci": auc(smi, mci, imp), "auc_cb_vs_mci": auc(cb, mci, imp),
           "auc_N_vs_MCI_young": auc((y == 0) & young, mci & young, imp),
           "auc_N_vs_MCI_old": auc((y == 0) & old, mci & old, imp)}
    log(f"  {tag:34s} n={out['n']:4d} acc {out['acc'] * 100:5.1f}% bal {out['bal'] * 100:5.1f}% AUC {out['auc_macro']:.3f} | "
        f"recall N/MCI/D {rec[0] * 100:.0f}/{rec[1] * 100:.0f}/{rec[2] * 100:.0f} | Normal recall: community "
        f"{out['rec_cb_normal'] * 100:.0f}% (n={cb.sum()}) vs SMI {out['rec_smi'] * 100:.0f}% (n={smi.sum()}) | "
        f"AUC impaired: SMI-vs-MCI {out['auc_smi_vs_mci']:.3f}, community-vs-MCI {out['auc_cb_vs_mci']:.3f}, "
        f"N-vs-MCI <70 {out['auc_N_vs_MCI_young']:.3f}, >=70 {out['auc_N_vs_MCI_old']:.3f}")
    return out


def main():
    root = HERE.parent
    sym = {r["serial"]: r["symptom"] for r in
           json.loads((root / "caueeg-dataset" / "annotation.json").read_text())["data"]}
    summary = []
    for rd in map(Path, sys.argv[1:]):
        name = rd.name
        log(f"analyze_oof: '{name}' - rebuilding out-of-fold predictions from 5 fold models")
        o, t = oof(rd)
        pd.concat([o, t]).to_csv(HERE / f"oof_{name}.csv", index=False)
        summary.append({"run": name, "split": "cv_oof", **report(o, f"{name} CV out-of-fold", sym)})
        summary.append({"run": name, "split": "test_ens5", **report(t, f"{name} test (5-fold ens)", sym)})
    out = HERE / "subgroup_summary.csv"
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
    new = pd.DataFrame(summary)
    if len(prev):
        prev = prev[~prev["run"].isin(new["run"])]
    pd.concat([prev, new]).to_csv(out, index=False)


if __name__ == "__main__":
    main()
