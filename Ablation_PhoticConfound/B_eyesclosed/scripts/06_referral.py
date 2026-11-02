"""Step 6: selective prediction - low-confidence recordings are 'referred' instead of classified.

Confidence = the ensemble's highest class probability. Cut-offs are calibrated on VAL only
(the confidence below which a target share of val recordings would be referred) and then
applied unchanged to test / test_no_overlap. Writes referral_report.txt + referral.json
into the run folder and lists the referred test recordings.

    python scripts/06_referral.py                          # run sanf_net, final.pt, 30% listed
    python scripts/06_referral.py --name sanf_net --list-rate 0.2
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.common.config import add_common_args, load_config
from src.common.logging_utils import log, write_json
from src.data.splits import CLASS_NAMES
from src.evaluation.metrics import compute_metrics
from src.training import checkpoint as ckpt
from src.training.engine import _loader, build_model, predict
from src.training.pipeline import eval_datasets, labels, load_bundle

RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SHORT = ["N", "MCI", "D"]


def binary_impaired(y, P):
    imp, pred = y != 0, P.argmax(1) != 0
    tp, tn = int((imp & pred).sum()), int((~imp & ~pred).sum())
    from sklearn.metrics import roc_auc_score
    return {"acc": (tp + tn) / len(y), "sensitivity": tp / imp.sum(), "specificity": tn / (~imp).sum(),
            "auc": float(roc_auc_score(imp, 1 - P[:, 0])), "tp": tp, "n_imp": int(imp.sum()), "tn": tn,
            "n_norm": int((~imp).sum())}


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--name", default="sanf_net")
    ap.add_argument("--checkpoint", choices=["final", "best"], default="final")
    ap.add_argument("--list-rate", type=float, default=0.3, help="referral rate whose referred test cases are listed")
    args = ap.parse_args()
    base = load_config(args.config, args.set)
    run_dir = Path(base["paths"]["runs"]) / args.name
    cfg = load_config(run_dir / "config_used.yaml")
    if cfg.get("cv"):
        raise SystemExit("use a seed run (CV runs have no shared val set to calibrate on)")
    cfg["train"]["num_workers"] = 0          # in-process loading: no Windows worker start-up cost
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_bundle(cfg)
    splits = ("val", "test", "test_no_overlap")
    loaders = {sp: _loader(ds, cfg, shuffle=False) for sp, ds in eval_datasets(bundle).items() if sp in splits}
    members = sorted(p for p in run_dir.glob("seed*") if (p / f"{args.checkpoint}.pt").exists())
    log(f"referral: '{args.name}' [{args.checkpoint}] ensemble of {len(members)} models; calibrating on val")
    probs = {sp: [] for sp in splits}
    for m_dir in members:
        model = build_model(bundle).to(device)
        model.load_state_dict(ckpt.load(m_dir / f"{args.checkpoint}.pt", device)["model"])
        for sp in splits:
            probs[sp].append(predict(model, loaders[sp], device)[0])
    P = {sp: np.mean(v, axis=0) for sp, v in probs.items()}
    Y = {sp: labels(bundle, sp) for sp in splits}
    conf = {sp: P[sp].max(1) for sp in splits}

    raw = json.loads((Path(cfg["paths"]["dataset_root"]) / "dementia.json").read_text(encoding="utf-8"))
    info = {r["serial"]: r for k in ("train_split", "validation_split", "test_split") for r in raw[k]}

    lines, table = [], []
    lines.append(f"SELECTIVE PREDICTION - run '{args.name}' [{args.checkpoint}], {len(members)}-model ensemble")
    lines.append("Confidence = highest class probability. Cut-offs calibrated on VAL, applied unchanged to test.")
    lines.append("")
    for sp in ("test", "test_no_overlap"):
        lines.append(f"=== {sp} (n={len(Y[sp])}) ===")
        lines.append(f"{'target':>7} {'cut-off':>7} {'referred':>9} {'acc kept':>9} {'F1 kept':>8} "
                     f"{'recall kept N/MCI/D':>20}   {'referred by true class N/MCI/D':>31}  MCI referred")
        for r in RATES:
            t = float(np.quantile(conf["val"], r)) if r > 0 else -1.0
            keep = conf[sp] > t
            ref = ~keep
            m = compute_metrics(Y[sp][keep], P[sp][keep])
            rec = "/".join(f"{m['per_class'][c]['recall'] * 100:3.0f}" for c in CLASS_NAMES)
            comp = "/".join(f"{int((Y[sp][ref] == k).sum()):2d}" for k in range(3))
            mci_ref = (ref & (Y[sp] == 1)).sum() / (Y[sp] == 1).sum()
            lines.append(f"{r * 100:6.0f}% {t:7.3f} {ref.mean() * 100:8.1f}% {m['accuracy'] * 100:8.1f}% "
                         f"{m['macro_f1']:8.4f} {rec:>20}   {comp:>31}  {mci_ref * 100:5.1f}%")
            table.append({"split": sp, "target_rate": r, "cutoff": t, "referred": float(ref.mean()),
                          "acc_kept": m["accuracy"], "f1_kept": m["macro_f1"],
                          "recall_kept": {c: m["per_class"][c]["recall"] for c in CLASS_NAMES},
                          "referred_by_class": [int((Y[sp][ref] == k).sum()) for k in range(3)]})
        for r in (0.0, 0.2, 0.3):
            t = float(np.quantile(conf["val"], r)) if r > 0 else -1.0
            pred = np.where(conf[sp] > t, P[sp].argmax(1), 3)            # 3 = referred
            title = "no referral" if r == 0 else f"{r * 100:.0f}% target referral (cut-off {t:.3f})"
            lines.append(f"  Confusion, {title}:   rows = true class, columns = model output")
            lines.append(f"    {'':10s}{'N':>6}{'MCI':>6}{'Dem':>6}{'Referred':>10}{'Total':>7}   recall (of kept)")
            for k, cname in enumerate(["Normal", "MCI", "Dementia"]):
                row = [int(((Y[sp] == k) & (pred == j)).sum()) for j in range(4)]
                kept = sum(row[:3])
                lines.append(f"    {cname:10s}{row[0]:6d}{row[1]:6d}{row[2]:6d}{row[3]:10d}{sum(row):7d}   "
                             f"{row[k] / kept * 100 if kept else 0:5.1f}%")
            tot = [int((pred == j).sum()) for j in range(4)]
            lines.append(f"    {'Total':10s}{tot[0]:6d}{tot[1]:6d}{tot[2]:6d}{tot[3]:10d}{len(pred):7d}")
            lines.append("")
        b = binary_impaired(Y[sp], P[sp])
        lines.append(f"  Normal vs impaired (MCI+Dementia), no referral: acc {b['acc'] * 100:.1f}% | "
                     f"impaired caught {b['sensitivity'] * 100:.1f}% ({b['tp']}/{b['n_imp']}) | "
                     f"normals kept {b['specificity'] * 100:.1f}% ({b['tn']}/{b['n_norm']}) | AUC {b['auc']:.3f}")
        lines.append("")

    t = float(np.quantile(conf["val"], args.list_rate))
    ref_idx = np.flatnonzero(conf["test"] <= t)
    ref_idx = ref_idx[np.argsort(conf["test"][ref_idx])]
    lines.append(f"=== REFERRED TEST RECORDINGS at {args.list_rate * 100:.0f}% val-calibrated rate "
                 f"(confidence <= {t:.3f}): {len(ref_idx)} of {len(Y['test'])} ===")
    lines.append(f"{'#':>3} {'serial':>6} {'age':>4} {'true':>5} {'pred':>5} {'P(N)':>6} {'P(MCI)':>7} {'P(D)':>6} "
                 f"{'conf':>6}  subtype")
    serials = bundle["mats"]["test"]["serials"]
    listed = []
    for n, i in enumerate(ref_idx, 1):
        s = serials[i]
        sub = ", ".join(x for x in info[s]["symptom"] if x not in ("normal", "mci", "dementia")) or "-"
        p = P["test"][i]
        flag = "" if p.argmax() == Y["test"][i] else "  (would be wrong)"
        lines.append(f"{n:3d} {s:>6} {int(info[s]['age']):4d} {SHORT[Y['test'][i]]:>5} {SHORT[p.argmax()]:>5} "
                     f"{p[0]:6.2f} {p[1]:7.2f} {p[2]:6.2f} {conf['test'][i]:6.3f}  {sub}{flag}")
        listed.append({"serial": s, "age": info[s]["age"], "true": SHORT[Y["test"][i]], "pred": SHORT[p.argmax()],
                       "probs": p.round(4).tolist(), "confidence": float(conf["test"][i]), "symptom": info[s]["symptom"]})
    n_wrong = sum(1 for d in listed if d["true"] != d["pred"])
    by_true = {c: sum(1 for d in listed if d["true"] == c) for c in SHORT}
    lines.append(f"Referred: {by_true} | {n_wrong} of {len(listed)} would have been misclassified if forced")

    report = "\n".join(lines)
    (run_dir / "referral_report.txt").write_text(report, encoding="utf-8")
    write_json(run_dir / "referral.json", {"run": args.name, "checkpoint": args.checkpoint, "table": table,
                                           "listed_rate": args.list_rate, "listed_cutoff": t, "referred_test": listed})
    print(report)
    row = [r for r in table if r["split"] == "test" and r["target_rate"] == args.list_rate][0]
    log(f"referral: at {args.list_rate * 100:.0f}% (val-calibrated) test referred {row['referred'] * 100:.1f}%, "
        f"acc on kept {row['acc_kept'] * 100:.1f}% (vs {table[0]['acc_kept'] * 100:.1f}% with no referral); "
        f"report -> {run_dir / 'referral_report.txt'}")


if __name__ == "__main__":
    main()
