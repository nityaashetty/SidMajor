"""Step 5: recording-level evaluation on val, test and test_no_overlap.

Reports, in one results.json:
  * per seed and mean +/- std over seeds          (the honest single-model number)
  * seed ensemble (probabilities averaged)        (cheap ensemble, CEEDNet-style)
  * seed ensemble + tabular biomarker model       (late fusion, weight chosen on VAL only)
  * mean modality gates per class                 (how much EEG / biomarkers / age were used)

    python scripts/05_evaluate.py                       # run = config `experiment`, checkpoint = eval.checkpoint
    python scripts/05_evaluate.py --name no_events --checkpoint best
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
import yaml

from src.common.config import add_common_args, load_config
from src.common.logging_utils import log, write_json
from src.data.splits import CLASS_NAMES, EVAL_SPLITS
from src.evaluation.metrics import aggregate, compute_metrics, fmt, fmt_agg
from src.training import checkpoint as ckpt
from src.training.engine import _loader, build_model, predict
from src.training.pipeline import eval_datasets, labels, load_bundle

REFERENCE = {"CEEDNet best single model (paper Table 4)": 0.7018, "CEEDNet 9-model ensemble": 0.7466,
             "CEEDNet ensemble, no-overlap test (paper Table 7)": 0.7666}


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--name", default=None)
    ap.add_argument("--checkpoint", choices=["final", "best"], default=None)
    args = ap.parse_args()
    base = load_config(args.config, args.set)
    name = args.name or base["experiment"]
    run_dir = Path(base["paths"]["runs"]) / name
    used = run_dir / "config_used.yaml"
    if not used.exists():
        raise SystemExit(f"no run at {run_dir} - train it first with scripts/04_train.py --name {name}")
    cfg = load_config(used)                       # evaluate exactly what was trained
    which = args.checkpoint or base["eval"]["checkpoint"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cv = cfg.get("cv")
    if cv:   # one model per fold; each fold has its own normative model / age scaling, so its own bundle
        members = [(f"fold{k}", k) for k in range(cv["n_folds"]) if (run_dir / f"fold{k}" / f"{which}.pt").exists()]
    else:
        members = [(p.name, None) for p in sorted(run_dir.glob("seed*")) if (p / f"{which}.pt").exists()]
    if not members:
        raise SystemExit(f"no {which}.pt checkpoints in {run_dir}")
    # with CV, 'val' is each fold's own held-out part: scored per fold and pooled out-of-fold, never averaged
    ens_splits = [sp for sp in EVAL_SPLITS if not (cv and sp == "val")]
    shared = None if cv else load_bundle(cfg)
    per_seed, probs, gates = [], {sp: [] for sp in EVAL_SPLITS}, {sp: [] for sp in EVAL_SPLITS}
    oof_y, oof_p, test_serials = [], [], None
    for member, fold in members:
        bundle = load_bundle(cfg, fold=fold) if cv else shared
        loaders = {sp: _loader(ds, cfg, shuffle=False) for sp, ds in eval_datasets(bundle).items()}
        ys = {sp: labels(bundle, sp) for sp in EVAL_SPLITS}
        serials = [bundle["mats"][sp]["serials"] for sp in ens_splits]
        if test_serials is not None and serials != test_serials:
            raise SystemExit("test recordings differ between members - cannot ensemble")
        test_serials = serials
        model = build_model(bundle).to(device)
        model.load_state_dict(ckpt.load(run_dir / member / f"{which}.pt", device)["model"])
        row = {"member": member}
        for sp in EVAL_SPLITS:
            p, g = predict(model, loaders[sp], device)
            row[sp] = compute_metrics(ys[sp], p)
            if sp in ens_splits:
                probs[sp].append(p)
                gates[sp].append(g)
            else:
                oof_y.append(ys[sp])
                oof_p.append(p)
        per_seed.append(row)
        held = "held-out fold" if cv else "val"
        log(f"eval '{name}' {member} [{which}]: {held} acc {row['val']['accuracy'] * 100:.2f}% | test {fmt(row['test'])} "
            f"| no-overlap acc {row['test_no_overlap']['accuracy'] * 100:.2f}%")

    result = {"run": name, "checkpoint": which, "members": [m for m, _ in members], "modalities": model.modalities,
              "per_seed": per_seed, "mean_over_seeds": {}, "seed_ensemble": {}, "reference_accuracy": REFERENCE}
    ens = {sp: np.mean(probs[sp], axis=0) for sp in ens_splits}
    for sp in EVAL_SPLITS:
        result["mean_over_seeds"][sp] = aggregate([r[sp] for r in per_seed])
    for sp in ens_splits:
        result["seed_ensemble"][sp] = compute_metrics(ys[sp], ens[sp])
    if cv:
        result["cv_out_of_fold"] = compute_metrics(np.concatenate(oof_y), np.concatenate(oof_p))

    g_test = np.mean(gates["test"], axis=0)
    result["mean_gates_by_true_class"] = {
        c: {m: float(g_test[ys["test"] == k][:, j].mean()) for j, m in enumerate(model.modalities)}
        for k, c in enumerate(CLASS_NAMES)}

    tab_path = Path(cfg["paths"]["features"]) / "tabular_probs.npz"
    if base["eval"]["fuse_tabular"] and tab_path.exists() and not cv:
        tab = np.load(tab_path, allow_pickle=False)
        for sp in EVAL_SPLITS:
            if list(tab[f"{sp}_serials"]) != bundle["mats"][sp]["serials"]:
                raise SystemExit("tabular_probs.npz order does not match - re-run 03_tabular_baseline.py")
        grid = np.round(np.arange(0.0, 1.01, 0.1), 2)
        val_f1 = [compute_metrics(ys["val"], (1 - w) * ens["val"] + w * tab["val_probs"])["macro_f1"] for w in grid]
        w = float(grid[int(np.argmax(val_f1))])
        result["fusion_with_tabular"] = {
            "tabular_model": str(tab["key"]), "weight_tabular": w, "chosen_on": "val macro-F1",
            **{sp: compute_metrics(ys[sp], (1 - w) * ens[sp] + w * tab[f"{sp}_probs"]) for sp in EVAL_SPLITS}}

    out = run_dir / ("results.json" if which == "final" else f"results_{which}.json")
    write_json(out, result)

    log(f"===== '{name}' [{which}] ({len(members)} {'folds' if cv else 'seeds'}) =====")
    if cv:
        log(f"  CV out-of-fold   (n={result['cv_out_of_fold']['n']}) : {fmt(result['cv_out_of_fold'])}")
    for sp in ("test", "test_no_overlap"):
        log(f"  {sp:16s} mean+/-std : {fmt_agg(result['mean_over_seeds'][sp])}")
        log(f"  {sp:16s} seed-ens : {fmt(result['seed_ensemble'][sp])}")
        if "fusion_with_tabular" in result:
            log(f"  {sp:16s} +tabular : {fmt(result['fusion_with_tabular'][sp])} "
                f"(w_tab={result['fusion_with_tabular']['weight_tabular']})")
    cm = result["seed_ensemble"]["test"]["confusion_matrix"]
    log(f"  test confusion (rows=true N/MCI/D): {cm} | gates by class: "
        + "; ".join(f"{c}: " + ", ".join(f"{m}={v:.2f}" for m, v in d.items())
                    for c, d in result["mean_gates_by_true_class"].items()))
    log(f"  saved {out}")


if __name__ == "__main__":
    main()
