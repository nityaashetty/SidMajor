"""Step 3 -- evaluate every seed on the test set and write results.json.

Protocol (identical on all three devices, which is what makes the three results
comparable):
  * one row per PATIENT, never per crop
  * `eval_views` deterministic evenly-spaced crops, aggregated inside the model
  * best.pt selected on val macro-F1, not val accuracy
  * reported as mean +/- std over seeds
  * scored on BOTH the standard test split and the no-overlap test split

The no-overlap run matters: ~20-24% of the standard test set shares a patient with
training (different recording dates). The CAUEEG authors publish the no-overlap
annotations for exactly this reason and report both (their Table 7). Reporting both
is the honest version and costs nothing.

results.json has a fixed schema so the three devices' files can be merged directly.
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from core import checkpoint as CK
from core import config as C
from core import data as D
from core import engine as E
from core import metrics as M
from core import splits as S
from model import build_model


def test_loader(cfg, sp_train, records, seed):
    d = cfg["data"]
    amean, astd = S.age_stats(sp_train)
    ds = D.CAUEEGCrops(
        records, cfg["paths"]["cache_dir"], crop_seconds=d["crop_seconds"],
        t0_seconds=d["t0_seconds"], n_views=d["eval_views"], mode="eval",
        normalize=d["normalize"], age_mean=amean, age_std=astd, seed=seed)
    nw = int(cfg.get("runtime", {}).get("num_workers", 4))
    bs = int(cfg["train"].get("eval_batch_size", cfg["train"]["batch_size"]))
    return D.make_loader(ds, bs, shuffle=False, num_workers=nw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    # Attempt 5: final.pt (EMA weights after the full schedule) is the primary model.
    # best.pt (raw weights at the best val-F1 epoch) is kept to compare the two rules.
    ap.add_argument("--checkpoint", default="final.pt",
                    choices=["final.pt", "best.pt", "last.pt"])
    ap.add_argument("--save-aux", action="store_true",
                    help="dump per-patient probabilities and attention/gate weights")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = C.load_config(root / args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(C.describe(cfg))
    print(f"device: {dev}\n")

    ds_root = cfg["paths"]["dataset_root"]
    sp = S.load_splits(ds_root, "dementia", verbose=False)
    eval_sets = {"test": sp["test"], "val": sp["val"]}
    try:
        spn = S.load_splits(ds_root, "dementia-no-overlap", verbose=False)
        eval_sets["test_no_overlap"] = spn["test"]
    except FileNotFoundError:
        print("[warn] dementia-no-overlap.json not found; skipping that split")

    seeds = args.seeds if args.seeds is not None else cfg["train"].get("seeds", [0])
    out_root = Path(cfg["paths"]["output_dir"]) / cfg["approach"]

    per_seed = []
    for seed in seeds:
        run_dir = out_root / f"seed{seed}"
        ck_path = run_dir / args.checkpoint
        if not ck_path.exists():
            print(f"[skip] seed {seed}: {ck_path} not found")
            continue

        model = build_model(cfg).to(dev)
        ck = CK.load(ck_path, model=model, map_location=dev, restore_rng=False,
                     fingerprint=cfg["_fingerprint"])
        entry = {"seed": seed, "best_epoch": ck.get("extra", {}).get("best_epoch"),
                 "checkpoint": str(ck_path), "splits": {}}
        print(f"\n{'='*70}\nSEED {seed}  (best epoch {entry['best_epoch']})\n{'='*70}")

        for name, recs in eval_sets.items():
            ld = test_loader(cfg, sp["train"], recs, seed)
            r = E.evaluate(model, ld, dev, amp=bool(cfg["train"].get("amp", True)),
                           return_aux=args.save_aux)
            if args.save_aux:
                np.savez(run_dir / f"aux_{name}.npz",
                         probs=r.pop("_probs"), y_true=r.pop("_y_true"),
                         aux=r.pop("_aux", np.array([])))
            r = {k: v for k, v in r.items() if not k.startswith("_")}
            entry["splits"][name] = r
            print(M.format_report(r, f"[{name}]"))
        per_seed.append(entry)

    if not per_seed:
        print("\nNo checkpoints evaluated. Run scripts/02_train.py first.")
        return

    agg = {}
    for name in eval_sets:
        runs = [e["splits"][name] for e in per_seed if name in e["splits"]]
        agg[name] = M.aggregate_seeds(runs, key=cfg["train"]["select_metric"])

    results = {
        "approach": cfg["approach"],
        "fingerprint": cfg["_fingerprint"],
        "config_path": cfg["_config_path"],
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "checkpoint": args.checkpoint,
        "eval_views": cfg["data"]["eval_views"],
        "use_age": cfg["model"].get("use_age"),
        "n_seeds": len(per_seed),
        "aggregate": agg,
        "per_seed": per_seed,
        "reference_phase1": {"test_accuracy": 0.5339, "test_macro_f1": 0.5226},
    }
    out = out_root / ("results.json" if args.checkpoint == "final.pt"
                      else f"results_{Path(args.checkpoint).stem}.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'='*70}\nAGGREGATE over {len(per_seed)} seeds -- {cfg['approach']}\n{'='*70}")
    for name, a in agg.items():
        print(f"  {name:18s} acc {a['accuracy_mean']*100:6.2f} +/- {a['accuracy_std']*100:4.2f} %"
              f"   macroF1 {a['macro_f1_mean']:.4f} +/- {a['macro_f1_std']:.4f}")
    print(f"\n  Phase-1 QSFE-Net reference: test acc 53.39%, macro-F1 0.5226")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
