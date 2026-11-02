"""Step 2 -- train. Multi-seed by default, resumable, one run directory per seed.

Why multi-seed is mandatory, not optional:
    Phase 1 reported single numbers from a 119-patient validation set, where the
    standard error is ~4.5 points. Its 8-run table spanned 47.06%-58.82% -- entirely
    inside that noise band -- and was then read as "the ceiling is 58-59%". It was not;
    it was sampling noise. Every number this script produces comes with a std over
    seeds so that reading cannot be repeated.

Resuming:
    python scripts/02_train.py                 # resumes any interrupted seed
    python scripts/02_train.py --fresh         # ignores existing checkpoints
    python scripts/02_train.py --seeds 0 1     # subset (useful across 3 devices)
"""
import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import torch

from core import checkpoint as CK
from core import config as C
from core import data as D
from core import engine as E
from core import metrics as M
from core import splits as S
from core.blocks import count_params
from model import build_model

DONE = "DONE.json"


def build_loaders(cfg, sp, seed):
    cache = cfg["paths"]["cache_dir"]
    d = cfg["data"]
    amean, astd = S.age_stats(sp["train"])

    common = dict(
        crop_seconds=d["crop_seconds"], t0_seconds=d["t0_seconds"],
        normalize=d["normalize"], age_mean=amean, age_std=astd, seed=seed,
    )
    train_ds = D.CAUEEGCrops(
        sp["train"], cache, n_views=d["train_views"], mode="train",
        train_repeats=d["train_repeats"], sampling=d.get("sampling", "auto"),
        noise_awgn=d.get("noise_awgn", 0.0), noise_mwgn=d.get("noise_mwgn", 0.0),
        **common)
    val_ds = D.CAUEEGCrops(sp["val"], cache, n_views=d["eval_views"], mode="eval",
                           **common)

    t = cfg["train"]
    nw = int(cfg.get("runtime", {}).get("num_workers", 4))
    # drop_last=True: BatchNorm crashes on a trailing batch of size 1.
    train_ld = D.make_loader(train_ds, t["batch_size"], shuffle=True,
                             num_workers=nw, drop_last=True)
    val_ld = D.make_loader(val_ds, int(t.get("eval_batch_size", t["batch_size"])),
                           shuffle=False, num_workers=nw)
    return train_ld, val_ld, (amean, astd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints")
    ap.add_argument("--strict-fingerprint", action="store_true",
                    help="refuse to resume if the scientific config changed")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = C.load_config(root / args.config)
    print(C.describe(cfg), "\n")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    sp = S.load_splits(cfg["paths"]["dataset_root"], cfg["data"]["task"], verbose=True)
    cw = S.class_weights(sp["train"])
    print(f"class weights: {[round(w,4) for w in cw]}\n")

    seeds = args.seeds if args.seeds is not None else cfg["train"].get("seeds", [0])
    out_root = Path(cfg["paths"]["output_dir"]) / cfg["approach"]
    summaries = []

    for seed in seeds:
        run_dir = out_root / f"seed{seed}"
        done_file = run_dir / DONE
        if done_file.exists() and not args.fresh:
            print(f"\n=== seed {seed}: already finished, skipping "
                  f"(delete {done_file} to redo) ===")
            summaries.append(json.loads(done_file.read_text(encoding="utf-8")))
            continue

        print(f"\n{'='*70}\nSEED {seed}  ->  {run_dir}\n{'='*70}")
        if args.fresh and run_dir.exists():
            for f in (CK.LAST, CK.BEST, "history.jsonl", "history.csv"):
                (run_dir / f).unlink(missing_ok=True)

        CK.set_seed(seed, deterministic=bool(cfg["train"].get("deterministic", False)))
        train_ld, val_ld, (amean, astd) = build_loaders(cfg, sp, seed)
        print(f"train samples/epoch {len(train_ld.dataset)} "
              f"({len(sp['train'])} patients x {cfg['data']['train_repeats']} fresh crops)")
        print(f"val patients {len(val_ld.dataset)} "
              f"x {cfg['data']['eval_views']} deterministic views")

        model = build_model(cfg)
        print(f"params: {count_params(model):,}")

        tr = E.Trainer(model, cfg, dev, run_dir, class_weights=cw)
        if not args.fresh:
            tr.maybe_resume(strict_fp=args.strict_fingerprint)

        t0 = time.time()
        res = tr.fit(train_ld, val_ld)
        res.update(seed=seed, run_dir=str(run_dir), wall_seconds=round(time.time()-t0, 1),
                   params=count_params(model), fingerprint=cfg["_fingerprint"],
                   age_mean=amean, age_std=astd)
        done_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
        summaries.append(res)

    print(f"\n{'='*70}\nTRAINING SUMMARY ({cfg['approach']})\n{'='*70}")
    for s in summaries:
        print(f"  seed {s['seed']}: best val {cfg['train']['select_metric']}="
              f"{s['best']:.4f} @ ep{s['best_epoch']}  ({s['wall_seconds']/60:.1f} min)")
    if summaries:
        import numpy as np
        b = np.array([s["best"] for s in summaries])
        print(f"  mean +/- std: {b.mean():.4f} +/- {b.std():.4f}  over {len(b)} seeds")
    (out_root / "train_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nwrote {out_root / 'train_summary.json'}")
    print("\nNext: python scripts/03_evaluate.py")


if __name__ == "__main__":
    main()
