"""Step 4: train SANF-Net for every seed in train.seeds (resumable; finished seeds are skipped).

    python scripts/04_train.py
    python scripts/04_train.py --set train.seeds=[0] train.epochs=3          # quick check
    python scripts/04_train.py --name no_events --set segments.use_events=false   # ablation run
    python scripts/04_train.py --cv 5          # 5-fold CV on train+val -> run 'sanf_net_cv5', one model per fold
"""
import argparse

import _bootstrap  # noqa: F401
import torch
import yaml
from pathlib import Path

from src.common.config import add_common_args, load_config
from src.common.logging_utils import log
from src.training.engine import _loader, fit_seed
from src.training.pipeline import labels, load_bundle, make_dataset


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--name", default=None, help="run name (default: config `experiment`, + _cv<k> with --cv)")
    ap.add_argument("--cv", type=int, default=None, metavar="K",
                    help="K-fold CV on train+val pooled; one model per fold (seed = fold index); test untouched")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.cv:
        cfg["cv"] = {"n_folds": args.cv, "seed": 0}
    name = args.name or (f"{cfg['experiment']}_cv{args.cv}" if args.cv else cfg["experiment"])
    run_dir = Path(cfg["paths"]["runs"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config_used.yaml"
    snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if cfg_path.exists():
        prev = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        changed = [k for k in ("data", "features", "segments", "model", "train", "cv") if prev.get(k) != snapshot.get(k)]
        changed = [k for k in changed if not (k == "train" and
                   {**prev[k], "seeds": None, "num_workers": None} == {**snapshot[k], "seeds": None, "num_workers": None})]
        if changed:
            raise SystemExit(f"run '{name}' exists with a different config ({changed}); use another --name")
    cfg_path.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = f"{args.cv} folds" if args.cv else f"seeds {cfg['train']['seeds']}"
    log(f"train: run '{name}' on {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} "
        f"| {members} | overrides {args.set or 'none'}")
    if args.cv:
        for k in range(args.cv):
            bundle = load_bundle(cfg, fold=k)
            log(f"fold {k}: {len(bundle['splits']['train'])} train / {len(bundle['splits']['val'])} held-out recordings")
            val_loader = _loader(make_dataset(bundle, "val", train=False), cfg, shuffle=False)
            fit_seed(bundle, k, run_dir, device, val_loader, labels(bundle, "val"), member=f"fold{k}")
    else:
        bundle = load_bundle(cfg)
        val_loader = _loader(make_dataset(bundle, "val", train=False), cfg, shuffle=False)
        for seed in cfg["train"]["seeds"]:
            fit_seed(bundle, seed, run_dir, device, val_loader, labels(bundle, "val"))
    log(f"train: run '{name}' complete -> python scripts/05_evaluate.py --name {name}")


if __name__ == "__main__":
    main()
