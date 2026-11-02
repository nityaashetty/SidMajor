"""Step 1 -- build the fp16 signal cache + train-set normalisation statistics.

One-time cost, measured on this machine: ~230 ms per EDF, so ~4.5 min for the 1187
recordings of the dementia task, producing ~7.3 GB (8.4 GB if you cache all 1379).
After this, a random 10 s crop costs 0.091 ms instead of 230 ms.

Safe to interrupt and re-run: the build is resumable and the data file is truncated
back to the last committed patient on restart.

If two approach folders live on the SAME machine they share this cache by default
(paths.cache_dir -> ../_cache/eeg_fp16), so build it once per device, not once per
approach.
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from core import cache as K
from core import config as C
from core import splits as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--all", action="store_true",
                    help="cache all 1379 recordings, not just the task's 1187")
    ap.add_argument("--stats-patients", type=int, default=400,
                    help="train patients used for normalisation statistics")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = C.load_config(root / args.config)
    print(C.describe(cfg), "\n")

    ds_root = Path(cfg["paths"]["dataset_root"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    sp = S.load_splits(ds_root, cfg["data"]["task"], verbose=True)

    if args.all:
        serials = sorted(p.stem for p in (ds_root / "signal" / "edf").glob("*.edf"))
    else:
        serials = sorted({r["serial"] for v in sp.values() for r in v})
    print(f"\n[cache] target: {len(serials)} recordings -> {cache_dir}")

    K.build_cache(ds_root, cache_dir, serials, verbose=True)

    # Normalisation statistics come from TRAIN serials only -- never val/test.
    train_serials = [r["serial"] for r in sp["train"]]
    print(f"\n[norm] computing per-channel mean/std from "
          f"{min(args.stats_patients, len(train_serials))} TRAIN patients "
          f"(val/test are never touched)")
    st = K.compute_norm_stats(cache_dir, train_serials,
                              max_patients=args.stats_patients,
                              seed=int(cfg["train"].get("seed", 0)))
    print(f"[norm] over {st['n_patients']} patients / {st['n_samples']:,} samples")
    print(f"[norm] mean range [{st['mean'].min():.3f}, {st['mean'].max():.3f}]  "
          f"std range [{st['std'].min():.3f}, {st['std'].max():.3f}]")
    print(f"[norm] saved -> {cache_dir / K.NORM_FILE}")
    print("\nNext: python scripts/02_train.py")


if __name__ == "__main__":
    main()
