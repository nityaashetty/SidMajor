"""Step 0: check environment, dataset, splits/leakage, event alignment and a GPU forward/backward
BEFORE spending compute. Safe to run any time; writes nothing except PROGRESS.log lines."""
import argparse
import importlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.common.config import add_common_args, load_config
from src.common.logging_utils import log


def check(name, fn):
    try:
        msg = fn()
        log(f"verify  OK   {name}{': ' + msg if msg else ''}")
        return True
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        log(f"verify  FAIL {name}: {type(e).__name__}: {e}")
        return False


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    root = Path(cfg["paths"]["dataset_root"])
    ok = True

    def packages():
        vers = {m: importlib.import_module(m).__version__ for m in
                ("numpy", "scipy", "pandas", "sklearn", "torch", "yaml", "pyedflib")}
        return ", ".join(f"{k} {v}" for k, v in vers.items())

    def dataset():
        n_edf = len(list((root / "signal" / "edf").glob("*.edf")))
        n_ev = len(list((root / "event").glob("*.json")))
        assert n_edf >= 1187 and n_ev >= 1187, f"found {n_edf} EDFs / {n_ev} event files"
        return f"{root} ({n_edf} EDF, {n_ev} event files)"

    def splits():
        from src.data.splits import load_splits
        s = load_splits(root)
        return " / ".join(f"{k} {len(v)}" for k, v in s.items()) + " - no leakage"

    def events():
        from src.data.events import map_events
        from src.data.edf_io import read_eeg
        from src.data.splits import load_splits
        s = load_splits(root)["train"][:20]
        worst = 0
        for r in s:
            ev = json.loads((root / "event" / f"{r['serial']}.json").read_text(encoding="utf-8"))
            x, fs = read_eeg(root / "signal" / "edf" / f"{r['serial']}.edf")
            mapped = [i for i, t in map_events(ev, 10 ** 9) if t == "paused"]
            if mapped:
                worst = max(worst, abs(mapped[-1] - x.shape[1]) / fs)
        assert worst <= 10, f"pause-corrected end is {worst:.1f} s off the signal end"
        return f"pause-corrected event clock matches signal end within {worst:.1f} s (20 recordings)"

    def store():
        from src.common.config import store_fingerprint
        from src.data.store import load_manifest
        m = load_manifest(cfg["paths"]["store"])
        assert m["fingerprint"] == store_fingerprint(cfg), "store built with a different data config"
        return f"{len(m['recordings'])} recordings at {m['fs']} Hz"

    def gpu_step():
        import torch
        from src.training.engine import build_model
        from src.training.losses import SANFLoss
        from src.training.pipeline import load_bundle, make_dataset
        bundle = load_bundle(cfg)
        ds = make_dataset(bundle, "train", train=True)
        batch = {k: torch.stack([ds[i][k] for i in range(4)]) for k in ds[0]}
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(bundle).to(dev)
        out = model({k: v.to(dev) for k, v in batch.items()})
        loss, _ = SANFLoss(cfg["train"]["class_weights"], cfg["train"]["focal_gamma"],
                           cfg["train"]["label_smoothing"], cfg["train"]["ordinal_weight"]).to(dev)(out, batch["label"].to(dev))
        loss.backward()
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0
        return (f"{dev} | batch x {tuple(batch['x'].shape)} | {len(bundle['bio_names'])} biomarkers | "
                f"loss {loss.item():.3f} | peak {mem:.2f} GB")

    for name, fn in [("packages", packages), ("dataset", dataset), ("splits", splits),
                     ("event alignment", events), ("signal store", store), ("model step", gpu_step)]:
        ok &= check(name, fn)
    log("verify: ALL OK" if ok else "verify: problems found - see FAIL lines above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
