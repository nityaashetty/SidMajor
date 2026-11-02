"""Step 0 -- verify the machine, the data and the code BEFORE spending GPU hours.

Run this first on every device. It performs no training and writes nothing except a
report file. It checks the things that silently broke Phase 1:
  * can the dataset be found at all (no hardcoded paths)
  * are all EDFs the shape the code assumes (200 Hz, 21 ch)
  * do the splits leak
  * is the core/ code identical to the other two devices (checksum)
  * does one forward+backward pass actually fit in this GPU's VRAM
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from core import config as C
from core import splits as S


def core_checksum(root: Path) -> str:
    h = hashlib.sha256()
    for f in sorted((root / "core").glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--scan-edf", type=int, default=25,
                    help="how many EDF headers to validate (0 = all 1379)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = C.load_config(root / args.config)
    report = {}

    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report["platform"] = platform.platform()
    report["python"] = platform.python_version()
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    print(f"  platform  : {report['platform']}")
    print(f"  python    : {report['python']}")
    print(f"  torch     : {report['torch']}  cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        report["gpu"] = p.name
        report["vram_gb"] = round(p.total_memory / 1e9, 2)
        print(f"  gpu       : {p.name}")
        print(f"  vram      : {report['vram_gb']} GB   SMs={p.multi_processor_count}")
        if report["vram_gb"] < 5.5:
            print("  !! under 5.5 GB usable -- lower train.batch_size / data.train_views")
    else:
        print("  gpu       : NONE (cpu only -- training will be very slow)")

    print("\n" + "=" * 70)
    print("CONFIG & PATHS")
    print("=" * 70)
    print(C.describe(cfg))
    report["fingerprint"] = cfg["_fingerprint"]
    report["core_checksum"] = core_checksum(root)
    print(f"core checksum : {report['core_checksum']}"
          "   <- must match on all three devices")

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)
    ds_root = Path(cfg["paths"]["dataset_root"])
    edf_dir = ds_root / "signal" / "edf"
    edfs = sorted(edf_dir.glob("*.edf"))
    print(f"  EDF files : {len(edfs)}")
    report["n_edf"] = len(edfs)

    n_scan = len(edfs) if args.scan_edf == 0 else min(args.scan_edf, len(edfs))
    if n_scan:
        import pyedflib
        bad, rates, chans, lens = [], set(), set(), []
        idx = np.linspace(0, len(edfs) - 1, n_scan).astype(int)
        for i in idx:
            f = edfs[i]
            try:
                r = pyedflib.EdfReader(str(f))
                rates.add(float(r.getSampleFrequency(0)))
                chans.add(int(r.signals_in_file))
                lens.append(int(r.getNSamples()[0]))
                r.close()
            except Exception as e:
                bad.append((f.name, str(e)[:50]))
        print(f"  scanned   : {n_scan} headers")
        print(f"  rates     : {sorted(rates)}   channels: {sorted(chans)}")
        print(f"  length s  : min={min(lens)/200:.0f} median={np.median(lens)/200:.0f} "
              f"max={max(lens)/200:.0f}")
        report.update(sample_rates=sorted(rates), channel_counts=sorted(chans),
                      unreadable=bad)
        if bad:
            print(f"  !! UNREADABLE: {bad[:3]}")
        if rates != {200.0} or chans != {21}:
            print("  !! unexpected header values -- core/cache.py assumes 200 Hz / 21 ch")

    print("\n" + "=" * 70)
    print("SPLITS & LEAKAGE")
    print("=" * 70)
    sp = S.load_splits(ds_root, cfg["data"]["task"], verbose=True)
    report["splits"] = {k: len(v) for k, v in sp.items()}

    crop_len = int(cfg["data"]["crop_seconds"] * 200)
    t0 = int(cfg["data"]["t0_seconds"] * 200)
    views = max(int(cfg["data"]["train_views"]), int(cfg["data"]["eval_views"]))
    print(f"\n  crop={crop_len} samples  t0={t0}  max_views={views}")
    print("  (segment COUNT is fixed; spacing adapts to each recording's length,")
    print("   because 456/1379 recordings hold fewer than 72 non-overlapping 10 s windows)")

    print("\n" + "=" * 70)
    print("MODEL SMOKE TEST")
    print("=" * 70)
    from model import build_model
    m = build_model(cfg).to(dev)
    from core.blocks import count_params
    report["params"] = count_params(m)
    print(f"  params    : {report['params']:,}")

    B = int(cfg["train"]["batch_size"])
    V = int(cfg["data"]["train_views"])
    x = torch.randn(B, V, 19, crop_len, device=dev)
    age = torch.randn(B, 1, device=dev)
    y = torch.randint(0, 3, (B,), device=dev)
    try:
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        logits, aux = m(x, age)
        torch.nn.functional.cross_entropy(logits, y).backward()
        opt.step()
        print(f"  fwd+bwd   : OK   logits={tuple(logits.shape)}  "
              f"aux={None if aux is None else tuple(aux.shape)}")
        if dev.type == "cuda":
            pk = torch.cuda.max_memory_allocated() / 1e9
            report["peak_vram_gb"] = round(pk, 2)
            print(f"  peak VRAM : {pk:.2f} GB at batch_size={B}, views={V}")
            if pk > report.get("vram_gb", 99) * 0.8:
                print("  !! >80% of VRAM -- reduce train.batch_size or data.train_views")
    except RuntimeError as e:
        print(f"  fwd+bwd   : FAILED -- {str(e)[:120]}")
        report["smoke_test"] = "FAILED"
    else:
        report["smoke_test"] = "OK"

    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'verify_report.json'}")
    print("\nNext: python scripts/01_build_cache.py")


if __name__ == "__main__":
    main()
