"""Step 7: zero-shot external test - apply a trained SANF-Net run to another dataset, no retraining.

Supported: isfahan (Kashefpoor 2016 + 2019 cohorts, Normal vs MCI, 19-ch EDF @256 Hz, eyes-closed)
           geneeg  (Mishra 2025, 35 control / 28 MCI, 17-ch text @250 Hz, 4-min eyes-closed P300 protocol;
                    T5/T6 are rebuilt by spherical-spline interpolation, clean seconds come from the .art track)
Each recording goes through the CAUEEG pipeline: common-average re-reference, 0.5-45 Hz band-pass,
resample to 100 Hz, whole recording treated as clean eyes-closed EEG (no event labels exist), the
same 48 biomarkers, z-scored with the CAUEEG-train normative model. No ages exist, so every patient
gets the same imputed age: age carries no information and the result is EEG-only.

Two channel-scaling variants are reported:
  caueeg  - CAUEEG train channel scale (pure zero-shot)
  site    - the external site's own median channel std (label-free harmonisation of amplitude)

    python scripts/07_external_eval.py --dataset isfahan
    python scripts/07_external_eval.py --dataset geneeg --age 73.4
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import pyedflib
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from src.common.config import APPROACH_ROOT, load_config
from src.common.logging_utils import log, write_json
from src.data.dataset import RecordingBags
from src.data.edf_io import EEG_CHANNELS
from src.data.events import STATE_EC
from src.data.store import preprocess
from src.features.biomarkers import recording_features
from src.training import checkpoint as ckpt
from src.training.engine import _loader, build_model, predict
from src.training.pipeline import eval_datasets, labels, load_bundle

EXTERNAL = APPROACH_ROOT.parent / "external_datasets"
LABELS = {"NORMAL": 0, "MCI": 1}


def isfahan_records():
    recs = []
    for cohort, tag in (("2016_Kashefpoor_27subjects", "isf16"), ("2019_Kashefpoor_34patients", "isf19")):
        base = EXTERNAL / "Isfahan_MCI" / cohort
        for r in pd.read_csv(base / "manifest.csv").itertuples():
            recs.append({"serial": f"{tag}_{r.file[:-4]}", "path": base / "processed" / "edf_fixed" / r.file,
                         "label": LABELS[r.label], "cohort": tag})
    return recs


GENEEG_CH = ["FP1", "FP2", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2", "T3", "T4", "FZ", "CZ", "PZ"]


def geneeg_records():
    recs = []
    for sub, lab, tag in (("Control_raw_files", 0, "ctrl"), ("MCI_raw_files", 1, "mci")):
        for p in sorted((EXTERNAL / "GENEEG" / "raw" / sub).glob("*.eeg")):
            recs.append({"serial": f"gen_{p.stem}", "path": p, "label": lab, "cohort": "geneeg", "reader": "geneeg"})
    return recs


def read_geneeg(path, fs=250.0):
    """17-ch text -> 19 channels in EEG_CHANNELS order (T5/T6 interpolated), average reference,
    plus a per-second clean mask from the .art track (seconds it does not cover count as unclean)."""
    import mne
    x17 = np.loadtxt(path, dtype=np.float64)
    art = np.loadtxt(path.with_suffix(".art"), dtype=np.int16)
    std_names = {"FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz"}
    names = [std_names.get(c, c) for c in GENEEG_CH] + ["T5", "T6"]
    data = np.vstack([x17.T, np.zeros((2, len(x17)))]) * 1e-6          # uV -> V for MNE
    raw = mne.io.RawArray(data, mne.create_info(names, fs, "eeg"), verbose="error")
    raw.set_montage("standard_1020", verbose="error")
    raw.info["bads"] = ["T5", "T6"]
    raw.interpolate_bads(reset_bads=True, verbose="error")
    order = [{"FZ": "Fz", "CZ": "Cz", "PZ": "Pz"}.get(c, c) for c in EEG_CHANNELS]
    x = (raw.get_data(picks=order) * 1e6).astype(np.float32)
    ok = np.zeros(len(x17), bool)
    ok[: len(art)] = art.max(axis=1) == 0
    n_sec = len(x17) // int(fs)
    clean = ok[: n_sec * int(fs)].reshape(n_sec, int(fs)).all(axis=1)
    return x - x.mean(axis=0, keepdims=True), fs, clean


def read_19(path):
    f = pyedflib.EdfReader(str(path))
    try:
        names = [l.strip().upper() for l in f.getSignalLabels()]
        idx = [names.index(c.upper()) for c in EEG_CHANNELS]
        fs = float(f.getSampleFrequency(idx[0]))
        x = np.stack([f.readSignal(i) for i in idx]).astype(np.float32)
    finally:
        f.close()
    return x - x.mean(axis=0, keepdims=True), fs           # common-average reference, like CAUEEG


def build_store(recs, store, dcfg):
    (store / "signals").mkdir(parents=True, exist_ok=True)
    (store / "timeline").mkdir(parents=True, exist_ok=True)
    stds = {}
    for r in recs:
        sig = store / "signals" / f"{r['serial']}.npy"
        if not sig.exists():
            if r.get("reader") == "geneeg":
                x, fs_in, clean_src = read_geneeg(r["path"])
            else:
                (x, fs_in), clean_src = read_19(r["path"]), None
            y = preprocess(x, fs_in, dict(dcfg, fs=dcfg["fs"]))
            n_sec = y.shape[1] // dcfg["fs"]
            state = np.full(n_sec, STATE_EC, np.int8)
            clean = np.ones(n_sec, bool)
            if clean_src is not None:
                clean[: min(n_sec, len(clean_src))] &= clean_src[:n_sec]
            clean[: dcfg["skip_start_sec"]] = False
            clean[-2:] = False
            np.save(sig, y.astype(np.float16))
            np.savez(store / "timeline" / f"{r['serial']}.npz", state=state, clean=clean,
                     photic=np.zeros((0, 3), np.float32))
        y = np.load(sig, mmap_mode="r")
        with np.load(store / "timeline" / f"{r['serial']}.npz") as tl:
            keep = np.repeat(tl["clean"], dcfg["fs"])
        seg = np.asarray(y[:, : len(keep)], dtype=np.float32)[:, keep]
        stds[r["serial"]] = seg.std(axis=1)
    return stds


def binary_report(y, P):
    p_imp = 1 - P[:, 0]                       # P(MCI) + P(Dementia)
    pred = (P.argmax(1) != 0).astype(int)
    return {"n": int(len(y)), "auc_impaired": float(roc_auc_score(y, p_imp)),
            "auc_mci_vs_normal": float(roc_auc_score(y, P[:, 1] / (P[:, 0] + P[:, 1]))),
            "acc": float(accuracy_score(y, pred)), "balanced_acc": float(balanced_accuracy_score(y, pred)),
            "normals_called_normal": float((pred[y == 0] == 0).mean()), "mci_flagged": float((pred[y == 1] == 1).mean()),
            "pred_counts_N_MCI_D": np.bincount(P.argmax(1), minlength=3).tolist()}


def fmt(b):
    return (f"n={b['n']:3d} | AUC(impaired) {b['auc_impaired']:.3f} | AUC(MCI vs N) {b['auc_mci_vs_normal']:.3f} | "
            f"acc {b['acc'] * 100:5.1f}% bal {b['balanced_acc'] * 100:5.1f}% | normals kept {b['normals_called_normal'] * 100:5.1f}% "
            f"| MCI flagged {b['mci_flagged'] * 100:5.1f}% | predicted N/MCI/D {b['pred_counts_N_MCI_D']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["isfahan", "geneeg"], default="isfahan")
    ap.add_argument("--name", default="sanf_net")
    ap.add_argument("--checkpoint", choices=["final", "best"], default="final")
    ap.add_argument("--age", type=float, default=68.0, help="imputed age for every recording")
    args = ap.parse_args()

    base = load_config()
    run_dir = Path(base["paths"]["runs"]) / args.name
    cfg = load_config(run_dir / "config_used.yaml")
    cfg["train"]["num_workers"] = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_bundle(cfg)                                  # CAUEEG: normative model, age stats, channel scale
    members = sorted(p for p in run_dir.glob("seed*") if (p / f"{args.checkpoint}.pt").exists())
    models = []
    for m in members:
        net = build_model(bundle).to(device)
        net.load_state_dict(ckpt.load(m / f"{args.checkpoint}.pt", device)["model"])
        models.append(net)

    recs = isfahan_records() if args.dataset == "isfahan" else geneeg_records()
    out_dir = Path(base["paths"]["artifacts"]) / "external" / args.dataset
    store = out_dir / "store"
    log(f"external: {args.dataset} - {len(recs)} recordings, building store (CAUEEG preprocessing) ...")
    stds = build_store(recs, store, cfg["data"])

    feats = pd.DataFrame([recording_features(r["serial"], str(store), cfg["data"]["fs"], cfg["features"]) for r in recs])
    feats.to_csv(out_dir / "biomarkers.csv", index=False)
    X = feats.set_index("serial").loc[[r["serial"] for r in recs], bundle["bio_names"]].to_numpy(np.float64)
    age = np.full(len(recs), args.age)
    Z = bundle["normative"].transform(X, age)
    age_z = ((age - bundle["age_mu"]) / bundle["age_sd"])[:, None]
    y = np.array([r["label"] for r in recs])

    caueeg_scale = np.asarray(bundle["manifest"]["channel_scale_uv"])
    site_scale = np.median(np.stack(list(stds.values())), axis=0)
    log(f"external: median channel std - CAUEEG train {np.median(caueeg_scale):.1f} uV, "
        f"{args.dataset} {np.median(site_scale):.1f} uV (ratio {np.median(site_scale) / np.median(caueeg_scale):.2f})")

    results = {"dataset": args.dataset, "run": args.name, "checkpoint": args.checkpoint, "members": len(models),
               "imputed_age": args.age, "variants": {}}
    rows = []
    for variant, scale in (("caueeg", caueeg_scale), ("site", site_scale)):
        manifest = {"fs": cfg["data"]["fs"], "channel_scale_uv": scale.tolist()}
        ds = RecordingBags(recs, Z, age_z, str(store), manifest, cfg["segments"], cfg["train"]["augment"], train=False)
        loader = _loader(ds, cfg, shuffle=False)
        P = np.mean([predict(m, loader, device)[0] for m in models], axis=0)
        res = {"all": binary_report(y, P)}
        for tag in sorted({r["cohort"] for r in recs}) if len({r["cohort"] for r in recs}) > 1 else []:
            mask = np.array([r["cohort"] == tag for r in recs])
            res[tag] = binary_report(y[mask], P[mask])
        results["variants"][variant] = res
        for r, p in zip(recs, P):
            rows.append({"variant": variant, "serial": r["serial"], "true": ["Normal", "MCI"][r["label"]],
                         "pred": ["N", "MCI", "D"][p.argmax()], "P_N": p[0], "P_MCI": p[1], "P_D": p[2]})

    # same ensemble, same metrics, on CAUEEG's own test set (Normal + MCI only) for comparison
    ref_loader = _loader(eval_datasets(bundle)["test"], cfg, shuffle=False)
    Pc = np.mean([predict(m, ref_loader, device)[0] for m in models], axis=0)
    yc = labels(bundle, "test")
    keep = yc != 2
    results["caueeg_test_normal_vs_mci"] = binary_report(yc[keep], Pc[keep])
    # ... and age-blind, exactly like the external data: same imputed age, biomarkers re-z-scored with it
    test_rows = bundle["splits"]["test"]
    cau = pd.read_csv(Path(cfg["paths"]["features"]) / "biomarkers.csv", dtype={"serial": str}).set_index("serial")
    Xc = cau.loc[[r["serial"] for r in test_rows], bundle["bio_names"]].to_numpy(np.float64)
    agec = np.full(len(test_rows), args.age)
    ds_blind = RecordingBags(test_rows, bundle["normative"].transform(Xc, agec),
                             ((agec - bundle["age_mu"]) / bundle["age_sd"])[:, None], cfg["paths"]["store"],
                             bundle["manifest"], cfg["segments"], cfg["train"]["augment"], train=False)
    Pb = np.mean([predict(m, _loader(ds_blind, cfg, shuffle=False), device)[0] for m in models], axis=0)
    results["caueeg_test_normal_vs_mci_age_blind"] = binary_report(yc[keep], Pb[keep])

    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    write_json(out_dir / "results.json", results)
    clean_frac = {r["serial"]: float(np.load(store / "timeline" / f"{r['serial']}.npz")["clean"].mean()) for r in recs}
    results["clean_fraction_median"] = {lab: float(np.median([clean_frac[r["serial"]] for r in recs if r["label"] == k]))
                                        for k, lab in enumerate(["Normal", "MCI"])}
    log(f"  clean fraction (median): {results['clean_fraction_median']}")
    log(f"===== external test: {args.dataset} (Normal vs MCI), SANF-Net '{args.name}' {len(models)}-model ensemble, no retraining =====")
    log(f"  reference  CAUEEG test N+MCI   : {fmt(results['caueeg_test_normal_vs_mci'])}")
    log(f"  reference  CAUEEG, age-blind   : {fmt(results['caueeg_test_normal_vs_mci_age_blind'])}")
    for variant in ("caueeg", "site"):
        for k in results["variants"][variant]:
            log(f"  {variant:6s} scale  {k:5s}            : {fmt(results['variants'][variant][k])}")
    log(f"  saved {out_dir / 'results.json'} and predictions.csv")


if __name__ == "__main__":
    main()
