"""Site calibration of SANF-Net on external MCI cohorts, using only each cohort's own Normals.

The CAUEEG normative model turns biomarkers into "how abnormal for this age" z-scores. At a new site
the Normals' z-scores are not centred on 0 (different amplifier, montage handling, population), so
everyone looks shifted. Calibration re-centres each cohort on its own Normals:
    shift        z' = z - median_site_normals(z)
    shift+scale  z' = (z - median) / (1.4826 * MAD of site normals), MAD floored at 0.5
Leave-one-out: a Normal subject's own recording is never part of its own calibration, so Normals
are not trivially pulled to 0. MCI recordings use all of the cohort's Normals.
Optional threshold calibration: P(impaired) cut-off at the 80th percentile of the LOO Normals.

Reads Approach 11 (models, normative model, external stores) without writing to it.

    python calibrate_external.py                         # sanf_net 3-model ensemble, both datasets
"""
import importlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
A11 = HERE.parent / "Approach11_StateAwareNormative"
sys.path[:0] = [str(A11), str(A11 / "scripts")]

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402
import torch                                                                # noqa: E402
from sklearn.metrics import roc_auc_score                                   # noqa: E402

from src.common.config import load_config                                   # noqa: E402
from src.data.dataset import RecordingBags                                  # noqa: E402
from src.training import checkpoint as ckpt                                 # noqa: E402
from src.training.engine import _loader, build_model, predict              # noqa: E402
from src.training.pipeline import load_bundle                               # noqa: E402

ext = importlib.import_module("07_external_eval")                          # record lists only
LOG = HERE / "PROGRESS.log"
AGE = {"isfahan": 68.0, "geneeg": 73.4}                                     # same imputed ages as Approach 11


def log(msg):
    import time
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def calibrate(Z, y, cohort, mode):
    if mode == "none":
        return Z.copy()
    out = np.empty_like(Z)
    for c in np.unique(cohort):
        idx = np.flatnonzero(cohort == c)
        norm = idx[y[idx] == 0]
        for i in idx:
            ref = norm[norm != i]                                           # leave-one-out for Normals
            med = np.median(Z[ref], 0)
            if mode == "shift":
                out[i] = Z[i] - med
            else:
                mad = np.maximum(1.4826 * np.median(np.abs(Z[ref] - med), 0), 0.5)
                out[i] = (Z[i] - med) / mad
    return out


def loo_threshold(s, y, cohort, q=0.8):
    """Per-recording cut-off = q-quantile of its cohort's Normals' P(impaired), excluding itself."""
    t = np.empty_like(s)
    for c in np.unique(cohort):
        idx = np.flatnonzero(cohort == c)
        norm = idx[y[idx] == 0]
        for i in idx:
            t[i] = np.quantile(s[norm[norm != i]], q)
    return t


def report(y, P, cohort):
    s = 1 - P[:, 0]
    pred = (P.argmax(1) != 0).astype(int)
    tcal = (s > loo_threshold(s, y, cohort)).astype(int)
    bal = lambda p: 0.5 * ((p[y == 0] == 0).mean() + (p[y == 1] == 1).mean())
    return {"n": len(y), "auc": roc_auc_score(y, s), "acc_argmax": (pred == y).mean(), "bal_argmax": bal(pred),
            "acc_cal": (tcal == y).mean(), "bal_cal": bal(tcal),
            "spec_cal": (tcal[y == 0] == 0).mean(), "sens_cal": (tcal[y == 1] == 1).mean()}


def main():
    run = A11 / "artifacts" / "runs" / "sanf_net"
    cfg = load_config(run / "config_used.yaml")
    cfg["train"]["num_workers"] = 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_bundle(cfg)
    models = []
    for m in sorted(p for p in run.glob("seed*") if (p / "final.pt").exists()):
        net = build_model(bundle).to(dev)
        net.load_state_dict(ckpt.load(m / "final.pt", dev)["model"])
        models.append(net)
    log(f"site calibration: SANF-Net 'sanf_net' {len(models)}-model ensemble (final), Approach 11 read-only")
    rows = []
    for ds_name in ("isfahan", "geneeg"):
        recs = ext.isfahan_records() if ds_name == "isfahan" else ext.geneeg_records()
        d = A11 / "artifacts" / "external" / ds_name
        store = d / "store"
        feats = pd.read_csv(d / "biomarkers.csv", dtype={"serial": str}).set_index("serial")
        X = feats.loc[[r["serial"] for r in recs], bundle["bio_names"]].to_numpy(np.float64)
        age = np.full(len(recs), AGE[ds_name])
        Z0 = bundle["normative"].transform(X, age)
        age_z = ((age - bundle["age_mu"]) / bundle["age_sd"])[:, None]
        y = np.array([r["label"] for r in recs])
        cohort = np.array([r.get("cohort", ds_name) for r in recs])
        stds = {}
        for r in recs:                                                      # label-free site amplitude scale
            sig = np.load(store / "signals" / f"{r['serial']}.npy", mmap_mode="r")
            with np.load(store / "timeline" / f"{r['serial']}.npz") as tl:
                keep = np.repeat(tl["clean"], cfg["data"]["fs"])
            stds[r["serial"]] = np.asarray(sig[:, :len(keep)], dtype=np.float32)[:, keep].std(axis=1)
        scales = {"caueeg": np.asarray(bundle["manifest"]["channel_scale_uv"]),
                  "site": np.median(np.stack(list(stds.values())), axis=0)}
        log(f"{ds_name}: {len(recs)} recordings, Normals per cohort "
            + ", ".join(f"{c} {int(((cohort == c) & (y == 0)).sum())}" for c in np.unique(cohort))
            + f" | mean |z| of site Normals before calibration {np.abs(Z0[y == 0]).mean():.2f} (CAUEEG Normals ~0.8)")
        for scale_name, scale in scales.items():
            manifest = {"fs": cfg["data"]["fs"], "channel_scale_uv": scale.tolist()}
            for mode in ("none", "shift", "shift+scale"):
                Z = calibrate(Z0, y, cohort, mode)
                dset = RecordingBags(recs, Z.astype(np.float32), age_z, str(store), manifest, cfg["segments"],
                                     cfg["train"]["augment"], train=False)
                P = np.mean([predict(m, _loader(dset, cfg, shuffle=False), dev)[0] for m in models], axis=0)
                groups = {"all": np.ones(len(y), bool)}
                if len(np.unique(cohort)) > 1:
                    groups.update({c: cohort == c for c in np.unique(cohort)})
                for g, mask in groups.items():
                    r = report(y[mask], P[mask], cohort[mask])
                    rows.append({"dataset": ds_name, "group": g, "amp_scale": scale_name, "calibration": mode, **r})
                    log(f"  {ds_name:8s} {g:6s} amp={scale_name:6s} calib={mode:11s} n={r['n']:3d} AUC {r['auc']:.3f} | "
                        f"argmax acc {r['acc_argmax'] * 100:5.1f}% bal {r['bal_argmax'] * 100:5.1f}% | LOO-threshold "
                        f"acc {r['acc_cal'] * 100:5.1f}% bal {r['bal_cal'] * 100:5.1f}% "
                        f"(spec {r['spec_cal'] * 100:.0f}%, sens {r['sens_cal'] * 100:.0f}%)")
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "site_calibration_results.csv", index=False)
    log(f"saved {HERE / 'site_calibration_results.csv'}")


if __name__ == "__main__":
    main()
