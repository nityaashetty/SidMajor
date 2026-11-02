"""Per-subject external predictions (no calibration vs full site calibration) for bootstrap CIs.

Reuses SiteCalibration/calibrate_external.py (same models, imputed ages, LOO calibration).
Two settings are saved: 'none' = CAUEEG amplitude scale, no calibration;
'site' = site amplitude scale + shift+scale normative re-centring on the cohort's own Normals (LOO).
Writes oof/external_<dataset>.csv with serial, cohort, y, setting, p0, p1, p2.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "SiteCalibration"))
import calibrate_external as ce                                            # noqa: E402

np, pd, torch = ce.np, ce.pd, ce.torch


def main():
    run = ce.A11 / "artifacts" / "runs" / "sanf_net"
    cfg = ce.load_config(run / "config_used.yaml")
    cfg["train"]["num_workers"] = 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = ce.load_bundle(cfg)
    models = []
    for m in sorted(p for p in run.glob("seed*") if (p / "final.pt").exists()):
        net = ce.build_model(bundle).to(dev)
        net.load_state_dict(ce.ckpt.load(m / "final.pt", dev)["model"])
        models.append(net)
    print(f"{len(models)} models", flush=True)
    for ds in ("isfahan", "geneeg"):
        recs = ce.ext.isfahan_records() if ds == "isfahan" else ce.ext.geneeg_records()
        d = ce.A11 / "artifacts" / "external" / ds
        store = d / "store"
        feats = pd.read_csv(d / "biomarkers.csv", dtype={"serial": str}).set_index("serial")
        X = feats.loc[[r["serial"] for r in recs], bundle["bio_names"]].to_numpy(np.float64)
        age = np.full(len(recs), ce.AGE[ds])
        Z0 = bundle["normative"].transform(X, age)
        age_z = ((age - bundle["age_mu"]) / bundle["age_sd"])[:, None]
        y = np.array([r["label"] for r in recs])
        cohort = np.array([r.get("cohort", ds) for r in recs])
        stds = []
        for r in recs:
            sig = np.load(store / "signals" / f"{r['serial']}.npy", mmap_mode="r")
            with np.load(store / "timeline" / f"{r['serial']}.npz") as tl:
                keep = np.repeat(tl["clean"], cfg["data"]["fs"])
            stds.append(np.asarray(sig[:, :len(keep)], dtype=np.float32)[:, keep].std(axis=1))
        settings = {"none": (np.asarray(bundle["manifest"]["channel_scale_uv"]), "none"),
                    "site": (np.median(np.stack(stds), axis=0), "shift+scale")}
        rows = []
        for name, (scale, mode) in settings.items():
            Z = ce.calibrate(Z0, y, cohort, mode)
            manifest = {"fs": cfg["data"]["fs"], "channel_scale_uv": scale.tolist()}
            dset = ce.RecordingBags(recs, Z.astype(np.float32), age_z, str(store), manifest, cfg["segments"],
                                    cfg["train"]["augment"], train=False)
            P = np.mean([ce.predict(m, ce._loader(dset, cfg, shuffle=False), dev)[0] for m in models], axis=0)
            rows.append(pd.DataFrame({"serial": [r["serial"] for r in recs], "cohort": cohort, "y": y,
                                      "setting": name, "p0": P[:, 0], "p1": P[:, 1], "p2": P[:, 2]}))
            print(f"{ds} {name}: AUC {ce.roc_auc_score(y, 1 - P[:, 0]):.3f}", flush=True)
        (HERE / "oof").mkdir(exist_ok=True)
        pd.concat(rows).to_csv(HERE / "oof" / f"external_{ds}.csv", index=False)


if __name__ == "__main__":
    main()
