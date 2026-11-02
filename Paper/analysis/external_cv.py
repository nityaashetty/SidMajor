"""External predictions with the 5-fold CV models (the same models as every CAUEEG result in the paper).

    python external_cv.py [run_name]        # default sanf_net_cv5

Each fold model is paired with its own fold's normative model and age scaling; predictions are
averaged over the 5 folds. Settings as in external_preds.py: 'none' (CAUEEG amplitude scale, no
calibration) and 'site' (site amplitude scale + leave-one-out shift+scale normative re-centring).
Writes oof/external_cv_<dataset>.csv.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "SiteCalibration"))
import calibrate_external as ce                                            # noqa: E402

np, pd, torch = ce.np, ce.pd, ce.torch


def site_inputs(ds, cfg):
    recs = ce.ext.isfahan_records() if ds == "isfahan" else ce.ext.geneeg_records()
    d = ce.A11 / "artifacts" / "external" / ds
    feats = pd.read_csv(d / "biomarkers.csv", dtype={"serial": str}).set_index("serial")
    stds = []
    for r in recs:
        sig = np.load(d / "store" / "signals" / f"{r['serial']}.npy", mmap_mode="r")
        with np.load(d / "store" / "timeline" / f"{r['serial']}.npz") as tl:
            keep = np.repeat(tl["clean"], cfg["data"]["fs"])
        stds.append(np.asarray(sig[:, :len(keep)], dtype=np.float32)[:, keep].std(axis=1))
    return recs, d / "store", feats, np.median(np.stack(stds), axis=0)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "sanf_net_cv5"
    run = ce.A11 / "artifacts" / "runs" / name
    cfg = ce.load_config(run / "config_used.yaml")
    cfg["train"]["num_workers"] = 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in ("isfahan", "geneeg"):
        recs, store, feats, site_scale = site_inputs(ds, cfg)
        y = np.array([r["label"] for r in recs])
        cohort = np.array([r.get("cohort", ds) for r in recs])
        age = np.full(len(recs), ce.AGE[ds])
        P = {"none": [], "site": []}
        for k in range(cfg["cv"]["n_folds"]):
            b = ce.load_bundle(cfg, fold=k)
            net = ce.build_model(b).to(dev)
            net.load_state_dict(ce.ckpt.load(run / f"fold{k}" / "final.pt", dev)["model"])
            Z0 = b["normative"].transform(feats.loc[[r["serial"] for r in recs], b["bio_names"]].to_numpy(np.float64), age)
            age_z = ((age - b["age_mu"]) / b["age_sd"])[:, None]
            for setting, scale, mode in (("none", np.asarray(b["manifest"]["channel_scale_uv"]), "none"),
                                         ("site", site_scale, "shift+scale")):
                Z = ce.calibrate(Z0, y, cohort, mode)
                dset = ce.RecordingBags(recs, Z.astype(np.float32), age_z, str(store),
                                        {"fs": cfg["data"]["fs"], "channel_scale_uv": scale.tolist()},
                                        cfg["segments"], cfg["train"]["augment"], train=False)
                P[setting].append(ce.predict(net, ce._loader(dset, cfg, shuffle=False), dev)[0])
        rows = []
        for setting, ps in P.items():
            p = np.mean(ps, 0)
            rows.append(pd.DataFrame({"serial": [r["serial"] for r in recs], "cohort": cohort, "y": y,
                                      "setting": setting, "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2]}))
            print(f"{ds} {setting}: AUC {ce.roc_auc_score(y, 1 - p[:, 0]):.3f}", flush=True)
        pd.concat(rows).to_csv(HERE / "oof" / f"external_cv_{ds}.csv", index=False)


if __name__ == "__main__":
    main()
