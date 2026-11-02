"""All paper numbers from the per-recording prediction CSVs in oof/, with bootstrap 95% CIs.

    python metrics.py      -> results/summary.json (+ printed tables)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OOF, RES = HERE / "oof", HERE / "results"
B = 1000
SYM = {r["serial"]: r["symptom"] for r in
       json.loads((ROOT / "caueeg-dataset" / "annotation.json").read_text())["data"]}

RUNS = ["base_age_lr", "base_bio_raw_lr", "base_bio_norm_lr", "base_bioage_norm_lr", "base_bioage_norm_hgb",
        "base_bioage_norm_lr_noproto", "base_eegnet_cv5", "base_resnet1d_cv5", "sanf_net_cv5", "abl_no_events_cv5", "abl_no_film_cv5", "abl_mean_pool_cv5", "abl_no_normative_cv5",
        "abl_no_bio_cv5", "abl_no_age_cv5", "abl_eeg_only_cv5", "abl_no_ordinal_cv5",
        "A_noprotocol_cv5", "B_eyesclosed_cv5"]


def pair_auc(y, P, neg, pos, score=None):
    s = 1 - P[:, 0] if score is None else score
    m = neg | pos
    return roc_auc_score(pos[m], s[m]) if neg.any() and pos.any() else np.nan


def core(y, P):
    pr = P.argmax(1)
    rec = [(pr[y == k] == k).mean() for k in range(3)]
    return {"acc": (pr == y).mean(), "bal": float(np.mean(rec)), "f1": f1_score(y, pr, average="macro"),
            "auc": roc_auc_score(y, P, multi_class="ovr", average="macro"),
            "rec_N": rec[0], "rec_MCI": rec[1], "rec_D": rec[2],
            "auc_N_MCI": pair_auc(y, P, y == 0, y == 1),
            "auc_MCI_D": pair_auc(y, P, y == 1, y == 2, score=P[:, 2])}


def sens_at_spec(y, P, spec=0.8):
    """MCI vs Normal: sensitivity at the threshold giving `spec` specificity on Normals (score 1-P(N))."""
    s = 1 - P[:, 0]
    t = np.quantile(s[y == 0], spec)
    return (s[y == 1] > t).mean()


def subgroup(df, y, P):
    cb = df.serial.map(lambda s: "cb_normal" in SYM[s]).to_numpy() & (y == 0)
    smi = df.serial.map(lambda s: "smi" in SYM[s]).to_numpy() & (y == 0)
    pr = P.argmax(1)
    return {"rec_cb": (pr[cb] == 0).mean(), "rec_smi": (pr[smi] == 0).mean(),
            "auc_cb_MCI": pair_auc(y, P, cb, y == 1), "auc_smi_MCI": pair_auc(y, P, smi, y == 1),
            "n_cb": int(cb.sum()), "n_smi": int(smi.sum())}


def boot(fn, n, rng, keys=None):
    """Percentile 95% CI. fn(idx) returns a number, or a dict when `keys` is given (one CI per key)."""
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(idx))
        except ValueError:
            continue
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {k: ci([v[k] for v in vals]) for k in keys} if keys else ci(vals)


def load(name):
    df = pd.read_csv(OOF / f"{name}.csv", dtype={"serial": str})
    return df[df.fold >= 0].reset_index(drop=True), df[df.fold < 0].reset_index(drop=True)


def evaluate(name, rng):
    o, t = load(name)
    out = {}
    for tag, df in (("oof", o), ("test", t)):
        y, P = df.y.to_numpy(), df[["p0", "p1", "p2"]].to_numpy()
        r = {k: float(v) for k, v in core(y, P).items()}
        r["n"] = len(y)
        r.update({k: float(v) for k, v in subgroup(df, y, P).items()})
        r["sens80_N_MCI"] = float(sens_at_spec(y[y < 2], P[y < 2]))
        keys = ("acc", "bal", "auc", "auc_N_MCI", "rec_MCI")
        for k, ci in boot(lambda i: core(y[i], P[i]), len(y), rng, keys).items():
            r[k + "_ci"] = ci
        if tag == "oof":
            accs = [(g[["p0", "p1", "p2"]].to_numpy().argmax(1) == g.y).mean() for _, g in df.groupby("fold")]
            aucs = [roc_auc_score(g.y, g[["p0", "p1", "p2"]].to_numpy(), multi_class="ovr") for _, g in df.groupby("fold")]
            r["fold_acc_mean"], r["fold_acc_sd"] = float(np.mean(accs)), float(np.std(accs, ddof=1))
            r["fold_auc_mean"], r["fold_auc_sd"] = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
            r["confusion"] = pd.crosstab(y, P.argmax(1)).reindex(index=range(3), columns=range(3), fill_value=0).values.tolist()
        out[tag] = r
    return out


def paired(a, b, rng):
    """Bootstrap CI of (a - b) on OOF for acc and macro AUC; recordings matched by serial."""
    oa, _ = load(a)
    ob, _ = load(b)
    ob = ob.set_index("serial").loc[oa.serial].reset_index()
    y = oa.y.to_numpy()
    Pa, Pb = oa[["p0", "p1", "p2"]].to_numpy(), ob[["p0", "p1", "p2"]].to_numpy()
    keys = ("acc", "auc", "auc_N_MCI", "rec_MCI")
    ca, cb = core(y, Pa), core(y, Pb)

    def diff(i):
        a, b = core(y[i], Pa[i]), core(y[i], Pb[i])
        return {k: a[k] - b[k] for k in keys}
    cis = boot(diff, len(y), rng, keys)
    return {k: {"diff": float(ca[k] - cb[k]), "ci": cis[k]} for k in keys}


def external(rng, prefix="external"):
    out = {}
    for ds in ("isfahan", "geneeg"):
        if not (OOF / f"{prefix}_{ds}.csv").exists():
            continue
        df = pd.read_csv(OOF / f"{prefix}_{ds}.csv")
        groups = {"all": None} if ds == "geneeg" else {"all": None, "isf16": "isf16", "isf19": "isf19"}
        for g, c in groups.items():
            for setting in ("none", "site"):
                d = df[df.setting == setting]
                if c:
                    d = d[d.cohort == c]
                y, s = d.y.to_numpy(), 1 - d.p0.to_numpy()
                pred = (d[["p0", "p1", "p2"]].to_numpy().argmax(1) != 0).astype(int)
                bal = 0.5 * ((pred[y == 0] == 0).mean() + (pred[y == 1] == 1).mean())
                out[f"{ds}_{g}_{setting}"] = {
                    "n": len(y), "n_mci": int(y.sum()), "auc": float(roc_auc_score(y, s)),
                    "auc_ci": boot(lambda i: roc_auc_score(y[i], s[i]), len(y), rng), "bal_argmax": float(bal)}
    return out


REPEATS = {"SANF-Net": "sanf_net_cv5", "Biomarkers+age LR": "base_bioage_norm_lr",
           "LR without protocol features": "base_bioage_norm_lr_noproto", "Age (LR)": "base_age_lr"}
REP_KEYS = ("acc", "bal", "auc", "auc_N_MCI", "rec_MCI", "auc_cb_MCI", "auc_smi_MCI")


def repeated(rng):
    """Repeated 5-fold CV (fold assignment seeds 0..4): mean and SD over repeats, plus SANF-Net - LR per repeat."""
    out, per, seeds = {}, {}, {}
    for lab, base in REPEATS.items():
        names = [base] + [f"{base}_r{r}" for r in range(1, 5)]
        names = [n for n in names if (OOF / f"{n}.csv").exists()]
        vals = []
        seeds[lab] = [0 if n == base else int(n.rsplit("_r", 1)[1]) for n in names]
        for n in names:
            o, _ = load(n)
            y, P = o.y.to_numpy(), o[["p0", "p1", "p2"]].to_numpy()
            vals.append({**core(y, P), **subgroup(o, y, P)})
        per[lab] = vals
        out[lab] = {"n_repeats": len(vals),
                    **{k: {"mean": float(np.mean([v[k] for v in vals])), "sd": float(np.std([v[k] for v in vals], ddof=1))
                           if len(vals) > 1 else 0.0} for k in REP_KEYS}}
    # pair SANF-Net and LR on the fold-assignment seeds both have
    common = sorted(set(seeds.get("SANF-Net", [])) & set(seeds.get("Biomarkers+age LR", [])))
    a = [per["SANF-Net"][seeds["SANF-Net"].index(r)] for r in common]
    b = [per["Biomarkers+age LR"][seeds["Biomarkers+age LR"].index(r)] for r in common]
    if len(a) > 1:
        out["SANF-Net - LR seeds"] = common
        out["SANF-Net - LR"] = {k: {"mean": float(np.mean([x[k] - z[k] for x, z in zip(a, b)])),
                                    "sd": float(np.std([x[k] - z[k] for x, z in zip(a, b)], ddof=1))} for k in REP_KEYS}
    return out


def main():
    rng = np.random.default_rng(0)
    RES.mkdir(exist_ok=True)
    summary = {"runs": {}, "paired": {}, "external": {}}
    for name in RUNS:
        if (OOF / f"{name}.csv").exists():
            summary["runs"][name] = evaluate(name, rng)
            o, t = summary["runs"][name]["oof"], summary["runs"][name]["test"]
            print(f"{name:24s} OOF acc {o['acc']*100:5.1f} bal {o['bal']*100:5.1f} AUC {o['auc']:.3f} "
                  f"N-MCI {o['auc_N_MCI']:.3f} MCIrec {o['rec_MCI']*100:4.1f} | test acc {t['acc']*100:5.1f} AUC {t['auc']:.3f}")
    # each comparison gets its own fixed bootstrap seed, so CIs do not move when the list changes
    for b in ("base_bioage_norm_lr", "base_bioage_norm_hgb", "base_eegnet_cv5", "base_resnet1d_cv5",
              *[r for r in RUNS if r.startswith("abl_")], "A_noprotocol_cv5", "B_eyesclosed_cv5"):
        if (OOF / f"{b}.csv").exists():
            summary["paired"][f"sanf_net_cv5 - {b}"] = paired("sanf_net_cv5", b, np.random.default_rng(1))
    if (OOF / "base_bioage_norm_lr_noproto.csv").exists():
        summary["paired"]["A_noprotocol_cv5 - base_bioage_norm_lr_noproto"] = paired(
            "A_noprotocol_cv5", "base_bioage_norm_lr_noproto", np.random.default_rng(1))
    if (OOF / "sanf_net_cv5_gates.csv").exists():
        g = pd.read_csv(OOF / "sanf_net_cv5_gates.csv")
        summary["gates_by_class"] = g.groupby("y")[["eeg", "bio", "age"]].mean().round(3).to_dict(orient="index")
    summary["external"] = external(rng)
    summary["external_cv"] = external(rng, "external_cv")
    summary["repeated_cv"] = repeated(rng)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"paired": summary["paired"], "external": summary["external"],
                      "external_cv": summary["external_cv"], "repeated_cv": summary["repeated_cv"]}, indent=1))


if __name__ == "__main__":
    main()
