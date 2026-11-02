"""Age-matched and protocol-stratified subgroup analysis (separates the age and protocol confounds).

    python agematched.py     -> results/agematched.json

On the 5-fold out-of-fold predictions (n=1069), each comparison group is matched 1:1 to MCI by age
(greedy nearest neighbour without replacement, caliper 2 years), and AUCs of P(impaired) = 1 - P(Normal)
are computed on the matched set, with a bootstrap CI over matched pairs. 'Age (LR)' on the matched
set should be near 0.5 and checks the matching. Community normals are further split by whether their
recording contains photic stimulation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OOF, RES = HERE / "oof", HERE / "results"
MODELS = {"SANF-Net": "sanf_net_cv5", "Biomarkers+age LR": "base_bioage_norm_lr",
          "LR without protocol features": "base_bioage_norm_lr_noproto",
          "SANF-Net (A) no photic": "A_noprotocol_cv5", "Age (LR)": "base_age_lr"}
CALIPER, B = 2.0, 1000


def match(ages_a, ages_b, rng):
    """Greedy 1:1 nearest-age matching of group a to group b (indices), random order of a."""
    free = list(range(len(ages_b)))
    pairs = []
    for i in rng.permutation(len(ages_a)):
        if not free:
            break
        d = np.abs(ages_b[free] - ages_a[i])
        j = int(np.argmin(d))
        if d[j] <= CALIPER:
            pairs.append((i, free.pop(j)))
    return pairs


def main():
    rng = np.random.default_rng(0)
    sym = {r["serial"]: r["symptom"] for r in
           json.loads((ROOT / "caueeg-dataset" / "annotation.json").read_text())["data"]}
    bio = pd.read_csv(ROOT / "Approach11_StateAwareNormative" / "artifacts" / "features" / "biomarkers.csv",
                      dtype={"serial": str}).set_index("serial")
    preds = {}
    for lab, run in MODELS.items():
        d = pd.read_csv(OOF / f"{run}.csv", dtype={"serial": str})
        preds[lab] = d[d.fold >= 0].set_index("serial")
    base = preds["SANF-Net"]
    grp = pd.Series({s: ("cb" if "cb_normal" in sym[s] else "smi" if "smi" in sym[s] else "other") if y == 0
                     else ("mci" if y == 1 else "dem") for s, y in base.y.items()})
    photic = bio.loc[base.index, "has_photic"] > 0
    groups = {"community": grp == "cb", "community, with photic": (grp == "cb") & photic,
              "community, no photic": (grp == "cb") & ~photic, "SMI": grp == "smi"}
    mci = base.index[grp == "mci"]
    out = {}
    for gname, gmask in groups.items():
        g = base.index[gmask]
        pairs = match(base.loc[g, "age"].to_numpy(), base.loc[mci, "age"].to_numpy(), rng)
        gi = g[[a for a, _ in pairs]]
        mi = mci[[b for _, b in pairs]]
        res = {"n_group": int(len(g)), "n_pairs": len(pairs),
               "age_group": float(base.loc[gi, "age"].mean()), "age_mci": float(base.loc[mi, "age"].mean()),
               "photic_share_group": float(photic[gi].mean()), "photic_share_mci": float(photic[mi].mean())}
        for lab, d in preds.items():
            s = np.r_[1 - d.loc[gi, "p0"].to_numpy(), 1 - d.loc[mi, "p0"].to_numpy()]
            y = np.r_[np.zeros(len(gi)), np.ones(len(mi))]
            n = len(pairs)
            boots = []
            for _ in range(B):
                k = rng.integers(0, n, n)
                idx = np.r_[k, k + n]
                boots.append(roc_auc_score(y[idx], s[idx]))
            res[lab] = {"auc": float(roc_auc_score(y, s)),
                        "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        out[gname] = res
        print(f"{gname:24s} pairs {len(pairs):3d} (of {len(g)}) age {res['age_group']:.1f}/{res['age_mci']:.1f} "
              f"photic {res['photic_share_group']:.0%}/{res['photic_share_mci']:.0%} | "
              + " | ".join(f"{lab} {res[lab]['auc']:.3f}" for lab in preds))
    RES.mkdir(exist_ok=True)
    (RES / "agematched.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
