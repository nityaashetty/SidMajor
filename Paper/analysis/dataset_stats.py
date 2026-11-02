"""Dataset-level statistics for the paper: event-clock alignment and protocol composition by subgroup.

    python dataset_stats.py      -> results/dataset_stats.json, results/alignment.csv
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A11 = ROOT / "Approach11_StateAwareNormative"
sys.path.insert(0, str(A11))

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402

from src.data.events import map_events                                      # noqa: E402
from src.data.splits import load_splits                                     # noqa: E402

FS_EVENT = 200


def alignment(serial, n_samples_100):
    ev = json.loads((ROOT / "caueeg-dataset" / "event" / f"{serial}.json").read_text())
    n = n_samples_100 * 2                                                   # EDF length at 200 Hz
    raw = [(int(i), " ".join(str(t).strip().lower().split())) for i, t in ev]
    paused = [i for i, t in raw if t == "paused"]
    if not paused:
        return None
    corr = [i for i, t in map_events(ev, 10 ** 9) if t == "paused"]       # no clipping
    n_pauses = sum(t == "recording resumed" for _, t in raw)
    return {"serial": serial, "n_pauses": n_pauses,
            "offset_raw_s": (paused[-1] - n) / FS_EVENT, "offset_corr_s": (corr[-1] - n) / FS_EVENT}


def main():
    splits = load_splits(ROOT / "caueeg-dataset")
    rows = splits["train"] + splits["val"] + splits["test"]
    man = json.loads((A11 / "artifacts" / "store" / "manifest.json").read_text())["recordings"]
    sym = {r["serial"]: r["symptom"] for r in
           json.loads((ROOT / "caueeg-dataset" / "annotation.json").read_text())["data"]}
    bio = pd.read_csv(A11 / "artifacts" / "features" / "biomarkers.csv", dtype={"serial": str}).set_index("serial")

    al = pd.DataFrame([a for r in rows if (a := alignment(r["serial"], man[r["serial"]]["n_samples"]))])
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    al.to_csv(out_dir / "alignment.csv", index=False)
    with_p = al[al.n_pauses > 0]
    stats = {
        "n_recordings": len(rows),
        "n_with_final_paused_event": int(len(al)),
        "n_with_pauses": int(len(with_p)),
        "abs_offset_raw_s_all": {q: float(al.offset_raw_s.abs().quantile(v)) for q, v in
                                 (("median", .5), ("p90", .9), ("max", 1.0))},
        "abs_offset_corr_s_all": {q: float(al.offset_corr_s.abs().quantile(v)) for q, v in
                                  (("median", .5), ("p90", .9), ("max", 1.0))},
        "abs_offset_raw_s_paused_recs": {q: float(with_p.offset_raw_s.abs().quantile(v)) for q, v in
                                         (("median", .5), ("p90", .9), ("max", 1.0))},
        "abs_offset_corr_s_paused_recs": {q: float(with_p.offset_corr_s.abs().quantile(v)) for q, v in
                                          (("median", .5), ("p90", .9), ("max", 1.0))},
    }

    def group(r):
        s = sym[r["serial"]]
        if r["label"] == 0:
            return "Normal: community" if "cb_normal" in s else ("Normal: SMI" if "smi" in s else "Normal: other")
        return ["Normal", "MCI", "Dementia"][r["label"]]

    comp = []
    for r in rows:
        m = man[r["serial"]]
        cs = m["clean_sec_by_state"]                                        # unknown, EC, EO, photic, HV
        comp.append({"group": group(r), "age": r["age"], "has_photic": bio.loc[r["serial"], "has_photic"],
                     "has_eo": bio.loc[r["serial"], "has_eo"], "dur_min": m["n_sec"] / 60,
                     "clean_min": m["clean_sec"] / 60, "ec_min": cs[1] / 60, "eo_min": cs[2] / 60,
                     "photic_min": cs[3] / 60, "hv_min": cs[4] / 60, "unknown_min": cs[0] / 60})
    comp = pd.DataFrame(comp)
    g = comp.groupby("group")
    stats["by_group"] = pd.concat([g.size().rename("n"), g.age.mean().rename("age_mean"),
                                   g.age.std().rename("age_sd"), g.mean(numeric_only=True).drop(columns="age")],
                                  axis=1).round(3).reset_index().to_dict(orient="records")
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
