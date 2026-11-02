"""Rebuild per-recording out-of-fold predictions of SANF-Net 5-fold CV runs (checkpoints only read).

    python export_oof.py <run_name> [<run_name> ...]       # runs under Approach11 artifacts/runs/

Writes oof/<run_name>.csv (serial, age, y, p0, p1, p2, fold; fold=-1 rows = official test,
averaged over the 5 fold models) and oof/<run_name>_gates.csv (mean modality gates per OOF recording).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A11 = HERE.parents[1] / "Approach11_StateAwareNormative"
sys.path.insert(0, str(A11))

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402
import torch                                                                # noqa: E402

from src.common.config import load_config                                   # noqa: E402
from src.training import checkpoint as ckpt                                 # noqa: E402
from src.training.engine import _loader, build_model, predict              # noqa: E402
from src.training.pipeline import labels, load_bundle, make_dataset        # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = HERE / "oof"


def export(run_dir):
    cfg = load_config(run_dir / "config_used.yaml")
    cfg["train"]["num_workers"] = 0
    rows, gates, test_p = [], [], []
    for k in range(cfg["cv"]["n_folds"]):
        b = load_bundle(cfg, fold=k)
        model = build_model(b).to(DEV)
        model.load_state_dict(ckpt.load(run_dir / f"fold{k}" / "final.pt", DEV)["model"])
        for sp in ("val", "test"):
            p, g = predict(model, _loader(make_dataset(b, sp, train=False), cfg, shuffle=False), DEV)
            m = b["mats"][sp]
            if sp == "val":
                rows += [{"serial": s, "age": a, "y": int(y), "p0": q[0], "p1": q[1], "p2": q[2], "fold": k}
                         for s, a, y, q in zip(m["serials"], m["age"], labels(b, sp), p)]
                if g is not None:
                    gates.append(pd.DataFrame(np.asarray(g), columns=model.modalities).assign(
                        serial=m["serials"], y=labels(b, sp)))
            else:
                test_p.append(p)
                test = pd.DataFrame({"serial": m["serials"], "age": m["age"], "y": labels(b, sp)})
    P = np.mean(test_p, 0)
    test["p0"], test["p1"], test["p2"], test["fold"] = P[:, 0], P[:, 1], P[:, 2], -1
    OUT.mkdir(exist_ok=True)
    pd.concat([pd.DataFrame(rows), test]).to_csv(OUT / f"{run_dir.name}.csv", index=False)
    if gates:
        pd.concat(gates).to_csv(OUT / f"{run_dir.name}_gates.csv", index=False)
    print(f"exported {run_dir.name}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        export(A11 / "artifacts" / "runs" / name)
