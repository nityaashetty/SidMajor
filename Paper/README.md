# Paper: SANF-Net on CAUEEG

`main.tex` (IEEEtran conference format) + `references.bib` + `figures/`. To compile, either upload
`main.tex`, `references.bib` and the `figures/` folder to Overleaf (pdfLaTeX), or run
`tectonic main.tex` locally. Search for `TODO` before submitting: author block, CEFA-Net and other related
work, GENEEG reference details, repository URL.

## Where every number comes from

All analysis code is in `analysis/` and reads Approach 11 without writing to it. The one exception is
the resume fix in `Approach11_StateAwareNormative/src/training/engine.py`.

| Step | Script | Output |
|---|---|---|
| SANF-Net 5-fold CV (already trained) | `Approach11/scripts/04_train.py --cv 5` | `artifacts/runs/sanf_net_cv5` |
| Ablations (8 × 5-fold CV), then OOF export, then metrics | `analysis/run_ablations.sh` | `artifacts/runs/abl_*`, `analysis/oof/abl_*.csv` |
| OOF + test predictions from fold checkpoints | `analysis/export_oof.py <run>` | `analysis/oof/<run>.csv`, `<run>_gates.csv` |
| Tabular baselines on the same folds | `analysis/baselines_cv.py` | `analysis/oof/base_*.csv` |
| Protocol-blind runs A/B | `Ablation_PhoticConfound/` (copied) | `analysis/oof/{A,B}_*.csv` |
| External predictions (no / site calibration) | `analysis/external_preds.py` | `analysis/oof/external_*.csv` |
| Event alignment + subgroup protocol stats | `analysis/dataset_stats.py` | `analysis/results/{dataset_stats.json, alignment.csv}` |
| All metrics, bootstrap CIs, paired tests | `analysis/metrics.py` | `analysis/results/summary.json` |
| Figures | `analysis/figures.py` (+ `figures/architecture.tex`, TikZ) | `figures/*.pdf` |

The prediction CSVs have one row per recording: `serial, age, y, p0, p1, p2, fold`.
Rows with `fold = -1` are the official test set, scored by the mean of the 5 fold models.
