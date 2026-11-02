# Phase 2 — three approaches, three devices

Three self-contained folders. Each runs on its own machine with no shared state, and the
three produce results in one schema so they can be compared directly.

```
Major_Project_Final/
├── caueeg-dataset/              # shared data (unchanged)
├── ProjectAttempt1/             # Phase 1 (unchanged — reference only)
├── _cache/                      # fp16 signal cache, built per device
├── Approach1_EEGNet/            # device A
├── Approach4_TemporalMIL/       # device B
└── Approach6_STFTCNN/           # device C
```

Each approach folder is identical except for `model.py` and `config.yaml`:

```
ApproachN_xxx/
├── config.yaml          # all paths + hyperparameters. NO path is hardcoded.
├── model.py             # the only architectural difference between folders
├── requirements.txt
├── README.md            # approach-specific notes
├── core/                # BYTE-IDENTICAL in all three folders
│   ├── config.py        #   path resolution (config → $CAUEEG_ROOT → upward search)
│   ├── cache.py         #   EDF → fp16 memmap + train-set normalisation stats
│   ├── splits.py        #   split loading + hard leakage guard
│   ├── data.py          #   Dataset: fresh crops, deterministic eval views, age
│   ├── blocks.py        #   EEGNetEncoder, AgeFusionHead (shared by A1 and A4)
│   ├── metrics.py       #   patient-level metrics, fixed schema
│   ├── checkpoint.py    #   atomic, fully resumable checkpoints
│   └── engine.py        #   train/eval loop, early stopping, logging
├── scripts/
│   ├── 00_verify.py     #   check machine + data + code BEFORE spending GPU time
│   ├── 01_build_cache.py
│   ├── 02_train.py
│   └── 03_evaluate.py
└── outputs/             # created on first run
```

---

## How to run (same four commands on every device)

```bash
cd ApproachN_xxx
pip install -r requirements.txt

python scripts/00_verify.py          # ~30 s   environment, data, leakage, VRAM smoke test
python scripts/01_build_cache.py     # ~5 min  one-time, ~7.3 GB  (shared between approaches
                                     #         on the same machine)
python scripts/02_train.py           # 30-85 min for all 5 seeds
python scripts/03_evaluate.py        # ~1 min  writes outputs/<approach>/results.json
```

If the dataset is not at `../caueeg-dataset`, either set `paths.dataset_root` in
`config.yaml` or export `CAUEEG_ROOT=/path/to/caueeg-dataset`. Nothing else is
machine-specific.

**Interrupted?** Just re-run `02_train.py`. It resumes from `last.pt` with optimiser,
scheduler, AMP scaler and RNG state intact, and skips seeds that already finished.

---

## Expected cost (measured on an RTX 4050 Laptop, 6.44 GB)

| | throughput | VRAM | 100 ep × 1 seed | 5 seeds |
|---|---|---|---|---|
| **A1** EEGNet, batch 64 | 2322 samp/s | 0.5 GB | ~6 min | **~30 min** |
| **A6** STFT-CNN, batch 32 | 1768 samp/s | 0.7 GB | ~10 min | **~50 min** |
| **A4** MIL 24 seg, batch 8 | 94 samp/s | 1.5 GB | ~17 min | **~85 min** |

Plus a one-time ~5 min cache build per device. All three fit in one evening.

---

## The ten Phase-1 defects these folders fix

Each is a measured problem in `ProjectAttempt1/`, not a stylistic preference.

| # | Phase-1 defect | Where it lived | Fix |
|---|---|---|---|
| 1 | **Age loaded, never used** — LR on age alone scores 55.08% test / F1 0.523 vs QSFE-Net's 53.39% / 0.5226; age+S1 reaches 61.86% | `npz_dataset.py:36` carried it, `qsfe_net.py:79` ignored it | `AgeFusionHead` in every model; `use_age` flag for the ablation |
| 2 | **Single-crop evaluation** — 10 s judged out of 301–1757 s; 63.5% of patients have 8 crops that disagree; voting 8 crops = +5.05 | `precompute_multicrop.py:71` (1 centre crop for val/test) | `eval_views` deterministic crops, logit-averaged in the model |
| 3 | **Frozen training crops** — same 8 vectors replayed every epoch ⇒ memorisation (train 77% / val 53%) | `precompute_multicrop.py` + `npz_dataset.py` | fresh crops every epoch from the cache (0.091 ms each) |
| 4 | **No input scaling** — θ/α ratio unbounded (max 113.5), 98.8× std ratio inside S1; `BatchNorm` sits *after* the first `Linear` | `qsfe_net.py:11-19` | train-set per-channel normalisation computed in `cache.py` |
| 5 | **Selection on 119-sample accuracy** — ±4.5 pt SE; the 8-run table (47.06–58.82%) was noise | `train_npz.py:136` | select on **macro-F1**, report mean±std over 5 seeds |
| 6 | **Hardcoded `D:\Major\...`** — nothing ran after the move | every script | runtime path resolution, zero hardcoded paths |
| 7 | **Un-resumable checkpoints** — only `model.state_dict()`; `qsfe_run7_58.82.pth` is now an orphan | `train_npz.py:139` | atomic checkpoints with optimiser/scheduler/scaler/RNG/config fingerprint |
| 8 | **No leakage guard** — `abnormal_train` contains 36 of the 118 dementia test recordings (30.5%) | not checked at all | `splits.assert_no_leakage` / `assert_no_contamination`, hard failure |
| 9 | **Crop could start at sample 0** — sensor settling included | `eeg_dataset.py:_crop` | `t0_seconds: 10` (the paper's `T0=2000`) |
| 10 | **Per-crop z-score** — discards absolute amplitude | `eeg_dataset.py:_normalize` | `normalize: trainset` (paper Sec 3.2); `percrop` kept as ablation |

---

## Things the Phase-2 notes got wrong, corrected here

| Note | Problem | What these folders do |
|---|---|---|
| A1: "four EEGNetEncoders … each receives the same raw signal" | 4 identical nets on identical input is an ensemble, not four clinical streams — it destroys the gate-weight interpretability that is the project's differentiator, and runs 4.2× slower (559 vs 2322 samp/s) | `n_branches: 1` default; `4` reproduces the note as an ablation |
| A1: "EEGNet … ~1,500 parameters" | the `Linear` head alone is ~63k; ×4 branches ≈ 254k | original EEGNet-v4 sizing (F1=8, D=2, F2=16) |
| A4: "~72 non-overlapping 10-second segments" | 456 of 1379 recordings (33%) have fewer than 72 | fixed segment **count**, adaptive **spacing**; default 24 |
| A4: memory | 72 seg × batch 8 = 4.46 GB of 6.44 GB; batch 16 does not fit | default 24 segments, batch 8, GroupNorm |
| A6: `n_fft=64` | 3.125 Hz/bin — cannot separate theta from alpha, the strongest signal in the data | `n_fft: 128` (1.56 Hz/bin), matching the paper's own formula |
| "Stage 1+2+3+4 → 78–82%" | the stages sum to 86–100% and all attack the same bottleneck; 78–82% would beat the dataset authors' 253M ensemble (74.66%) | no target is baked into the code; report mean±std and compare to 74.66% / 70.18% |

---

## Reference numbers for the paper

| | CAUEEG-Dementia test accuracy |
|---|---|
| Phase-1 QSFE-Net | 53.39% (macro-F1 0.5226) |
| LR on age alone (1 feature) | 55.08% |
| LR on age + S1 | 61.86% (macro-F1 0.598) |
| Ieracitano-CNN (paper Table 4) | 54.27% |
| CEEDNet best **single** model | 70.18% (2D-VGG-19, 20.2M params) |
| CEEDNet **ensemble** | 74.66% (256.7M params) |

Note: the 77.32% / 79.16% figures in `QSFE_NET_FULL_CONTEXT.md` are from CAUEEG-**Abnormal**,
a different (binary) task. The correct comparison for this 3-class work is 74.66% / 70.18%.

---

## Reporting checklist

Run each of these; they are cheap and they are what referees will ask for.

1. `use_age: true` vs `false` — isolates how much is EEG and how much is age. Given that
   age alone ≈ Phase-1's whole model, this is the most important ablation in the project.
2. `test` **and** `test_no_overlap` — `03_evaluate.py` does both automatically.
3. A4 only: `pool: attention` vs `pool: mean` — attention over 24 segments on 950
   patients overfits easily and mean pooling is a hard baseline.
4. 5 seeds, mean±std. Never a single best number.
5. Confirm `core checksum` from `00_verify.py` matches on all three devices before
   comparing results.
