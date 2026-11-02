# Approach 11: SANF-Net (State-Aware, age-Normative Fusion network)

This is a clean restart. Everything here is built from the **original `caueeg-dataset`
only**: the raw EDFs, event JSONs and split JSONs. It doesn't use `_cache/`, earlier precomputed
or cropped files, or code from earlier approaches. Every generated file goes to `artifacts/`
inside this folder.

## Why this design

| Bottleneck seen in Phase 1 to Attempt 5 | What Approach 11 does |
|---|---|
| Random 10 s crops mix eyes-open, eyes-closed, photic stimulation, hyperventilation and artifacts. Clinical EEG slowing is read **eyes-closed**. | Uses the **event annotations**, which no earlier CAUEEG work (including CEEDNet) used. Every second is labelled EC / EO / photic / HV / unknown and clean or not. Segments are drawn by state (60% EC by default), and each segment carries a state token. |
| Event timestamps don't line up with the EDF. | Verified on all 1187 recordings: event indices count **paused** time, but the EDF has it cut out. `src/data/events.py` subtracts the cumulative pause time; afterwards the clocks agree to within 0–6 s. Guard bands absorb the rest, and the pause joins are masked. |
| Age alone gives 55% and is tangled with EEG slowing (EEG slows with normal ageing). | **Age-normative biomarkers**: each biomarker becomes a z-score against **train-set Normals of the same age**, i.e. "how abnormal is this EEG for someone this age". Age enters the network as its own input, so the two contributions can be separated. |
| MCI is the class everyone confuses, and it lies *between* Normal and Dementia. | An **ordinal head** (one severity score plus two ordered thresholds) is trained alongside a focal-CE head. |
| 950 patients, so deep nets overfit (train 77% vs val 53% in Phase 1). | A 143k-parameter encoder, **same-class bag mixing** (part of a patient's segments swapped with another patient who has the same label), modality dropout, weight decay 5e-2, EMA, and a fixed cosine schedule. |
| Selecting a checkpoint on a 119-recording val set is noisy (±4.4 pt). | Evaluation uses the **final EMA** weights by default. Best-val weights are kept as well. Results are reported as mean ± std over 5 seeds. |

### Borrowed from CEFA-Net (Diagnostics 2026)
* **Clinical biomarker branch**: their 8 ratios (TAR, DAR, slow/fast, fronto-posterior alpha, alpha and beta asymmetry, frontal theta, posterior alpha). This was their biggest ablation gain (+5.6 pt).
* **Multi-scale temporal convolution + squeeze-excitation**. CEFA uses three separate CNNs; here it's one compact network with three kernel scales (0.11 / 0.25 / 0.51 s), which suits ~950 patients better.
* **Focal loss with a heavier MCI weight**, amplitude/noise augmentation, and a seed ensemble.
* *Not* borrowed: per-segment evaluation and the standard split with patient overlap. Those produced CEFA's 81%; on the patient-disjoint test their number is 73.8%. All numbers here are **per recording**, on both `test` and `test_no_overlap`.

### Novel parts
1. Behavioural-state-aware segment sampling with state-conditioned (FiLM) embeddings, built on the pause-corrected event clock.
2. **Reactivity biomarkers**, which only exist with events: posterior **alpha reactivity** (EC → EO suppression, reduced in AD) and **photic driving** SNR at the flash frequency.
3. Normative modelling of all biomarkers against age, using train-split Normals only.
4. A modality-gated fusion of EEG, biomarkers and age. The gates show, per recording, how much each source was used (they're reported per class).
5. An ordinal + focal two-head output.

## Architecture

```
24 x 10 s segments [19 ch x 1000 @100 Hz], each tagged EC/EO/photic
  └ MultiScaleSegmentEncoder: temporal conv k=11|25|51 → depthwise spatial(19) → separable conv + SE
     → temporal conv + SE → mean+std pool → 128-d   (shared across segments)
  └ StateFiLM(state) → gated attention pooling over segments → EEG 128-d
48 biomarkers (age-normative z) → MLP → 64-d
age (standardised) → MLP → 16-d
  └ ModalityGatedFusion (softmax gates, modality dropout) → 128-d
     ├ softmax head (focal CE, MCI weight 1.4, label smoothing)
     └ ordinal head (score − thresholds, cumulative BCE)       → averaged probabilities
```

## Layout

```
Approach11_StateAwareNormative/
├── config.yaml              every setting; override with --set key=value
├── PROGRESS.log             live log (the progress window tails this)
├── src/
│   ├── common/      config.py (paths, fingerprints), logging_utils.py
│   ├── data/        splits.py (leakage guard), edf_io.py, events.py (state timeline),
│   │                store.py (EDF → filtered 100 Hz store), dataset.py (state-aware bags)
│   ├── features/    spectral.py, biomarkers.py (48 recording-level markers), normative.py
│   ├── models/      encoder.py, fusion.py (FiLM, attention pool, gated fusion), heads.py, net.py
│   ├── training/    pipeline.py, engine.py (resumable loop), losses.py, ema.py, checkpoint.py
│   └── evaluation/  metrics.py
├── scripts/  00_verify · 01_build_store · 02_build_features · 03_tabular_baseline · 04_train · 05_evaluate
├── slurm/train.sbatch       the whole pipeline on the cluster
└── artifacts/               generated: store/ (3.3 GB), features/, runs/<name>/seed*/
```

## Run

```bash
cd Approach11_StateAwareNormative
pip install -r requirements.txt
python scripts/01_build_store.py        # ~1-5 min, 3.3 GB, resumable
python scripts/02_build_features.py     # ~30 s
python scripts/03_tabular_baseline.py   # ~1 min, CPU: age / biomarker baselines
python scripts/00_verify.py             # environment, leakage, event alignment, GPU step
python scripts/04_train.py              # 5 seeds x 100 epochs, ~11 min/seed on an RTX 4050
python scripts/05_evaluate.py           # writes artifacts/runs/sanf_net/results.json

# 5-fold CV ensemble (train+val pooled, one model per fold, test untouched)
python scripts/04_train.py --cv 5       # run 'sanf_net_cv5', ~5 min per fold
python scripts/05_evaluate.py --name sanf_net_cv5
```

With `--cv`, each fold refits the normative biomarker model and the age scaling on its own
training part. `05_evaluate.py` reports each fold's held-out score, the pooled out-of-fold score
over all 1069 train+val recordings, and the test / no-overlap scores of each fold model and of
their 5-model average. Tabular fusion is skipped in CV mode, since there is no common val set.

Re-running any step resumes or skips finished work. Changing a `data` setting invalidates the store,
and the scripts refuse to mix stale artifacts.

## Ablations (one flag each, all evaluated the same way)

```bash
python scripts/04_train.py --name no_events     --set segments.use_events=false     # CEEDNet-style random crops
python scripts/04_train.py --name no_normative  --set features.normative=false      # plain standardisation
python scripts/04_train.py --name no_bio        --set model.use_biomarkers=false
python scripts/04_train.py --name no_age        --set model.use_age=false
python scripts/04_train.py --name eeg_only      --set model.use_biomarkers=false model.use_age=false
python scripts/04_train.py --name no_ordinal    --set model.ordinal=false
python scripts/04_train.py --name no_film       --set model.use_state_film=false
python scripts/04_train.py --name mean_pool     --set model.pool=mean
python scripts/05_evaluate.py --name <name>
```

## Results

See `RESULTS.md` (written once the full run has been evaluated).
