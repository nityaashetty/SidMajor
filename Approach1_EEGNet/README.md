# Approach 1 — EEGNet backbone (device A)

Learns temporal + spatial filters directly from the raw 19-channel signal, replacing the
hand-crafted S1–S4 feature vectors entirely.

## Run

```bash
pip install -r requirements.txt
python scripts/00_verify.py          # ~30 s
python scripts/01_build_cache.py     # ~5 min, ~7.3 GB  (skip if another approach on this
                                     #                   machine already built ../_cache/)
python scripts/02_train.py           # ~30 min for 5 seeds
python scripts/03_evaluate.py        # writes outputs/eegnet/results.json
```

Measured here: **2322 samp/s, 0.52 GB VRAM at batch 64**, ~6 min per 100-epoch seed.

## What this folder does differently from the Phase-2 notes

**One encoder, not four.** The notes say four EEGNetEncoders each fed *the same raw
signal*. That is a 4-model ensemble with a learned mixing weight — the labels S1/S2/S3/S4
stop meaning anything once the inputs are identical, so the per-class gate-weight figure
and Scientific Findings 3, 4 and 6 become unreportable. It is also 4.2× slower
(559 vs 2322 samp/s) and 294,375 vs 69,659 parameters for the same information.

Set `model.n_branches: 4` to reproduce the notes' version as an ablation — worth one run
for the paper, to show the four branches buy nothing.

**Original EEGNet-v4 sizing** (F1=8, D=2, F2=16), not the F1=16/F2=32 variant in the
notes. The notes also claim "~1,500 parameters"; the real four-branch figure is 294k.

## Ablations to run

| Config change | Question it answers |
|---|---|
| `model.use_age: false` | how much is EEG vs how much is age — **the key one** |
| `model.n_branches: 4` | do the notes' four branches help? |
| `data.eval_views: 1` | how much does multi-crop TTA buy? |
| `data.normalize: percrop` | was Phase-1's per-crop z-score hurting? |

Each is one line in `config.yaml`; change `paths.output_dir` or the `approach` name so
runs don't overwrite each other.
