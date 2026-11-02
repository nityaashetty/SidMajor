# Approach 4 — temporal segment aggregation / MIL (device B)

Encodes many segments spread across the whole recording and pools them with learned
attention, instead of judging a patient from one 10-second window.

This is the approach the Phase-1 measurements support most directly:

- 63.5% of patients have 8 crops that **disagree** on the prediction
- within-patient variance is **52.5%** of total feature variance
- crude majority-voting of 8 crops was already worth **+5.05 points**

## Run

```bash
pip install -r requirements.txt
python scripts/00_verify.py
python scripts/01_build_cache.py     # skip if ../_cache/ already built on this machine
python scripts/02_train.py           # ~85 min for 5 seeds
python scripts/03_evaluate.py --save-aux    # --save-aux dumps the attention weights
```

Measured here: **94 samp/s, 1.50 GB VRAM at 24 segments × batch 8**, ~17 min per seed.

## What this folder does differently from the Phase-2 notes

**Segment count is fixed; spacing adapts.** The notes say "divide the full 12-minute
recording into ~72 non-overlapping 10-second segments". Measured across all 1379 EDFs:
durations run **301–1757 s** (median 854 s), and **456 recordings — 33% — hold fewer than
72 segments**. A hard 72 would pad or repeat for a third of the cohort and bias the
attention toward the filler. Default is 24 segments, evenly spread over whatever length
the recording has.

**VRAM.** 72 segments × batch 8 = 4.46 GB of 6.44 GB usable, and batch 16 does not fit.
24 segments runs 3.1× faster with essentially the same coverage. `norm: group` because
BatchNorm is unreliable at batch 8.

**Sampling.** Training uses *stratified* starts — one random position inside each of 24
equal bins — so every epoch sees full temporal coverage *and* fresh crops. Evaluation
uses deterministic evenly-spaced starts, so all three devices score identical windows.

## Ablations to run

| Config change | Question it answers |
|---|---|
| `model.pool: mean` | **run this always** — attention over 24 segments on 950 patients overfits easily, and mean pooling is a hard baseline |
| `model.use_age: false` | how much is EEG vs age |
| `data.train_views: 8` / `48` | how many segments actually matter |
| `model.pool: max` | is one decisive segment enough? |

## Interpretability

`03_evaluate.py --save-aux` writes `aux_test.npz` with per-patient attention over the 24
segments. That answers "**which parts of the recording drove this diagnosis**" and
replaces the Phase-1 gate-weight figure — the only one of the three approaches that keeps
an interpretability story.
