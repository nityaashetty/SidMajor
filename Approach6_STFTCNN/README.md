# Approach 6 — STFT spectrogram CNN (device C)

Feeds a time–frequency image to a 2D CNN instead of time-averaged Welch statistics, so
non-stationarity inside the window survives. This is closest to what actually worked for
the dataset authors: their best **single** dementia model was 2D-VGG-19 at 70.18%.

## Run

```bash
pip install -r requirements.txt
python scripts/00_verify.py          # check the printed peak VRAM for YOUR card
python scripts/01_build_cache.py     # skip if ../_cache/ already built on this machine
python scripts/02_train.py           # ~50 min for 5 seeds
python scripts/03_evaluate.py
```

The STFT runs **on the GPU inside `forward()`** — 13.6 ms per batch of 64, ~0.2 s per
epoch. Precomputing it to disk would cost 4.7 GB for 8 crops (≈38 GB at 32 crops) to save
a fifth of a second, so there is no precompute step.

## What this folder does differently from the Phase-2 notes

**`n_fft = 128`, not 64.** This is the load-bearing correction. At 200 Hz:

```
n_fft= 64 -> 3.125 Hz/bin : delta 1.3 bins, theta 1.3 bins, alpha 1.6 bins
n_fft=128 -> 1.562 Hz/bin : delta 2.6 bins, theta 2.6 bins, alpha 3.2 bins
n_fft=256 -> 0.781 Hz/bin : delta 5.1 bins, theta 5.1 bins, alpha 6.4 bins
```

At 3.125 Hz/bin theta (4–8 Hz) and alpha (8–13 Hz) are **not separable** — and the
theta/alpha ratio is the strongest signal in this dataset (measured class means
0.94 / 1.31 / 1.98 for Normal / MCI / Dementia). `n_fft=64` would blur away the exact
biomarker the model is meant to find. 128 matches the CAUEEG paper's own formula,
`win_len = floor(2*sqrt(2T)+0.5) = 126` for T=2000. **Do not lower it.**

**Frequencies above 45 Hz are dropped.** The recording filter is 0.5–70 Hz, the paper's
occlusion study found >60 Hz irrelevant to this task, and clipping removes 50/60 Hz mains
interference.

**`spatial_first` option.** The notes pass the 19 electrodes straight in as image colour
channels, which collapses all spatial structure in layer one. That is the default here
(cheap, works), but `model.spatial_first: true` applies a learned 1×1 spatial filter
across electrodes first, keeping the spectro-temporal axes intact. Worth one run.

## Ablations to run

| Config change | Question it answers |
|---|---|
| `model.use_age: false` | how much is EEG vs age |
| `model.n_fft: 64` | demonstrates the resolution problem — good figure for the paper |
| `model.n_fft: 256` | finer frequency, coarser time — which matters more? |
| `model.spatial_first: true` | does an explicit spatial filter beat channel-as-colour? |
| `data.eval_views: 1` | how much does multi-crop TTA buy? |
