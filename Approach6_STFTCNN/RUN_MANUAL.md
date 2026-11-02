# RUN MANUAL — Approach 6 - STFT spectrogram CNN

Operating manual for **this folder only**. Device operator: follow it top to bottom.
You do not need to read the other two approach folders to run this one.

- **This approach:** a 2D CNN over GPU-computed STFT spectrograms, so non-stationarity inside the window survives instead of being averaged away by Welch
- **Output namespace:** `outputs/stft_cnn/`
- **Total time:** ~1 h (one-time cache build + 5 training seeds + evaluation)

Companion documents: `README.md` (why this approach is built the way it is),
`../PHASE2_README.md` (how all three fit together).

---

## 0. Before you start

| Requirement | Value |
|---|---|
| Python | 3.10 - 3.12 |
| GPU | any CUDA GPU with **>= 4 GB** free VRAM (works on CPU, but ~50x slower) |
| Disk | **~8 GB free** for the signal cache |
| RAM | 8 GB minimum, 16 GB comfortable |
| Dataset | the `caueeg-dataset/` folder (1379 `.edf` files + `dementia.json`) |

The dataset is **not** copied into this folder. Put it anywhere and point at it - see §2.

> **Reference hardware.** All timings in this manual were measured on an RTX 4050 Laptop
> (6.44 GB VRAM) with 16 CPU cores. Your numbers will differ; `scripts/00_verify.py`
> prints the real figures for your machine.

---

## 1. Install

```bash
cd Approach6_STFTCNN

# 1a. PyTorch FIRST, matched to your CUDA version. For CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU-only machine:  pip install torch

# 1b. everything else
pip install -r requirements.txt
```

Confirm the GPU is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If this prints `False`, you installed a CPU-only wheel. Reinstall with the right index URL.

---

## 2. Point at the dataset

Pick **one** of these. Option A needs no edits and is the default.

**A - leave it alone.** If the layout is:

```
Major_Project_Final/
├── caueeg-dataset/
└── Approach6_STFTCNN/
```

...nothing to do. The folder searches upward for `caueeg-dataset/` automatically.

**B - environment variable** (good when the dataset lives on another drive):

```bash
# Linux / macOS
export CAUEEG_ROOT=/mnt/data/caueeg-dataset
# Windows PowerShell
$env:CAUEEG_ROOT = "D:\data\caueeg-dataset"
# Windows cmd
set CAUEEG_ROOT=D:\data\caueeg-dataset
```

**C - edit `config.yaml`:**

```yaml
paths:
  dataset_root: D:/data/caueeg-dataset
```

A valid root contains `dementia.json` and `signal/edf/`.

> **Cache location.** By default the signal cache goes to `../_cache/eeg_fp16`, *outside*
> this folder, so a second approach on the **same machine** reuses it instead of building
> another 7.3 GB copy. Change `paths.cache_dir` if that drive is short on space.

---

## 3. Step 0 - verify (~30 s)

**Always run this first, on every machine, before spending GPU time.**

```bash
python scripts/00_verify.py
```

It checks, in order: Python/torch/CUDA versions and VRAM -> config and resolved paths ->
`core/` checksum -> EDF headers (sample rate, channel count, durations) -> split sizes and
**leakage** -> model build -> one real forward+backward pass with peak VRAM.

**Read three lines of the output:**

1. `core checksum : ________` - must be **identical** on all three devices. If it differs,
   someone edited `core/`; re-sync before comparing any results.
2. `leakage check: PASS` - if this throws instead, **stop** and report it. It means a
   recording appears in two splits and no result from this run would be valid.
3. `peak VRAM : X.XX GB at batch_size=32` - if that is above ~80% of your card, lower
   `train.batch_size` in `config.yaml` now, not after a crash 40 minutes in.

Scan every EDF header instead of 25 (~1 min, worth doing once per device):

```bash
python scripts/00_verify.py --scan-edf 0
```

Writes `outputs/verify_report.json`.

---

## 4. Step 1 - build the signal cache (~5 min, ~7.3 GB)

```bash
python scripts/01_build_cache.py
```

Converts the 1187 EDFs of the dementia task into one fp16 memmap, then computes
per-channel normalisation statistics **from the training split only**.

Why it matters: a full EDF read costs **230 ms**; a random 10 s crop from this cache costs
**0.091 ms**. That 2500x gap is what lets every epoch draw *fresh* crops instead of
replaying frozen ones.

| Flag | Effect |
|---|---|
| *(none)* | caches the 1187 recordings this task needs |
| `--all` | caches all 1379 (~8.4 GB) - only if you plan to use the abnormal split |
| `--stats-patients N` | train patients used for normalisation stats (default 400) |

**Safe to interrupt.** Re-run it: the build resumes, truncating back to the last fully
written patient first. **Safe to re-run when complete** - it skips everything already done.

**Skip this step** if another approach folder on this same machine already built
`../_cache/eeg_fp16`. Check with:

```bash
ls ../_cache/eeg_fp16          # want: signals.dat, manifest.json, norm_stats.npz
```

---

## 5. Step 2 - train (~50 min for 5 seeds)

```bash
python scripts/02_train.py
```

Trains seeds 0-4 sequentially, each into `outputs/stft_cnn/seed<N>/`.

Five seeds is **not optional**. Phase 1 reported single numbers off a 119-patient
validation set where the standard error is ~4.5 points, and read an 8-run spread of
47.06-58.82% as a hard ceiling. It was sampling noise. Every number here ships with a
standard deviation so that cannot happen again.

Per-epoch output:

```
  ep  12/100  train 0.9321/58.40%  |  val 61.34% f1=0.5892  *best*  3.3s
```

`*best*` means val **macro-F1** improved and `best.pt` was rewritten. Selection is on
macro-F1, never accuracy - the classes are 367/334/249.

| Flag | Effect |
|---|---|
| `--seeds 0 1 2` | run a subset (useful to split seeds across machines) |
| `--fresh` | ignore existing checkpoints and restart from scratch |
| `--strict-fingerprint` | refuse to resume if the scientific config changed |
| `--config other.yaml` | use a different config (how you run ablations) |

### Interrupting and resuming

Ctrl-C, a crash, a reboot, a flat battery - all fine. Just run the same command again:

```bash
python scripts/02_train.py
```

It picks up from `last.pt` with optimiser moments, LR-schedule position, AMP scaler and
all four RNG states restored, and **skips seeds that already finished** (marked by
`DONE.json`). To redo a finished seed, delete its `DONE.json`.

Checkpoints are written atomically (temp file -> `os.replace`), and `last.pt` is written
*after* `best.pt`, so a crash mid-save cannot leave you with a corrupt resume point.

---

## 6. Step 3 - evaluate (~1 min)

```bash
python scripts/03_evaluate.py
```

Loads each seed's `best.pt` and scores it on **three** splits:

| Split | n | What it is |
|---|---|---|
| `val` | 119 | sanity only - it selected the checkpoint, so it is optimistic |
| `test` | 118 | the headline number |
| `test_no_overlap` | 90 | test with patient-overlap recordings removed |

Both test numbers go in the paper. ~20-24% of the standard test set shares a *patient*
with training (different recording dates); the CAUEEG authors publish the no-overlap
annotations for exactly this reason and report both (their Table 7).

Evaluation is **patient-level**: one row per patient, `eval_views` deterministic
evenly-spaced crops aggregated inside the model. Deterministic means all three devices
score identical windows.

| Flag | Effect |
|---|---|
| `--seeds 0 1` | evaluate a subset |
| `--checkpoint last.pt` | score the final epoch instead of the best |
| `--save-aux` | dump per-patient probabilities |

Writes **`outputs/stft_cnn/results.json`** - the file to send back. Fixed schema, so
the three devices' files merge directly.

---

## 7. Ablations

Run these after the main result. Each one answers a question a referee will ask.

| Priority | Config change | Question it answers |
|---|---|---|
| **1** | `model.use_age: false` | How much is EEG vs how much is just age? **Mandatory.** |
| **2** | `model.n_fft: 64` | Demonstrates the frequency-resolution problem. At 64 you get 3.125 Hz/bin and theta (4-8 Hz) cannot be separated from alpha (8-13 Hz) - the strongest signal in this dataset. Expect a clear drop. Good figure for the paper. |
| 3 | `model.spatial_first: true` | Does a learned spatial filter beat treating electrodes as image colour channels? |
| 4 | `model.n_fft: 256` | Finer frequency, coarser time - which matters more? |
| 5 | `data.eval_views: 1` | How much does multi-crop TTA actually buy? |

Each is one line in `config.yaml`. **Copy the config first** so runs do not overwrite each
other:

```bash
cp config.yaml config_noage.yaml
# edit config_noage.yaml:  model.use_age: false
#                          approach: stft_cnn_noage      <-- change this too!
python scripts/02_train.py    --config config_noage.yaml
python scripts/03_evaluate.py --config config_noage.yaml
```

Changing `approach:` is what redirects output to a separate `outputs/` subfolder. Forget
it and the new run overwrites the old one.

---

## 8. What gets produced

```
outputs/
├── verify_report.json              step 0
└── stft_cnn/
    ├── results.json                <-- THE DELIVERABLE
    ├── train_summary.json          best val score per seed
    ├── seed0/
    │   ├── best.pt                 best val macro-F1 (used by 03_evaluate)
    │   ├── last.pt                 resume point
    │   ├── DONE.json               seed finished marker
    │   ├── history.csv             per-epoch metrics (plot from this)
    │   └── history.jsonl           same, append-only
    └── seed1/ ... seed4/
```

The cache (`../_cache/eeg_fp16/`, ~7.3 GB) is reusable and does **not** need sending.

---

## 9. Sanity checks - is it actually working?

Compare against these measured reference points:

| Reference | Test accuracy |
|---|---|
| Majority class (predict Normal always) | 38.98% |
| Logistic regression on **age alone**, 1 feature | 55.08% |
| **Phase-1 QSFE-Net** (830 features, 79k params) | **53.39%** (macro-F1 0.5226) |
| Logistic regression on age + S1 | 61.86% |
| Ieracitano-CNN (CEEDNet paper, Table 4) | 54.27% |
| CEEDNet best **single** model | 70.18% (20.2M params) |
| CEEDNet **ensemble** | 74.66% (256.7M params) |

> The 77.32% / 79.16% figures in `QSFE_NET_FULL_CONTEXT.md` are from CAUEEG-**Abnormal**,
> a different binary task. The correct comparison for this 3-class work is 74.66% / 70.18%.

**Warning signs:**

| Symptom | Likely cause |
|---|---|
| test accuracy ~39%, one class fills every confusion row | collapsed to majority class - LR too high, or class weights not applied |
| train accuracy far above val and still climbing | overfitting - raise `dropout`/`weight_decay`, lower `train_repeats` |
| val macro-F1 swinging +/-5 points per epoch | normal on 119 patients - judge by the mean over 5 seeds, not one epoch |
| **test much higher than val** | not necessarily a bug: val is 119 patients and test is 118; both are noisy |
| anything **above ~75%** | treat as suspect. That beats a 256M-parameter ensemble on this split. Re-check the `00_verify.py` leakage output before believing it |

---

## 10. Troubleshooting

**`FileNotFoundError: Could not locate the CAUEEG dataset root`**
-> §2. The error prints every path it tried.

**`torch.cuda.OutOfMemoryError`**
-> Lower `train.batch_size` in `config.yaml` (halve it). `batch_size: 16` is safe. Re-run
`00_verify.py` to confirm the new peak VRAM before retraining.

**Training hangs at 0% / DataLoader worker errors (Windows)**
-> Set `runtime.num_workers: 0` in `config.yaml`. Slower, always works.

**`Cache at ... is empty` / `norm_stats.npz missing` / `N serials are not in the cache`**
-> `python scripts/01_build_cache.py` (re-running is safe and resumes).

**`LeakageError: Recording-level split leakage detected`**
-> **Do not work around this.** A recording is in two splits; the dataset JSON or the
config is wrong. Report it before training anything.

**`[ckpt] WARNING: Checkpoint fingerprint ... != current config ...`**
-> You changed a scientific setting mid-run. Either revert the config, or restart that
seed with `--fresh`. Resuming across a config change produces a result that belongs to
neither config.

**Cache build died partway / disk filled**
-> Free space and re-run `01_build_cache.py`. It truncates back to the last complete
patient automatically. Needs ~7.3 GB (~8.4 GB with `--all`).

**`RuntimeError: Expected more than 1 value per channel` (BatchNorm)**
-> A batch of size 1 reached a BatchNorm layer. Training already uses `drop_last=True`; if
you hit this, your `batch_size` is 1 - raise it.

**Results differ slightly between devices with the same seed**
-> Expected. Bitwise reproducibility across different GPUs is not achievable. That is
precisely why we report mean +/- std over 5 seeds rather than single numbers.

---

## 11. Coordinating the three devices

Before anyone compares anything:

1. All three ran `00_verify.py` and printed the **same `core checksum`**.
2. All three used the same `data.task`, `crop_seconds`, `t0_seconds`, `normalize` and
   `train.seeds`. (Only `model.*`, `batch_size` and view counts legitimately differ.)
3. Each device sends back its **`outputs/stft_cnn/results.json`** plus
   `outputs/verify_report.json`.

Do not send `.pt` checkpoints or the cache - they are large and regenerable.

Final comparison table to assemble, per approach, on `test` and `test_no_overlap`:

```
approach        use_age   test acc (mean +/- std)   test macro-F1 (mean +/- std)
------------------------------------------------------------------------------
eegnet          true      __.__ +/- _.__            0.____ +/- 0.____
eegnet          false     __.__ +/- _.__            0.____ +/- 0.____
temporal_mil    true      ...
stft_cnn        true      ...
------------------------------------------------------------------------------
Phase-1 QSFE-Net          53.39                     0.5226
```

The `use_age: false` row is the most important line in the project. Logistic regression on
age alone already beats Phase-1's entire model, so without that row nobody can tell how
much of any result is EEG and how much is the patient's age.

---

## 12. Quick reference

```bash
# one-time, per machine
pip install -r requirements.txt
python scripts/00_verify.py
python scripts/01_build_cache.py

# main run
python scripts/02_train.py
python scripts/03_evaluate.py

# resume after any interruption - same command
python scripts/02_train.py

# the key ablation
cp config.yaml config_noage.yaml      # set model.use_age: false AND approach: stft_cnn_noage
python scripts/02_train.py    --config config_noage.yaml
python scripts/03_evaluate.py --config config_noage.yaml

# deliverable
cat outputs/stft_cnn/results.json
```
