# Attempt 5 — EEGNet with a regularised training recipe

Same model (EEGNet-v4, F1=8/D=2/F2=16, ~70k params, age-fc) and same data pipeline as
`Approach1_EEGNet`. **Only the training recipe changes**, so any difference from
Approach 1 comes from these three things:

| # | Change | Approach 1 | Attempt 5 | Why |
|---|---|---|---|---|
| 1 | Weight decay | 1e-4 (all params) | **5e-2** (conv/linear weights only) | CEEDNet's tuned 1D models use 4e-2 to 7e-2 (paper Table B.1) |
| 2 | MixUp | none | **alpha = 0.2** (crops, age and loss mixed together) | CEEDNet Sec 3.3; ablation Table 9 |
| 3 | Model selection | best val-F1 epoch, early stop at patience 25 | **full 150-epoch cosine, no early stop; EMA weights at the end (`final.pt`)** | val is 119 patients (±4.4 pt SE). A1 seeds stopped at ep 46–92 with LR still high |

`best.pt` (raw weights at the best val macro-F1 epoch, the Approach 1 rule) is still
saved, so both selection rules can be compared on the same runs.

## Run

```bash
python scripts/02_train.py                          # 5 seeds, ~10 min each on RTX 4050
python scripts/03_evaluate.py                       # final.pt (EMA) -> results.json
python scripts/03_evaluate.py --checkpoint best.pt  # best-val epoch -> results_best.json
```

The fp16 cache is shared at `../_cache/eeg_fp16`. If it is missing, run
`scripts/01_build_cache.py` first.

## Code changes vs Approach 1

- `core/engine.py`: `ModelEMA`, `param_groups` (no decay on norm/bias), MixUp in
  `train_one_epoch`, EMA val scores logged as `ema_val_acc` / `ema_val_macro_f1`,
  and `final.pt` written at the end of training.
- `scripts/03_evaluate.py`: `--checkpoint final.pt` is the default, and
  `results_<ckpt>.json` is written for the other checkpoints.
- `config.yaml`: `weight_decay`, `mixup_alpha`, `ema_decay`, `epochs`,
  `early_stop_patience` (marked `A5`).

Note: with MixUp on, `train_acc` is scored against the dominant label of each mixed
pair, so it reads lower than in Approach 1. That is expected, not underfitting.
