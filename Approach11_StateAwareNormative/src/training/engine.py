"""Training loop for one seed: AdamW + warmup-cosine, AMP, EMA, fully resumable.

Writes to <run_dir>/seed<k>/:
  last.pt        everything needed to resume (model, EMA, optimiser, scheduler, scaler, history)
  best.pt        EMA weights at the best val macro-F1 epoch
  final.pt       EMA weights after the last epoch (default for evaluation: no selection on
                 the noisy 119-recording val set)
  history.json   per-epoch losses and val metrics
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common.logging_utils import log, seed_everything, write_json
from ..evaluation.metrics import compute_metrics
from ..models.net import SANFNet
from . import checkpoint as ckpt
from .ema import ModelEMA
from .losses import SANFLoss
from .pipeline import make_dataset


def build_model(bundle):
    return SANFNet(bundle["cfg"]["model"], n_bio=len(bundle["bio_names"]), n_states=bundle["n_states"])


def _loader(ds, cfg, shuffle, seed=0):
    t = cfg["train"]
    nw = int(t["num_workers"])
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=t["batch_size"], shuffle=shuffle, num_workers=nw, pin_memory=True,
                      persistent_workers=nw > 0, drop_last=shuffle, generator=g)


def _to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    probs, idx, gates = [], [], []
    for batch in loader:
        b = _to(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = model(b)
        probs.append(out["probs"].float().cpu())
        gates.append(out["gates"].float().cpu())
        idx.append(batch["index"])
    order = torch.cat(idx).argsort()
    return torch.cat(probs)[order].numpy(), torch.cat(gates)[order].numpy()


def _lr_lambda(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return f


def _param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "embedding" in n.lower() or n.endswith((".gamma.weight", ".beta.weight"))
         else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def fit_seed(bundle, seed, run_dir, device, val_loader, y_val, member=None):
    cfg = bundle["cfg"]
    t = cfg["train"]
    who = member or f"seed {seed}"
    seed_dir = Path(run_dir) / (member or f"seed{seed}")
    seed_dir.mkdir(parents=True, exist_ok=True)
    if (seed_dir / "final.pt").exists():
        log(f"{who}: already finished - skipping")
        return
    seed_everything(seed)

    train_ds = make_dataset(bundle, "train", train=True, seed=seed)
    train_loader = _loader(train_ds, cfg, shuffle=True, seed=seed)
    model = build_model(bundle).to(device)
    ema = ModelEMA(model, t["ema_decay"])
    opt = torch.optim.AdamW(_param_groups(model, t["weight_decay"]), lr=t["lr"])
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(t["epochs"] * steps_per_epoch, t["warmup_epochs"] * steps_per_epoch))
    use_amp = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = SANFLoss(t["class_weights"], t["focal_gamma"], t["label_smoothing"], t["ordinal_weight"]).to(device)

    start_epoch, best_f1, history = 0, -1.0, []
    if (seed_dir / "last.pt").exists():
        st = ckpt.load(seed_dir / "last.pt", device)
        model.load_state_dict(st["model"])
        ema.module.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best_f1, history = st["epoch"] + 1, st["best_f1"], st["history"]
        torch.set_rng_state(st["rng_cpu"].cpu())
        log(f"{who}: resumed at epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    if start_epoch == 0:
        log(f"{who}: training {n_params / 1e3:.1f}k params, {t['epochs']} epochs x {steps_per_epoch} steps, "
            f"modalities {model.modalities}")

    for epoch in range(start_epoch, t["epochs"]):
        t0 = time.time()
        model.train()
        train_ds.set_epoch(epoch)
        tot, n, correct = 0.0, 0, 0
        for batch in train_loader:
            b = _to(batch, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(b)
            loss, _ = loss_fn(out, b["label"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            bs = b["label"].shape[0]
            tot += loss.item() * bs
            n += bs
            correct += (out["probs"].argmax(1) == b["label"]).sum().item()

        probs, gates = predict(ema.module, val_loader, device, use_amp)
        vm = compute_metrics(y_val, probs)
        rec = {"epoch": epoch, "train_loss": tot / n, "train_acc": correct / n, "val_acc": vm["accuracy"],
               "val_macro_f1": vm["macro_f1"], "val_auc": vm["roc_auc_ovr_macro"],
               "gates": {k: round(float(g), 3) for k, g in zip(model.modalities, gates.mean(0))},
               "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
        history.append(rec)
        tag = ""
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            ckpt.save(seed_dir / "best.pt", {"model": ema.module.state_dict(), "epoch": epoch, "val": vm})
            tag = " *best*"
        if epoch % 5 == 0 or epoch == t["epochs"] - 1 or tag:
            log(f"{who} ep {epoch + 1:3d}/{t['epochs']} loss {rec['train_loss']:.4f} train {rec['train_acc'] * 100:.1f}% "
                f"| val(EMA) {vm['accuracy'] * 100:.1f}% F1 {vm['macro_f1']:.4f} AUC {vm['roc_auc_ovr_macro']:.3f} "
                f"| gates {rec['gates']} | {rec['sec']:.1f}s{tag}", progress=bool(tag) or epoch % 10 == 0)
        ckpt.save(seed_dir / "last.pt", {
            "model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best_f1": best_f1,
            "history": history, "rng_cpu": torch.get_rng_state()})
        write_json(seed_dir / "history.json", history)

    ckpt.save(seed_dir / "final.pt", {"model": ema.module.state_dict(), "epoch": t["epochs"] - 1})
    log(f"{who}: finished. best val F1 {best_f1:.4f}; final EMA val F1 {history[-1]['val_macro_f1']:.4f}")
