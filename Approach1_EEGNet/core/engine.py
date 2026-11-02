"""Train / evaluate loop shared by all three approaches.

Identical across Approach1 / Approach4 / Approach6 on purpose: the three run on three
different machines, and the only way their numbers are comparable is if the optimiser,
the selection rule, the evaluation protocol and the output schema are byte-identical.
Only model.py and config.yaml differ between the folders.

Model contract:  forward(x, age) -> (logits[B, C], aux)
    x   : [B, V, 19, T]  V views/segments per patient
    age : [B, 1]         z-scored with TRAIN statistics
    aux : anything (attention weights, gate weights) or None
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import checkpoint as ckpt
from . import metrics as M


def _amp_ctx(enabled, device):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.amp.autocast("cuda", enabled=False)


def train_one_epoch(model, loader, optimizer, criterion, device, *,
                    scaler=None, grad_clip=1.0, amp=False, log_every=0):
    model.train()
    tot_loss, correct, seen, nb = 0.0, 0, 0, 0
    for i, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        age = batch["age"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _amp_ctx(amp, device):
            logits, _ = model(x, age)
            loss = criterion(logits, y)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        tot_loss += loss.item()
        nb += 1
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.numel()
        if log_every and (i + 1) % log_every == 0:
            print(f"    step {i+1}/{len(loader)} loss={tot_loss/nb:.4f}")
    return {"loss": tot_loss / max(nb, 1), "acc": correct / max(seen, 1)}


@torch.no_grad()
def evaluate(model, loader, device, *, criterion=None, amp=False, return_aux=False):
    """One row per patient: views are aggregated inside the model."""
    model.eval()
    P, Y, A, tot_loss, nb = [], [], [], 0.0, 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        age = batch["age"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with _amp_ctx(amp, device):
            logits, aux = model(x, age)
            if criterion is not None:
                tot_loss += criterion(logits, y).item()
                nb += 1
        P.append(torch.softmax(logits.float(), 1).cpu().numpy())
        Y.append(y.cpu().numpy())
        if return_aux and aux is not None:
            A.append(aux.detach().float().cpu().numpy())

    probs = np.concatenate(P)
    y_true = np.concatenate(Y)
    res = M.compute(y_true, probs.argmax(1), probs)
    res["loss"] = tot_loss / nb if nb else None
    if return_aux and A:
        res["_aux"] = np.concatenate(A)
        res["_probs"] = probs
        res["_y_true"] = y_true
    return res


class Trainer:
    def __init__(self, model, cfg, device, run_dir, *, class_weights=None):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        t = cfg["train"]
        self.epochs = int(t["epochs"])
        self.amp = bool(t.get("amp", True)) and device.type == "cuda"
        self.grad_clip = float(t.get("grad_clip", 1.0))
        self.patience = int(t.get("early_stop_patience", 20))
        self.select = t.get("select_metric", "macro_f1")
        self.min_epochs = int(t.get("min_epochs", 10))

        w = torch.tensor(class_weights, dtype=torch.float32, device=device) \
            if class_weights else None
        self.criterion = nn.CrossEntropyLoss(
            weight=w, label_smoothing=float(t.get("label_smoothing", 0.0)))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(t["lr"]),
            weight_decay=float(t["weight_decay"]))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs,
            eta_min=float(t.get("lr_min", 0.0)))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.start_epoch = 1
        self.best = -float("inf")
        self.best_epoch = 0
        self.history = []
        self.bad = 0

    # ------------------------------------------------------------------ resume
    def maybe_resume(self, strict_fp=False):
        p = ckpt.find_resume(self.run_dir)
        if p is None:
            return False
        c = ckpt.load(p, model=self.model, optimizer=self.optimizer,
                      scheduler=self.scheduler, scaler=self.scaler,
                      map_location=self.device, restore_rng=True,
                      fingerprint=self.cfg.get("_fingerprint"), strict_fp=strict_fp)
        self.start_epoch = int(c["epoch"]) + 1
        self.best = float(c["best"])
        self.history = c.get("history", [])
        self.best_epoch = c.get("extra", {}).get("best_epoch", 0)
        self.bad = c.get("extra", {}).get("bad", 0)
        print(f"[resume] continuing from epoch {self.start_epoch} "
              f"(best {self.select}={self.best:.4f} @ epoch {self.best_epoch})")
        return True

    # ------------------------------------------------------------------- logging
    def _log(self, row):
        self.history.append(row)
        with open(self.run_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        csv_path = self.run_dir / "history.csv"
        new = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

    # ---------------------------------------------------------------------- fit
    def fit(self, train_loader, val_loader):
        print(f"[train] epochs {self.start_epoch}..{self.epochs}  "
              f"amp={self.amp}  select=val_{self.select}")
        for epoch in range(self.start_epoch, self.epochs + 1):
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            t0 = time.time()
            tr = train_one_epoch(
                self.model, train_loader, self.optimizer, self.criterion,
                self.device, scaler=self.scaler, grad_clip=self.grad_clip, amp=self.amp)
            va = evaluate(self.model, val_loader, self.device,
                          criterion=self.criterion, amp=self.amp)
            self.scheduler.step()

            score = va[self.select]
            row = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "train_loss": round(tr["loss"], 5),
                "train_acc": round(tr["acc"], 5),
                "val_loss": round(va["loss"], 5) if va["loss"] is not None else None,
                "val_acc": round(va["accuracy"], 5),
                "val_bal_acc": round(va["balanced_accuracy"], 5),
                "val_macro_f1": round(va["macro_f1"], 5),
                "secs": round(time.time() - t0, 1),
            }

            improved = score > self.best
            if improved:
                self.best, self.best_epoch, self.bad = score, epoch, 0
                ckpt.save(self.run_dir / ckpt.BEST, model=self.model,
                          optimizer=self.optimizer, scheduler=self.scheduler,
                          scaler=self.scaler, epoch=epoch, best=self.best,
                          history=self.history, fingerprint=self.cfg["_fingerprint"],
                          extra={"best_epoch": epoch, "val": _clean(va)})
            else:
                self.bad += 1

            self._log(row)
            print(f"  ep {epoch:3d}/{self.epochs}  "
                  f"train {tr['loss']:.4f}/{tr['acc']*100:5.2f}%  |  "
                  f"val {va['accuracy']*100:5.2f}% f1={va['macro_f1']:.4f}"
                  f"{'  *best*' if improved else f'  ({self.bad}/{self.patience})'}"
                  f"  {row['secs']:.1f}s")

            # last.pt is written AFTER best.pt so a crash between the two still leaves
            # last.pt pointing at a fully consistent earlier state.
            ckpt.save(self.run_dir / ckpt.LAST, model=self.model,
                      optimizer=self.optimizer, scheduler=self.scheduler,
                      scaler=self.scaler, epoch=epoch, best=self.best,
                      history=self.history, fingerprint=self.cfg["_fingerprint"],
                      extra={"best_epoch": self.best_epoch, "bad": self.bad})

            if epoch >= self.min_epochs and self.bad >= self.patience:
                print(f"[train] early stop at epoch {epoch} "
                      f"(no val_{self.select} gain for {self.bad} epochs)")
                break

        print(f"[train] best val_{self.select}={self.best:.4f} @ epoch {self.best_epoch}")
        return {"best": self.best, "best_epoch": self.best_epoch,
                "epochs_run": len(self.history)}


def _clean(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}
