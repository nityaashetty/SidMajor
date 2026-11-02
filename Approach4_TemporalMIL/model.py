"""Approach 4 -- temporal segment aggregation (multiple-instance learning).

This is the approach the Phase-1 measurements most directly support:
    * 63.5% of patients have 8 crops that disagree on the prediction
    * within-patient variance is 52.5% of total feature variance
    * majority-voting 8 crops was worth +5.05 points with no retraining
A patient-level decision made from one 10 s window is mostly reading window noise.
Attention pooling over many segments is the principled fix.

TWO DEPARTURES FROM THE PHASE-2 NOTES

 1. SEGMENT COUNT. The notes say "divide the full 12-minute recording into ~72
    non-overlapping 10-second segments". Measured over all 1379 EDFs: durations run
    301-1757 s (median 854 s) and 456 recordings -- 33% -- hold FEWER than 72 segments.
    A fixed 72 would force padding or repetition for a third of the cohort and bias the
    attention weights toward whatever filler was used. Here the segment COUNT is fixed
    and the SPACING adapts (core/data.py), which is well-defined at every length.

 2. BATCH SIZE / NORM. Measured on this GPU: 72 segments at batch 8 peaks at 4.46 GB of
    6.44 GB usable, and batch 16 does not fit. BatchNorm is unreliable at batch<=8, so
    `model.norm: group` is the default here (it is `batch` in Approach 1, which runs at
    batch 64). 24 segments at batch 8 runs 3.1x faster for near-identical coverage.

The attention weights are returned as `aux`: they say WHICH parts of a recording drove
the decision, which restores the interpretability story that a raw-signal model would
otherwise lose. Save them with `03_evaluate.py --save-aux`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.blocks import AgeFusionHead, EEGNetEncoder


class AttentionPool(nn.Module):
    """Gated attention pooling (Ilse et al. 2018), the standard MIL pooler."""

    def __init__(self, dim, hidden=64, gated=True):
        super().__init__()
        self.gated = gated
        self.V = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.U = nn.Linear(dim, hidden) if gated else None

    def forward(self, h, mask=None):            # h: [B, V, dim]
        a = torch.tanh(self.V(h))
        if self.gated:
            a = a * torch.sigmoid(self.U(h))
        scores = self.w(a).squeeze(-1)           # [B, V]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=1)
        return torch.einsum("bv,bvd->bd", w, h), w


class TemporalMIL(nn.Module):
    def __init__(self, *, n_channels=19, n_samples=2000, num_classes=3,
                 F1=8, D=2, F2=16, kern_len=101, dropout=0.25, embed_dim=64,
                 pool="attention", attn_hidden=64, head_hidden=64,
                 head_dropout=0.5, use_age=True, norm="group"):
        super().__init__()
        if pool not in ("attention", "mean", "max"):
            raise ValueError(f"pool must be attention|mean|max, got {pool!r}")
        self.pool_kind = pool
        # ONE encoder shared across segments -- identical class to Approach 1, so the
        # two experiments differ only in how views are aggregated.
        self.encoder = EEGNetEncoder(n_channels, n_samples, F1=F1, D=D, F2=F2,
                                     kern_len=kern_len, dropout=dropout,
                                     out_dim=embed_dim, norm=norm)
        self.pool = AttentionPool(embed_dim, attn_hidden) if pool == "attention" else None
        self.head = AgeFusionHead(embed_dim, num_classes, hidden=head_hidden,
                                  dropout=head_dropout, use_age=use_age)

    def forward(self, x, age=None):
        """x: [B, V, C, T] -> logits [B, num_classes], attention [B, V] or None."""
        B, V, C, T = x.shape
        h = self.encoder(x.reshape(B * V, C, T)).view(B, V, -1)   # [B, V, embed]

        if self.pool_kind == "attention":
            pooled, attn = self.pool(h)
        elif self.pool_kind == "mean":
            # Always run this baseline. A Linear(embed,1) attention over V segments on
            # 950 patients overfits easily, and mean pooling is a hard baseline to beat.
            pooled, attn = h.mean(1), None
        else:
            pooled, attn = h.max(1).values, None

        return self.head(pooled, age), attn


def build_model(cfg):
    m = cfg["model"]
    d = cfg["data"]
    return TemporalMIL(
        n_channels=19,
        n_samples=int(d["crop_seconds"] * 200),
        num_classes=int(m.get("num_classes", 3)),
        F1=int(m.get("F1", 8)), D=int(m.get("D", 2)), F2=int(m.get("F2", 16)),
        kern_len=int(m.get("kern_len", 101)),
        dropout=float(m.get("dropout", 0.25)),
        embed_dim=int(m.get("embed_dim", 64)),
        pool=m.get("pool", "attention"),
        attn_hidden=int(m.get("attn_hidden", 64)),
        head_hidden=int(m.get("head_hidden", 64)),
        head_dropout=float(m.get("head_dropout", 0.5)),
        use_age=bool(m.get("use_age", True)),
        norm=m.get("norm", "group"),
    )
