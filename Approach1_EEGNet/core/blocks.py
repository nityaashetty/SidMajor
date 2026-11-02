"""Shared building blocks.

EEGNetEncoder is used IDENTICALLY by Approach 1 and Approach 4 so that the only
difference between those two results is the view-aggregation strategy (logit mean vs
learned attention). If the encoders differed, the comparison would be uninterpretable.

AgeFusionHead implements the CAUEEG paper's `age-fc` option (Sec 3.1, "Age signal"):
age is concatenated to the feature vector just before the classifier. Their ablation
(Table 9) puts age at +5.3 points; logistic regression on these features measures +8.5.
Phase 1 carried age in every batch and never used it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Conv2dSame(nn.Conv2d):
    """Conv2d with 'same' padding that is safe for even kernels."""
    def __init__(self, *a, **kw):
        kw.setdefault("bias", False)
        super().__init__(*a, padding="same", **kw)


class EEGNetEncoder(nn.Module):
    """EEGNet-v4 (Lawhern et al. 2018), adapted to 200 Hz.

    Defaults follow the ORIGINAL paper (F1=8, D=2, F2=16, kernel = fs/2), not the
    F1=16/D=2/F2=32 variant in the Phase-2 notes -- the original is ~4x smaller and
    was designed for exactly this regime (few hundred subjects).
    """

    def __init__(self, n_channels=19, n_samples=2000, *, F1=8, D=2, F2=16,
                 kern_len=101, pool1=4, pool2=8, dropout=0.25, out_dim=64,
                 norm="batch"):
        super().__init__()
        self.n_channels, self.n_samples = n_channels, n_samples
        # Kernel lengths are forced ODD: torch's padding='same' with an EVEN kernel
        # silently materialises a zero-padded copy of the input on every forward
        # (~116 MB/step at 24 segments). 101 ~= fs/2 at 200 Hz, as EEGNet specifies.
        kern_len = int(kern_len) | 1

        def nrm(c):
            # GroupNorm is available because Approach 4 is forced to small batches by
            # VRAM (72 segments x batch 8 = 4.46 GB of 6.44 GB measured), and BatchNorm
            # is unreliable at batch<=8.
            return nn.BatchNorm2d(c) if norm == "batch" else nn.GroupNorm(min(8, c), c)

        self.block1 = nn.Sequential(
            Conv2dSame(1, F1, (1, kern_len)),
            nrm(F1),
        )
        self.block2 = nn.Sequential(                       # depthwise spatial filter
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nrm(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(                       # separable temporal filter
            nn.Conv2d(F1 * D, F1 * D, (1, 15), groups=F1 * D, padding="same", bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nrm(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            flat = self._features(torch.zeros(1, 1, n_channels, n_samples)).shape[1]
        self.flat_dim = int(flat)
        self.proj = nn.Linear(self.flat_dim, out_dim)
        self.out_dim = out_dim

    def _features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x.flatten(1)

    def forward(self, x):                                  # [N, 19, T] -> [N, out_dim]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.proj(self._features(x))


class AgeFusionHead(nn.Module):
    """[N, feat] (+ age) -> logits. `use_age=False` gives the exact ablation twin."""

    def __init__(self, feat_dim, num_classes=3, hidden=64, dropout=0.5, use_age=True):
        super().__init__()
        self.use_age = use_age
        d = feat_dim + (1 if use_age else 0)
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, feat, age=None):
        if self.use_age:
            if age is None:
                raise ValueError("use_age=True but no age tensor was passed")
            feat = torch.cat([feat, age], dim=1)
        return self.net(feat)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
