"""Approach 1 -- EEGNet backbone on the raw signal.

DELIBERATE DEPARTURE FROM THE PHASE-2 NOTES
    The notes specify "four EEGNetEncoders replace four StreamEncoders ... each branch
    receives the same raw signal but is initialized differently." That is a 4-model
    ensemble with a learned mixing weight, not four clinical streams: once the input is
    identical, S1/S2/S3/S4 stop meaning frequency-slowing / coherence / complexity /
    asymmetry, and every interpretability result in the Phase-1 report (the per-class
    gate weights; Scientific Findings 3, 4 and 6) becomes unreportable. Measured cost of
    the four-branch version on this GPU: 559 samp/s vs 2322 samp/s -- 4.2x slower for
    the same information.

    So the default here is ONE encoder. `model.n_branches: 4` reproduces the notes'
    version exactly if you want the ablation for the paper.

Views: training uses 1 random crop; evaluation uses `data.eval_views` deterministic
crops whose LOGITS are averaged -- test-time augmentation. Phase 1 evaluated a single
centre crop, and 8-crop aggregation measured +5.05 points on the Phase-1 features.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from core.blocks import AgeFusionHead, EEGNetEncoder


class EEGNetClassifier(nn.Module):
    def __init__(self, *, n_channels=19, n_samples=2000, num_classes=3,
                 F1=8, D=2, F2=16, kern_len=101, dropout=0.25, embed_dim=64,
                 head_hidden=64, head_dropout=0.5, use_age=True, n_branches=1,
                 norm="batch"):
        super().__init__()
        self.n_branches = int(n_branches)
        self.encoders = nn.ModuleList([
            EEGNetEncoder(n_channels, n_samples, F1=F1, D=D, F2=F2,
                          kern_len=kern_len, dropout=dropout, out_dim=embed_dim,
                          norm=norm)
            for _ in range(self.n_branches)
        ])
        feat = embed_dim * self.n_branches

        # Gated fusion is kept ONLY in the multi-branch ablation, where it mirrors
        # Phase-1's GatedFusion (qsfe_net.py:25-51) so the two are comparable.
        self.gate = None
        if self.n_branches > 1:
            self.gate = nn.Sequential(
                nn.Linear(feat, 64), nn.ReLU(),
                nn.Linear(64, self.n_branches), nn.Sigmoid())

        self.head = AgeFusionHead(feat, num_classes, hidden=head_hidden,
                                  dropout=head_dropout, use_age=use_age)

    def forward(self, x, age=None):
        """x: [B, V, C, T]   age: [B, 1]  ->  logits [B, num_classes], gates or None."""
        B, V, C, T = x.shape
        flat = x.reshape(B * V, C, T)

        vecs = [enc(flat) for enc in self.encoders]           # each [B*V, embed]
        feat = torch.cat(vecs, dim=1)

        gates = None
        if self.gate is not None:
            g = self.gate(feat)
            feat = torch.cat([g[:, i:i + 1] * vecs[i] for i in range(self.n_branches)],
                             dim=1)
            gates = g.view(B, V, -1).mean(1)

        age_rep = None
        if age is not None:
            age_rep = age.repeat_interleave(V, dim=0)          # [B*V, 1]
        logits = self.head(feat, age_rep)                      # [B*V, num_classes]

        # Average logits across views == test-time augmentation. With V=1 (training)
        # this is a no-op, so train and eval share one code path.
        logits = logits.view(B, V, -1).mean(1)
        return logits, gates


def build_model(cfg):
    m = cfg["model"]
    d = cfg["data"]
    return EEGNetClassifier(
        n_channels=19,
        n_samples=int(d["crop_seconds"] * 200),
        num_classes=int(m.get("num_classes", 3)),
        F1=int(m.get("F1", 8)), D=int(m.get("D", 2)), F2=int(m.get("F2", 16)),
        kern_len=int(m.get("kern_len", 101)),
        dropout=float(m.get("dropout", 0.25)),
        embed_dim=int(m.get("embed_dim", 64)),
        head_hidden=int(m.get("head_hidden", 64)),
        head_dropout=float(m.get("head_dropout", 0.5)),
        use_age=bool(m.get("use_age", True)),
        n_branches=int(m.get("n_branches", 1)),
        norm=m.get("norm", "batch"),
    )
