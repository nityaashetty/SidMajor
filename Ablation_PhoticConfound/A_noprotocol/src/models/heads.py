"""Classification heads. Normal < MCI < Dementia is an ordered scale, and MCI sits between the
other two; a cumulative-link (ordinal) head encodes that, the softmax head does not."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalHead(nn.Module):
    """One severity score s and K-1 ordered thresholds: P(y > k) = sigmoid(s - t_k)."""

    def __init__(self, dim, n_classes=3):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.t0 = nn.Parameter(torch.tensor(-1.0))
        self.deltas = nn.Parameter(torch.full((n_classes - 2,), 0.5))

    def thresholds(self):
        return torch.cat([self.t0.view(1), self.t0 + torch.cumsum(F.softplus(self.deltas) + 1e-3, 0)])

    def forward(self, h):
        return self.score(h) - self.thresholds()  # [B, K-1] cumulative logits

    @staticmethod
    def to_probs(cum_logits):
        p_gt = torch.sigmoid(cum_logits.float())                 # P(y > k), decreasing in k
        ones = torch.ones_like(p_gt[:, :1])
        zeros = torch.zeros_like(p_gt[:, :1])
        upper = torch.cat([ones, p_gt], 1)
        lower = torch.cat([p_gt, zeros], 1)
        p = (upper - lower).clamp_min(1e-6)
        return p / p.sum(1, keepdim=True)
