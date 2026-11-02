"""Focal cross-entropy (CEFA-Net) + cumulative-link ordinal loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SANFLoss(nn.Module):
    def __init__(self, class_weights, focal_gamma, label_smoothing, ordinal_weight):
        super().__init__()
        self.register_buffer("w", torch.tensor(class_weights, dtype=torch.float32))
        self.gamma, self.ls, self.ow = focal_gamma, label_smoothing, ordinal_weight

    def forward(self, out, y):
        logits = out["logits"].float()
        n = logits.shape[1]
        logp = F.log_softmax(logits, dim=1)
        target = F.one_hot(y, n).float() * (1 - self.ls) + self.ls / n
        ce = -(target * logp).sum(1)                                   # smoothed CE per sample
        if self.gamma > 0:
            pt = logp.gather(1, y[:, None]).squeeze(1).exp()
            ce = ce * (1 - pt).pow(self.gamma)
        w = self.w[y]
        loss = (w * ce).sum() / w.sum()
        parts = {"ce": loss.detach()}
        if "cum_logits" in out and self.ow > 0:
            cum = out["cum_logits"].float()                            # [B, n-1]
            k = torch.arange(n - 1, device=y.device)
            tgt = (y[:, None] > k[None]).float()                       # P(y > k) targets
            ordl = F.binary_cross_entropy_with_logits(cum, tgt, reduction="none").sum(1)
            ordl = (w * ordl).sum() / w.sum()
            loss = loss + self.ow * ordl
            parts["ordinal"] = ordl.detach()
        return loss, parts
