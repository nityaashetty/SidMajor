"""State conditioning, segment pooling and modality-gated fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateFiLM(nn.Module):
    """Feature-wise modulation of a segment embedding by its behavioural state (EC/EO/photic/...).
    Initialised to identity, so it only departs from 'state-agnostic' if that helps."""

    def __init__(self, n_states, dim):
        super().__init__()
        self.gamma = nn.Embedding(n_states, dim)
        self.beta = nn.Embedding(n_states, dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, h, state):
        return h * (1 + self.gamma(state)) + self.beta(state)


class GatedAttentionPool(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018) over the segments of one recording."""

    def __init__(self, dim, hidden=64, dropout=0.1):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, mask):
        a = self.w(self.drop(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h)))).squeeze(-1)  # [B, K]
        a = a.masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1)
        return (w.unsqueeze(-1) * h).sum(1), w


class MeanPool(nn.Module):
    def forward(self, h, mask):
        w = mask.float() / mask.float().sum(1, keepdim=True).clamp_min(1)
        return (w.unsqueeze(-1) * h).sum(1), w


class ModalityGatedFusion(nn.Module):
    """Projects each modality to a common space and mixes them with per-recording softmax gates.

    The gates say how much each recording's prediction leaned on EEG vs biomarkers vs age.
    Modality dropout (training only) randomly removes modalities so no single one - age in
    particular - can carry the model.
    """

    def __init__(self, dims, dim, dropout, modality_dropout):
        super().__init__()
        self.names = list(dims)
        self.proj = nn.ModuleDict({n: nn.Sequential(nn.Linear(d, dim), nn.LayerNorm(dim), nn.ELU(),
                                                    nn.Dropout(dropout)) for n, d in dims.items()})
        self.gate = nn.Linear(dim * len(dims), len(dims))
        self.p_drop = modality_dropout

    def forward(self, inputs):
        z = torch.stack([self.proj[n](inputs[n]) for n in self.names], dim=1)  # [B, M, D]
        logits = self.gate(z.flatten(1))
        if self.training and self.p_drop > 0 and len(self.names) > 1:
            drop = torch.rand_like(logits) < self.p_drop
            all_dropped = drop.all(dim=1, keepdim=True)
            drop = drop & ~all_dropped
            logits = logits.masked_fill(drop, float("-inf"))
        g = F.softmax(logits, dim=1)
        return (g.unsqueeze(-1) * z).sum(1), g
