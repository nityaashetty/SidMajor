"""Approach 6 -- STFT spectrogram CNN.

This is closest to what actually worked for the dataset authors: their best SINGLE
dementia model was 2D-VGG-19 at 70.18% (CEEDNet paper, Table 4).

DEPARTURE FROM THE PHASE-2 NOTES -- THE STFT PARAMETERS ARE WRONG THERE
    The notes specify n_fft=64. At fs=200 Hz that is 3.125 Hz per bin:

        n_fft= 64 -> 3.125 Hz/bin : delta 1.3 bins, theta 1.3 bins, alpha 1.6 bins
        n_fft=128 -> 1.562 Hz/bin : delta 2.6 bins, theta 2.6 bins, alpha 3.2 bins
        n_fft=256 -> 0.781 Hz/bin : delta 5.1 bins, theta 5.1 bins, alpha 6.4 bins

    At 3.125 Hz/bin theta (4-8 Hz) and alpha (8-13 Hz) are not separable -- and the
    theta/alpha ratio is the single strongest signal in this dataset (measured class
    means 0.94 / 1.31 / 1.98 for Normal / MCI / Dementia). n_fft=64 would blur away the
    biomarker the model is supposed to find. Default here is 128, matching the CAUEEG
    paper's own formula win_len = floor(2*sqrt(2T)+0.5) = 126 for T=2000.

    The notes also treat the 19 electrodes as image colour channels in the first conv,
    which collapses all spatial structure in layer one. Kept as the default because it
    is cheap and works, but `model.spatial_first: true` applies a learned spatial filter
    before the 2D stack instead, which is the more faithful EEG design.

The STFT is computed ON THE GPU inside forward(). Measured: 13.6 ms per batch of 64,
i.e. ~0.2 s per 950-sample epoch. Precomputing it to disk would cost 4.7 GB for 8 crops
(and ~38 GB at 32 crops) to save a fifth of a second -- so we do not precompute.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from core.blocks import AgeFusionHead


class STFTFront(nn.Module):
    """[N, C, T] -> [N, C, F, frames] log-power spectrogram, computed on GPU."""

    def __init__(self, n_fft=128, hop=8, fmax_hz=45.0, sfreq=200.0, log=True):
        super().__init__()
        self.n_fft, self.hop, self.log = int(n_fft), int(hop), bool(log)
        self.register_buffer("window", torch.hann_window(int(n_fft)), persistent=False)
        n_bins = int(n_fft) // 2 + 1
        # Clip above ~45 Hz: the CAUEEG analogue filter is 0.5-70 Hz, the occlusion
        # study found >60 Hz irrelevant to the dementia task, and dropping it removes
        # mains interference around 50/60 Hz.
        self.keep = max(1, min(n_bins, int(round(fmax_hz / (sfreq / n_fft))) + 1))
        self.n_freq = self.keep

    def forward(self, x):                        # x: [N, C, T]
        N, C, T = x.shape
        z = torch.stft(x.reshape(N * C, T), n_fft=self.n_fft, hop_length=self.hop,
                       window=self.window, center=True, return_complex=True)
        p = z.abs().pow(2)[:, : self.keep, :]
        if self.log:
            p = torch.log1p(p)
        return p.view(N, C, self.keep, p.shape[-1])


class STFTCNN(nn.Module):
    def __init__(self, *, n_channels=19, n_samples=2000, num_classes=3,
                 n_fft=128, hop=8, fmax_hz=45.0, widths=(32, 64, 128),
                 spatial_first=False, spatial_filters=32, dropout=0.25,
                 embed_dim=64, head_hidden=64, head_dropout=0.5, use_age=True,
                 norm="batch"):
        super().__init__()
        self.front = STFTFront(n_fft, hop, fmax_hz)
        self.spatial_first = bool(spatial_first)

        in_ch = n_channels
        self.spatial = None
        if self.spatial_first:
            # Learned spatial filter across electrodes, applied per time-frequency bin,
            # before any 2D mixing. Keeps the spectro-temporal axes intact.
            self.spatial = nn.Sequential(
                nn.Conv2d(n_channels, spatial_filters, kernel_size=1, bias=False),
                nn.BatchNorm2d(spatial_filters) if norm == "batch"
                else nn.GroupNorm(min(8, spatial_filters), spatial_filters),
                nn.ELU(),
            )
            in_ch = spatial_filters

        def nrm(c):
            return nn.BatchNorm2d(c) if norm == "batch" else nn.GroupNorm(min(8, c), c)

        layers, c_prev = [], in_ch
        for w in widths:
            layers += [
                nn.Conv2d(c_prev, w, 3, padding=1, bias=False), nrm(w), nn.ELU(),
                nn.Conv2d(w, w, 3, padding=1, bias=False), nrm(w), nn.ELU(),
                nn.MaxPool2d(2), nn.Dropout2d(dropout),
            ]
            c_prev = w
        layers += [nn.AdaptiveAvgPool2d((4, 4))]
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(c_prev * 16, embed_dim)
        self.head = AgeFusionHead(embed_dim, num_classes, hidden=head_hidden,
                                  dropout=head_dropout, use_age=use_age)

    def forward(self, x, age=None):
        """x: [B, V, C, T] -> logits [B, num_classes], None."""
        B, V, C, T = x.shape
        spec = self.front(x.reshape(B * V, C, T))        # [B*V, C, F, frames]
        if self.spatial is not None:
            spec = self.spatial(spec)
        feat = self.proj(self.cnn(spec).flatten(1))      # [B*V, embed]

        age_rep = age.repeat_interleave(V, dim=0) if age is not None else None
        logits = self.head(feat, age_rep).view(B, V, -1).mean(1)   # view-averaged = TTA
        return logits, None


def build_model(cfg):
    m = cfg["model"]
    d = cfg["data"]
    return STFTCNN(
        n_channels=19,
        n_samples=int(d["crop_seconds"] * 200),
        num_classes=int(m.get("num_classes", 3)),
        n_fft=int(m.get("n_fft", 128)),
        hop=int(m.get("hop", 8)),
        fmax_hz=float(m.get("fmax_hz", 45.0)),
        widths=tuple(m.get("widths", [32, 64, 128])),
        spatial_first=bool(m.get("spatial_first", False)),
        spatial_filters=int(m.get("spatial_filters", 32)),
        dropout=float(m.get("dropout", 0.25)),
        embed_dim=int(m.get("embed_dim", 64)),
        head_hidden=int(m.get("head_hidden", 64)),
        head_dropout=float(m.get("head_dropout", 0.5)),
        use_age=bool(m.get("use_age", True)),
        norm=m.get("norm", "batch"),
    )
