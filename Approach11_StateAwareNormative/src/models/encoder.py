"""Segment encoder: multi-scale temporal convolution -> depthwise spatial -> separable conv with
squeeze-excitation (CEFA-Net's three-scale + SE idea, folded into one compact EEGNet-style network
instead of three separate CNNs, to suit ~950 training patients)."""
import torch
import torch.nn as nn


class SqueezeExcite(nn.Module):
    """Channel re-weighting; works on [N, C, ...] of any rank."""

    def __init__(self, ch, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, max(ch // reduction, 4)), nn.ReLU(inplace=True),
                                nn.Linear(max(ch // reduction, 4), ch), nn.Sigmoid())

    def forward(self, x):
        s = self.fc(x.flatten(2).mean(-1))
        return x * s.view(*s.shape, *([1] * (x.dim() - 2)))


class MultiScaleSegmentEncoder(nn.Module):
    """[N, C=19, T] -> [N, emb_dim]."""

    def __init__(self, n_channels=19, kernels=(11, 25, 51), filters_per_kernel=8, depth_multiplier=2,
                 sep_filters=64, out_filters=96, emb_dim=128, dropout=0.3):
        super().__init__()
        f1 = filters_per_kernel * len(kernels)
        f2 = f1 * depth_multiplier
        self.scales = nn.ModuleList(
            nn.Conv2d(1, filters_per_kernel, (1, k), padding=(0, k // 2), bias=False) for k in kernels)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Sequential(
            nn.Conv2d(f1, f2, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separable = nn.Sequential(
            nn.Conv2d(f2, f2, (1, 15), padding=(0, 7), groups=f2, bias=False),
            nn.Conv2d(f2, sep_filters, 1, bias=False),
            nn.BatchNorm2d(sep_filters), nn.ELU())
        self.se1 = SqueezeExcite(sep_filters)
        self.pool2 = nn.Sequential(nn.AvgPool2d((1, 5)), nn.Dropout(dropout))
        self.temporal = nn.Sequential(
            nn.Conv1d(sep_filters, out_filters, 7, padding=3, bias=False),
            nn.BatchNorm1d(out_filters), nn.ELU())
        self.se2 = SqueezeExcite(out_filters)
        self.proj = nn.Sequential(nn.Linear(2 * out_filters, emb_dim), nn.ELU(), nn.Dropout(dropout))

    def forward(self, x):
        x = x.unsqueeze(1)                                   # [N, 1, C, T]
        x = self.bn1(torch.cat([conv(x) for conv in self.scales], dim=1))
        x = self.spatial(x)                                  # [N, f2, 1, T/4]
        x = self.pool2(self.se1(self.separable(x)))          # [N, sep, 1, T/20]
        x = self.se2(self.temporal(x.squeeze(2)))            # [N, out, T/20]
        x = torch.cat([x.mean(-1), x.std(-1)], dim=1)        # mean + std pooling over time
        return self.proj(x)
