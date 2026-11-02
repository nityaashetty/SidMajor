"""Exponential moving average of weights (BatchNorm buffers are copied, not averaged)."""
import copy

import torch


class ModelEMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point and not k.endswith(("running_mean", "running_var")):
                v.mul_(self.decay).add_(src, alpha=1 - self.decay)
            else:
                v.copy_(src)
