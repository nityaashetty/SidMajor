"""SANF-Net: State-Aware, age-Normative Fusion network.

  segments [B,K,19,T] --MultiScaleSegmentEncoder--> [B,K,E] --StateFiLM(state)--> --pool--> EEG  [B,E]
  biomarkers (age-normative z) --MLP-----------------------------------------------------> BIO  [B,64]
  age (standardised) --MLP----------------------------------------------------------------> AGE  [B,16]
                         ModalityGatedFusion(EEG, BIO, AGE) -> h [B,D]
                         softmax head  +  ordinal head  -> averaged class probabilities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import MultiScaleSegmentEncoder
from .fusion import GatedAttentionPool, MeanPool, ModalityGatedFusion, StateFiLM
from .heads import OrdinalHead


class SANFNet(nn.Module):
    def __init__(self, cfg, n_bio, n_states, n_channels=19, n_classes=3):
        super().__init__()
        m = cfg
        self.use = {"eeg": m["use_eeg"], "bio": m["use_biomarkers"], "age": m["use_age"]}
        if not any(self.use.values()):
            raise ValueError("at least one of use_eeg / use_biomarkers / use_age must be true")
        dims = {}
        if self.use["eeg"]:
            self.encoder = MultiScaleSegmentEncoder(n_channels, tuple(m["kernels"]), m["filters_per_kernel"],
                                                    m["depth_multiplier"], m["sep_filters"], m["out_filters"],
                                                    m["emb_dim"], m["dropout"])
            self.film = StateFiLM(n_states, m["emb_dim"]) if m["use_state_film"] else None
            self.pool = GatedAttentionPool(m["emb_dim"]) if m["pool"] == "attention" else MeanPool()
            dims["eeg"] = m["emb_dim"]
        if self.use["bio"]:
            self.bio_mlp = nn.Sequential(nn.Dropout(0.1), nn.Linear(n_bio, 128), nn.ELU(), nn.Dropout(m["dropout"]),
                                         nn.Linear(128, 64), nn.ELU())
            dims["bio"] = 64
        if self.use["age"]:
            self.age_mlp = nn.Sequential(nn.Linear(1, 16), nn.ELU(), nn.Linear(16, 16), nn.ELU())
            dims["age"] = 16
        self.fusion = ModalityGatedFusion(dims, m["fusion_dim"], m["dropout"], m["modality_dropout"])
        self.head = nn.Sequential(nn.Dropout(m["dropout"]), nn.Linear(m["fusion_dim"], n_classes))
        self.ordinal = OrdinalHead(m["fusion_dim"], n_classes) if m["ordinal"] else None
        self.modalities = list(dims)

    def forward(self, batch):
        inputs, attn = {}, None
        if self.use["eeg"]:
            x = batch["x"]
            B, K, C, T = x.shape
            h = self.encoder(x.reshape(B * K, C, T)).view(B, K, -1)
            if self.film is not None:
                h = self.film(h, batch["state"])
            inputs["eeg"], attn = self.pool(h, batch["mask"])
        if self.use["bio"]:
            inputs["bio"] = self.bio_mlp(batch["bio"])
        if self.use["age"]:
            inputs["age"] = self.age_mlp(batch["age"])
        z, gates = self.fusion(inputs)
        logits = self.head(z)
        out = {"logits": logits, "gates": gates, "attn": attn}
        p = F.softmax(logits.float(), dim=1)
        if self.ordinal is not None:
            out["cum_logits"] = self.ordinal(z)
            p = 0.5 * (p + OrdinalHead.to_probs(out["cum_logits"]))
        out["probs"] = p
        return out
