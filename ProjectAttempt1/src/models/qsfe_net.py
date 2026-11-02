import torch
import torch.nn as nn

class StreamEncoder(nn.Module):
    """
    One FC encoder for one clinical stream.
    Takes a feature vector, outputs a 32-dim representation.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class GatedFusion(nn.Module):
    """
    Takes all four stream vectors, learns which streams
    matter most for this patient, weights them accordingly.
    """
    def __init__(self):
        super().__init__()
        # 4 streams x 32 dims = 128 input
        self.gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()  # 4 weights between 0 and 1
        )

    def forward(self, v1, v2, v3, v4):
        combined = torch.cat([v1, v2, v3, v4], dim=1)  # [B, 128]
        gates = self.gate(combined)                      # [B, 4]

        # Apply gate weights to each stream
        g1 = gates[:, 0:1] * v1
        g2 = gates[:, 1:2] * v2
        g3 = gates[:, 2:3] * v3
        g4 = gates[:, 3:4] * v4

        fused = torch.cat([g1, g2, g3, g4], dim=1)     # [B, 128]
        return fused, gates


class QSFENet(nn.Module):
    """
    Quadrant-Stream Fusion EEG Network.
    Four clinically grounded streams fused via learned gating.
    """
    def __init__(self, num_classes=3):
        super().__init__()

        # Four stream encoders
        self.s1_encoder = StreamEncoder(95)   # frequency slowing
        self.s2_encoder = StreamEncoder(684)  # coherence
        self.s3_encoder = StreamEncoder(19)   # complexity
        self.s4_encoder = StreamEncoder(32)    # asymmetry

        # Gated fusion
        self.fusion = GatedFusion()

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, s1, s2, s3, s4):
        v1 = self.s1_encoder(s1)
        v2 = self.s2_encoder(s2)
        v3 = self.s3_encoder(s3)
        v4 = self.s4_encoder(s4)

        fused, gates = self.fusion(v1, v2, v3, v4)
        logits = self.classifier(fused)

        return logits, gates


if __name__ == "__main__":
    import torch

    model = QSFENet(num_classes=3)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass with dummy batch
    batch_size = 8
    s1 = torch.randn(batch_size, 95)
    s2 = torch.randn(batch_size, 684)
    s3 = torch.randn(batch_size, 19)
    s4 = torch.randn(batch_size, 32)

    logits, gates = model(s1, s2, s3, s4)

    print(f"Output logits shape: {logits.shape}")
    print(f"Gate weights shape: {gates.shape}")
    print(f"Sample gate weights: {gates[0].detach().numpy()}")


