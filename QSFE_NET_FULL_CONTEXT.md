# QSFE-Net — Full Project Context
## For Claude Code — Read Everything Before Touching Any File

---

> **You are Claude Code running in the IDE of Sanchit Raut (SPIT Mumbai, B.Tech CSE 2027).**
> This document is the single source of truth for this project.
> Do not make assumptions. Do not suggest revisiting closed decisions.
> Do not write code without first reading the relevant existing file.
> Ask for files one at a time if you need them.

---

## Project Identity

- **Project title:** Deep Learning-Based Cross-Dataset EEG Analysis for Early Dementia Detection
- **Team:** Sanchit Raut, Siddarth Potdar (2023800089), Vedant Kannurkar (2024801005)
- **Guide:** Dr. Sujata Kulkarni (doctorate-level EEG dementia researcher — treat as domain expert)
- **Institution:** SPIT Mumbai, B.Tech Major Project Group 18
- **Phase 1:** Already submitted and evaluated. Framing and narrative are locked.

---

## The Problem Being Solved

Dementia affects 57M people globally, projected to reach 153M by 2050. India alone: 5.1M cases in 2021, projected 20M+ by 2050. Early detection — before symptoms appear — is the only window where intervention helps.

Current AI approaches either:
1. Require MRI scanners (expensive, inaccessible, slow)
2. Use EEG-based models that are only ever tested on the data they were trained on — so reported accuracies are meaningless for real-world deployment

**Our goal:** A lightweight, interpretable EEG-based model that generalizes across datasets, running on basic hospital hardware. Not just accuracy — deployability and interpretability matter equally.

---

## Dataset

### Primary: CAUEEG (Chung-Ang University Hospital EEG)
- **Source:** Kim, Youn, Paik et al., NeuroImage 2023
- **Location:** `D:\Major\caueeg-dataset\`
- **Files:** 1,379 EDF files in `signal\edf\`
- **Signal:** 21 channels × ~145,600 samples per file @ 200 Hz (~12 min per patient)
- **Channel layout:**
  - Channels 0–18: EEG brain channels (Fp1, F3, C3, P3, O1, Fp2, F4, C4, P4, O2, F7, T3, T5, F8, T4, T6, FZ, CZ, PZ)
  - Channel 19: EKG — **always drop**
  - Channel 20: Photic stimulation — **always drop**
- **Labels:** `dementia.json` contains pre-made patient-wise splits
  - Train: 950 patients (Normal 367, MCI 334, Dementia 249)
  - Val: 119 patients (Normal 46, MCI 42, Dementia 31)
  - Test: 118 patients (Normal 46, MCI 41, Dementia 31)
- **Crop:** 2000 samples = 10 seconds at 200 Hz

### Cross-validation: UCI EEG Alzheimer's Dataset
- **Status:** NOT YET DOWNLOADED
- **Source:** Kaggle — "EEG Alzheimer dataset UCI"
- **Description:** 848,640 records, 16 EEG features + 1 label, no missing values
- **Purpose:** Cross-dataset validation — the core research claim

---

## What Dementia Does to EEG — Clinical Ground Truth

Understanding this is essential before any architecture decision:

1. **Frequency slowing** — Alpha waves (8–13 Hz) decrease. Theta (4–8 Hz) and delta (0.5–4 Hz) increase. Theta/alpha ratio rises with disease severity. This is the most reliably detectable marker.

2. **Coherence loss** — Long-range communication between brain regions breaks down. Frontal-parietal and temporal-parietal connections degrade first in Alzheimer's. The specific pattern of which pairs lose coherence matters — not the global mean.

3. **Complexity reduction** — Healthy brain EEG is chaotic and unpredictable (high entropy). Dementia brain is more regular and predictable (low entropy). But this is subtle and hard to extract from short windows.

4. **Hemispheric asymmetry** — Dementia affects left and right hemispheres unevenly. But population-level averages mask this because direction of asymmetry varies per patient.

---

## Phase 1: Initial Architecture — QSFE-Net

### Design philosophy
Built from first principles based on the four clinical phenomena above. Four parallel streams, one per phenomenon. A learned gating mechanism integrates them, weighted per patient.

### Architecture (final state, DO NOT change for Phase 1 evaluation)

```
StreamEncoder(input_dim):
    Linear(input_dim → 64) → BatchNorm1d → ReLU → Dropout(0.4)
    → Linear(64 → 32) → BatchNorm1d → ReLU
    Output: 32-dim vector

GatedFusion:
    Concat [v1, v2, v3, v4] → [B, 128]
    Gate network: Linear(128→64) → ReLU → Linear(64→4) → Sigmoid
    Apply: gi × vi for each stream i
    Output: concat(g1×v1, g2×v2, g3×v3, g4×v4) → [B, 128]

Classifier:
    Linear(128→64) → ReLU → Dropout(0.6) → Linear(64→3)
    Output: logits for Normal / MCI / Dementia
```

### Feature streams (Phase 1 final state)

| Stream | Input | Shape | Computation |
|--------|-------|-------|-------------|
| S1 | Theta/alpha ratio + 4 band powers × 19 ch | (95,) | Welch PSD per channel |
| S2 | Mean coherence per band per pair | (684,) | scipy.signal.coherence, 171 pairs × 4 bands |
| S3 | Spectral entropy per channel | (19,) | Entropy of Welch PSD |
| S4 | Multi-band hemispheric asymmetry | (32,) | 8 symmetric pairs × 4 bands |

Total features: 830 (95+684+19+32). Total parameters: 79,431.

### File structure (Phase 1)

```
D:\Major\
├── caueeg-dataset\
│   ├── signal\edf\              (1379 EDF files)
│   ├── dementia.json
│   └── annotation.json
├── src\
│   ├── datasets\
│   │   ├── load_edf.py               reads EDF, returns (21,T) numpy
│   │   ├── load_annotations.py       reads dementia.json splits
│   │   ├── eeg_dataset.py            PyTorch Dataset — random crop, normalize
│   │   ├── feature_extraction.py     computes S1/S2/S3/S4 features
│   │   ├── precompute_multicrop.py   precomputes 8 crops/patient, saves .npz
│   │   ├── npz_dataset.py            fast Dataset loading from .npz
│   │   └── eda.py                    EDA figures — already complete
│   ├── models\
│   │   └── qsfe_net.py               QSFE-Net architecture
│   └── train\
│       ├── train_npz.py              training loop
│       ├── evaluate.py               test set evaluation
│       └── ablation.py               ablation study
├── outputs\
│   ├── features_multicrop\           7837 .npz files (950×8 + 119 + 118)
│   ├── figures\                      6 EDA figures
│   ├── ablation\                     ablation results
│   ├── qsfe_npz_best.pth             Run 8 checkpoint (53.39% test, S1=95)
│   └── qsfe_run7_58.82.pth           Run 7 checkpoint (58.82% val, S1=19 — INCOMPATIBLE)
└── requirements.txt
```

---

## Phase 1 Results — Complete Training History

| Run | Key change | Best val acc | Notes |
|-----|-----------|-------------|-------|
| 1 | Baseline, LR=1e-3, 5 crops | 53.78% | Overfit after ep12 |
| 2 | LR=3e-4, multicrop | 47.06% | Severe overfit |
| 3 | Fixed features, 5 crops | 57.98% | First meaningful result |
| 4 | 8 crops, wd=3e-4, patience=8 | 52.94% | Too much regularization |
| 5 | wd=1e-4 | 53.78% | Oscillating |
| 6 | LR=1e-4 | 54.62% | Below Run 3 |
| 7 | LR=3e-4, 8 crops, wd=3e-4 | **58.82%** | Best val — S1=19 arch |
| 8 | S1 expanded to 95 features | 53.78% val / **53.39% test** | Best generalization |

**8 runs confirmed: accuracy ceiling is ~58-59%. Further hyperparameter tuning will not help.**

### Evaluate.py output — Run 8 (real, verified)

```
Test Accuracy: 53.39%
Macro F1: 0.5226

Confusion Matrix:
              Normal    MCI    Dementia
Normal            30     13           3
MCI               12     20           9
Dementia           6     12          13

Per-class metrics:
  Normal:   Precision 0.6250  Recall 0.6522  F1 0.6383
  MCI:      Precision 0.4444  Recall 0.4878  F1 0.4651
  Dementia: Precision 0.5200  Recall 0.4194  F1 0.4643

Gate weights per class:
  Normal:   S1=0.765  S2=0.975  S3=0.498  S4=0.470
  MCI:      S1=0.765  S2=0.978  S3=0.425  S4=0.387
  Dementia: S1=0.750  S2=0.978  S3=0.426  S4=0.329
```

### Ablation study results (real, verified)

| Model | Streams | Val Acc |
|-------|---------|---------|
| A | S1 only | 46.2% |
| B | S1 + S2 | **58.82%** |
| C | S1 + S2 + S3 | 49.58% |
| D | Full QSFE-Net (all 4) | 55.46% |
| E | All 4, no gating | 55.46% |

---

## Why Phase 1 Failed to Break 60% — Root Cause Analysis

This is the most important section. Understanding why we failed is what drives Phase 2.

### Problem 1: Feature computation collapses information

S2 coherence: we computed mean coherence per frequency band per electrode pair. This gives one number per pair per band. What we lost: the spatial pattern of which specific pairs are disconnected. Dementia doesn't uniformly reduce all coherence — it preferentially disrupts frontal-parietal connections. A global mean across 171 pairs hides exactly the information that distinguishes MCI from Dementia.

S3 entropy: we computed entropy of the amplitude histogram, then switched to spectral entropy. But spectral entropy computed on a 10-second window is a coarse statistic. The non-stationarity — how entropy changes moment-to-moment within a recording — is where the signal lives.

S4 asymmetry: population-level asymmetry averages near zero because some patients have left-dominant disruption and others have right-dominant. The mean hides the within-patient signal.

S1 alone showed a clean class gradient. The other three added noise, not signal.

### Problem 2: Random crop variance

Training saw one random 10-second window from a 12-minute recording per patient per epoch. The model learned to distinguish specific time windows rather than patient-level disease state. This is why train accuracy reached 77% while validation stayed at 53% — the model memorized which crops belonged to which patient class rather than learning what dementia looks like.

### Problem 3: Dataset size is the fundamental ceiling

950 training patients is genuinely small for deep learning. EEG has extremely high inter-subject variability — two dementia patients can have completely different-looking signals. The model has insufficient examples to learn a robust boundary, especially between MCI and Dementia which are clinically adjacent stages.

### Problem 4: Hand-crafted features impose a representational ceiling

By computing summary statistics (ratio, mean coherence, entropy, asymmetry) and feeding numbers to FC layers, we told the model exactly what to look at. If any of our features is computed suboptimally, the model cannot recover from it. The model can only weight features we gave it — it cannot discover that the time-varying pattern of coherence matters, or that specific electrode pairs are more diagnostic than others.

---

## Phase 2: New Approaches — Why We Are Starting Fresh

The goal shifts from 58% to 70-80% (90% is not achievable on 950 patients — even CEEDNet's ensemble of 10 models with 253M parameters reached 79.16% on the same dataset).

The fundamental change: stop hand-crafting summary statistics and let the model learn representations from richer input.

---

## 8 Approaches for Phase 2 — Detailed Specifications

---

### Approach 1 — EEGNet backbone (RECOMMENDED FIRST)
**Expected gain: 10-15 points. Start here.**

Replace each StreamEncoder's FC layers with a proper 1D CNN. EEGNet (Lawhern et al. 2018) is purpose-built for EEG with ~1,500 parameters. It learns temporal and spatial patterns jointly from raw signal rather than from precomputed statistics.

**Input change:** Raw signal [19, 2000] instead of precomputed feature vectors. No precomputation needed.

**Architecture:**
```python
class EEGNetEncoder(nn.Module):
    def __init__(self, n_channels=19, n_samples=2000, out_dim=32):
        super().__init__()
        # Temporal convolution: learns frequency filters
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16)
        )
        # Spatial (depthwise) convolution: learns channel combinations
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25)
        )
        # Separable convolution: reduces temporal dimension
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )
        # Compute output size dynamically
        self.fc = nn.Linear(self._get_flat_size(n_channels, n_samples), out_dim)

    def _get_flat_size(self, ch, samples):
        x = torch.zeros(1, 1, ch, samples)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return x.numel()

    def forward(self, x):
        # x: [B, 19, 2000] → unsqueeze to [B, 1, 19, 2000]
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        x = x.flatten(1)
        return self.fc(x)
```

**Gated fusion stays the same.** Four EEGNetEncoders replace four StreamEncoders. Each still outputs 32-dim. Gating and classifier unchanged.

**Why the four branches still make sense with EEGNet:**
Each branch receives the same raw signal but is initialized differently and trained with different gradient pathways — they learn different representations of the same signal. The gating mechanism selects which learned representation is most relevant per patient.

**Precomputation:** Not needed. Raw signal is fast to load from EDF. Use original `eeg_dataset.py`.

**Training config starting point:**
```python
LR = 1e-3          # higher LR works better for CNNs vs FC
BATCH_SIZE = 16    # EEGNet is memory-heavier
EPOCHS = 100
OPTIMIZER = AdamW, weight_decay=1e-4
SCHEDULER = CosineAnnealingLR, T_max=100
EARLY_STOP = 15    # give more patience for CNN convergence
```

---

### Approach 2 — Functional connectivity matrix as S2

**Expected gain: 8-12 points, concentrated on MCI-Dementia boundary.**

Instead of mean coherence per band (collapsing to 684 numbers), compute the full 19×19 coherence matrix per frequency band — 4 matrices stacked to [4, 19, 19]. Feed through a 2D CNN that learns which specific electrode pairs lose connectivity in dementia.

**Feature computation change (only S2):**
```python
def compute_stream2_matrix(signal):
    """
    Full functional connectivity matrices.
    Output shape: (4, 19, 19)
    """
    n_ch = signal.shape[0]
    bands = [DELTA, THETA, ALPHA, BETA]
    matrices = np.zeros((4, n_ch, n_ch), dtype=np.float32)

    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                matrices[:, i, j] = 1.0
                continue
            f, coh = scipy_signal.coherence(
                signal[i], signal[j], fs=SAMPLE_RATE, nperseg=256
            )
            for b_idx, band in enumerate(bands):
                idx = np.logical_and(f >= band[0], f <= band[1])
                matrices[b_idx, i, j] = np.mean(coh[idx])

    return matrices  # shape: (4, 19, 19)
```

**S2 encoder change:**
```python
class ConnectivityEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # → [B, 32, 4, 4]
        )
        self.fc = nn.Linear(32 * 4 * 4, out_dim)

    def forward(self, x):
        # x: [B, 4, 19, 19]
        x = self.cnn(x)
        x = x.flatten(1)
        return self.fc(x)
```

**Why this specifically helps MCI vs Dementia:** MCI shows early disruption of the default mode network (medial prefrontal to posterior cingulate connectivity). Dementia shows additional salience network disruption. A 2D CNN on the 19×19 matrix can learn these network-level signatures that global mean coherence completely hides.

---

### Approach 3 — Contrastive pretraining (SimCLR on EEG)

**Expected gain: 12-18 points. Highest impact, most complex.**

Use all 1,379 EDF files (including the abnormal split not used in dementia training) for self-supervised pretraining before classification. SimCLR: two random crops from the same patient are a positive pair, crops from different patients are negatives. Train encoder to produce similar representations for same-patient crops.

**Why this works on small labeled datasets:** The labels are the bottleneck (950 patients). Self-supervised pretraining learns general EEG structure from all 1,379 patients without needing labels. The classification fine-tuning then has a much better starting point.

**Pretraining loop:**
```python
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2: [B, projection_dim]
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2B, dim]
        z = F.normalize(z, dim=1)
        sim = torch.mm(z, z.T) / self.temperature  # [2B, 2B]
        # mask out diagonal
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)
        # positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2*B),
            torch.arange(0, B)
        ]).to(z.device)
        return F.cross_entropy(sim, labels)

# Pretraining:
# 1. Load any EDF file (including abnormal split)
# 2. Take two random crops → two augmented views
# 3. Pass both through EEGNet encoder + projection head
# 4. Compute NT-Xent loss
# 5. Train 50 epochs
# Fine-tuning:
# 1. Remove projection head
# 2. Add gated fusion + classifier
# 3. Train on 950 labeled patients
```

---

### Approach 4 — Temporal segment aggregation

**Expected gain: 10-15 points. Directly solves the random crop problem.**

Divide the full 12-minute recording into ~72 non-overlapping 10-second segments. Encode each, then aggregate across time using attention that learns which segments are most informative.

```python
class TemporalAggregator(nn.Module):
    def __init__(self, encoder, feat_dim=32):
        super().__init__()
        self.encoder = encoder
        self.attention = nn.Linear(feat_dim, 1)

    def forward(self, segments):
        # segments: [B, n_segs, C, T]
        B, N, C, T = segments.shape
        segs_flat = segments.view(B * N, C, T)
        enc = self.encoder(segs_flat)      # [B*N, 32]
        enc = enc.view(B, N, -1)           # [B, N, 32]
        weights = self.attention(enc)       # [B, N, 1]
        weights = F.softmax(weights, dim=1)
        out = (weights * enc).sum(dim=1)   # [B, 32]
        return out
```

**This eliminates random crop variance entirely.** The model sees all segments and learns to upweight the most discriminative ones.

---

### Approach 5 — Graph Neural Network on electrode connectivity

**Expected gain: 8-15 points, especially for MCI-Dementia boundary.**

Model 19 electrodes as graph nodes. Edge weights = coherence between electrode pairs. A GCN propagates information along edges — learning which brain network disruptions matter.

```python
from torch_geometric.nn import GCNConv, global_mean_pool

class EEGGraphEncoder(nn.Module):
    def __init__(self, node_features=5, out_dim=32):
        super().__init__()
        # node_features = 5 band powers per electrode
        self.conv1 = GCNConv(node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 32)
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x, edge_index, edge_weight, batch):
        # x: [19*B, 5] node features
        # edge_index: electrode connectivity graph
        # edge_weight: coherence values
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.conv3(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)    # [B, 32]
        return self.fc(x)
```

**Build edge_index from all 171 electrode pairs. Edge weights = alpha coherence per pair.**
Install: `pip install torch_geometric`

---

### Approach 6 — STFT spectrogram CNN (replaces S1 and S3)

**Expected gain: 12-18 points. Highest single-feature-change impact.**

Instead of Welch PSD (time-averaged), compute STFT spectrogram — a 2D image showing how frequency content changes over time within each 10-second window. This captures non-stationarity: dementia patients have unstable, fluctuating brainwave patterns.

```python
def compute_spectrogram_stream(signal, fs=200, n_fft=64, hop=8):
    """
    Input: signal [19, 2000]
    Output: [19, n_freqs, n_frames] ≈ [19, 33, 250]
    """
    spectrograms = []
    for ch in range(signal.shape[0]):
        f, t, Zxx = scipy_signal.stft(
            signal[ch], fs=fs, nperseg=n_fft, noverlap=n_fft-hop
        )
        power = np.abs(Zxx) ** 2
        spectrograms.append(power)
    return np.stack(spectrograms, axis=0).astype(np.float32)

class SpectrogramEncoder(nn.Module):
    def __init__(self, n_channels=19, out_dim=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, x):
        # x: [B, 19, n_freqs, n_frames]
        x = self.cnn(x)
        return self.fc(x.flatten(1))
```

**This directly replaces S1 and S3** — both become unnecessary because the STFT captures frequency slowing (visible as power shift to lower frequencies over time) and complexity (visible as texture variation in the spectrogram).

---

### Approach 7 — Multi-task learning with abnormality detection

**Expected gain: 8-12 points. Low effort, reliable.**

CAUEEG has two tasks: dementia (3-class, 950 patients) and abnormality (2-class, larger dataset). Train on both simultaneously. The abnormality task acts as a regularizer preventing dementia-specific overfitting.

```python
class MultiTaskQSFENet(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared encoder (EEGNet or STFT-CNN)
        self.shared_encoder = EEGNetEncoder(out_dim=128)
        # Task-specific heads
        self.dementia_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 3)   # Normal / MCI / Dementia
        )
        self.abnormal_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 2)   # Normal / Abnormal
        )

    def forward(self, x, task='dementia'):
        feat = self.shared_encoder(x)
        if task == 'dementia':
            return self.dementia_head(feat)
        return self.abnormal_head(feat)

# Loss: 0.7 × dementia_loss + 0.3 × abnormal_loss
```

**Training loop alternates batches between dementia and abnormality task.**
The abnormality dataset has more patients → better generalization of the shared encoder.

---

### Approach 8 — LaBraM fine-tuning (highest ceiling)

**Expected gain: 20-30 points. Gets you to 75-85%. Most complex.**

LaBraM (Large Brain Model, Yang et al. 2024) is pretrained on 2,500 hours of EEG across 15 datasets. Available on HuggingFace. Fine-tune the top 2 transformer layers on your 950 labeled patients.

```python
from transformers import AutoModel

class LaBraMDementia(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("935yang/LaBraM-base")
        # Freeze all but last 2 transformer layers
        for name, param in self.backbone.named_parameters():
            if 'layer.10' not in name and 'layer.11' not in name:
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        # x: [B, 19, 2000] — LaBraM expects standard 10-20 EEG
        out = self.backbone(x).last_hidden_state[:, 0, :]  # CLS token
        return self.classifier(out)
```

**Compute requirement:** LaBraM-base fits in 6GB VRAM with gradient checkpointing enabled. RTX 4060 (8GB) can handle it with batch size 8.
```python
from torch.utils.checkpoint import checkpoint_sequential
# Enable in backbone forward pass
```

---

## Recommended Phase 2 Implementation Order

**If starting completely fresh, follow this order:**

### Stage 1 — Approach 1 + Approach 4 together (3-4 weeks)

Implement EEGNet as encoder AND temporal segment aggregation simultaneously. These solve the two biggest problems: representational ceiling (hand-crafted features) and random crop variance. Expected result: 68-72%.

**New data pipeline:**
```
raw EDF → 72 segments × [19, 2000]
→ EEGNetEncoder per segment → [72, 32]
→ TemporalAttentionAggregator → [32] per stream
→ GatedFusion (unchanged) → Classifier (unchanged)
```

No precomputation needed. Load EDF directly in DataLoader.

### Stage 2 — Approach 2 (1 week)

Replace S2 mean coherence with full 19×19 connectivity matrix + 2D CNN. Precompute connectivity matrices if needed (less heavy than full coherence). Expected additional gain: 5-8%.

### Stage 3 — Approach 7 (1 week)

Add multi-task training with abnormality detection. Low effort, reliable regularization. Expected additional gain: 3-5%.

### Stage 4 — Approach 3 or 8 (2-3 weeks)

Either contrastive pretraining (more publishable, more novel) or LaBraM fine-tuning (higher ceiling, less novel since it's standard transfer learning). Expected additional gain: 10-15%.

**Realistic target: 78-82% with Stages 1-4 complete.**

---

## What NOT to Do (Closed Decisions)

- Do not suggest retraining the Phase 1 QSFE-Net model — 8 runs confirmed the ceiling
- Do not suggest hyperparameter tuning as the solution — proven ineffective
- Do not use `outputs/features/` (single crop, deprecated)
- Do not use `precompute_features.py` (deprecated)
- Do not claim 90% is achievable on 950 patients — CEEDNet ensemble reached 79.16% with 253M parameters
- Do not change Phase 1 architecture for Phase 1 evaluation purposes

---

## Hardware

| Item | Detail |
|------|--------|
| Machine | HP OMEN 16 (16-xf0xxx) |
| CPU | AMD Ryzen 7 7840HS |
| GPU | NVIDIA RTX 4060 Laptop 8GB GDDR6 |
| RAM | 16GB DDR5 |
| OS | Windows 11 |
| Python | 3.12, PyTorch CUDA 12.1 |
| GPU mode | Discrete only |

**Always:** charger plugged in, OMEN Gaming Hub Comfort mode, stop if CPU > 85°C.

```python
import torch
assert torch.cuda.is_available()
assert 'RTX 4060' in torch.cuda.get_device_name(0)
```

---

## Key Numbers — Single Reference

| Metric | Value |
|--------|-------|
| Train / Val / Test patients | 950 / 119 / 118 |
| Phase 1 best val accuracy | 58.82% (Run 7) |
| Phase 1 best test accuracy | 53.39% (Run 8) |
| Phase 1 Macro F1 | 0.5226 |
| Phase 1 val-test gap | 1.24 points |
| QSFE-Net Phase 1 parameters | 79,431 |
| CEEDNet single model best | 77.32% (25.7M params) |
| CEEDNet ensemble | 79.16% (253.8M params) |
| Realistic Phase 2 target | 75-82% |

---

## Scientific Findings from Phase 1 (Report in Paper)

1. S1 (frequency slowing) is the dominant discriminating EEG biomarker. Clear gradient: Normal 0.269 → MCI 0.334 → Dementia 0.463.
2. S2 (coherence) adds value: B (S1+S2) = 58.82% vs A (S1 only) = 46.2%.
3. S3 gate weight near-neutral (0.498) for Normal, suppressed (0.425) for MCI/Dementia — clinically meaningful pattern.
4. S4 gate shows progressive suppression: 0.470 → 0.387 → 0.329 — asymmetry becomes less discriminating in severe dementia.
5. Ablation: B (S1+S2) outperforms full model D — S3/S4 introduce noise at 950-patient scale.
6. Gating and concatenation tie in accuracy (55.46%) — gating justified for interpretability.
7. Dataset size (950 patients) confirmed as the accuracy bottleneck. This is an honest scientific finding.
8. No artifact rejection applied — Normal patient EEG shows blink artifacts. Preprocessing limitation for paper.

---

*Version: Final Phase 1 + Phase 2 planning*
*Date: May 2026*
*Author: Sanchit Raut (Group 18, SPIT Mumbai)*
*For Claude Code: this file is the single source of truth. Ask for existing files before editing.*
