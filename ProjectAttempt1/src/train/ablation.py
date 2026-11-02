import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json, os, sys, numpy as np
from tqdm import tqdm

sys.path.append(r"D:\Major\src\datasets")
sys.path.append(r"D:\Major\src\models")

from npz_dataset import NPZDataset

DATASET_PATH = r"D:\Major\caueeg-dataset"
FEATURE_DIR  = r"D:\Major\outputs\features_multicrop"
SAVE_DIR     = r"D:\Major\outputs\ablation"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS     = 50
LR         = 3e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOP = 8


def get_class_weights(records):
    counts = [0, 0, 0]
    for r in records:
        counts[r["class_label"]] += 1
    total = sum(counts)
    return torch.tensor([total/(3*c) for c in counts], dtype=torch.float32)


class StreamEncoder(nn.Module):
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


class AblationModel(nn.Module):
    """
    Flexible model for ablation study.
    streams: list of stream names to include e.g. ['s1','s2','s3','s4']
    use_gating: if False, simple concatenation instead of learned gates
    """
    def __init__(self, streams, use_gating=True, num_classes=3):
        super().__init__()
        self.streams    = streams
        self.use_gating = use_gating

        # Input dims per stream
        dims = {'s1': 95, 's2': 684, 's3': 19, 's4': 32}

        self.encoders = nn.ModuleDict({
            s: StreamEncoder(dims[s]) for s in streams
        })

        n = len(streams)
        fused_dim = 32 * n

        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(fused_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n),
                nn.Sigmoid()
            )

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, num_classes)
        )

    def forward(self, batch):
        vectors = [self.encoders[s](batch[s].to(DEVICE)) for s in self.streams]
        concat  = torch.cat(vectors, dim=1)

        if self.use_gating and len(self.streams) > 1:
            gates   = self.gate(concat)
            gated   = torch.cat([
                gates[:, i:i+1] * vectors[i]
                for i in range(len(self.streams))
            ], dim=1)
            return self.classifier(gated)
        else:
            return self.classifier(concat)


def train_eval(model, train_loader, val_loader, weights, name):
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")

    criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc    = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        correct, total = 0, 0
        for batch in train_loader:
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        train_acc = correct / total

        # Val
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["label"].to(DEVICE)
                logits = model(batch)
                correct += (logits.argmax(1) == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total

        scheduler.step()

        print(f"Epoch {epoch:02d} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f"{name}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\n{name} — Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc


def main():
    print(f"Device: {DEVICE}")

    with open(os.path.join(DATASET_PATH, "dementia.json")) as f:
        data = json.load(f)

    train_set = NPZDataset(data["train_split"], FEATURE_DIR, n_crops=8)
    val_set   = NPZDataset(data["validation_split"], FEATURE_DIR, n_crops=1)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    weights = get_class_weights(data["train_split"])

    # Five ablation versions
    configs = [
        ("A_S1_only",       ['s1'],              True),
        ("B_S1_S2",         ['s1','s2'],         True),
        ("C_S1_S2_S3",      ['s1','s2','s3'],    True),
        ("D_full_QSFE",     ['s1','s2','s3','s4'],True),
        ("E_no_gating",     ['s1','s2','s3','s4'],False),
    ]

    results = {}
    for name, streams, use_gating in configs:
        model = AblationModel(streams, use_gating).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{name} | Streams: {streams} | Gating: {use_gating} | Params: {total_params:,}")
        best = train_eval(model, train_loader, val_loader, weights, name)
        results[name] = best

    print("\n" + "="*50)
    print("ABLATION STUDY RESULTS")
    print("="*50)
    print(f"{'Model':<20} {'Val Acc':>10}")
    for name, acc in results.items():
        print(f"{name:<20} {acc:>10.4f}")

    # Save results
    with open(os.path.join(SAVE_DIR, "ablation_results.txt"), "w") as f:
        f.write("ABLATION STUDY RESULTS\n")
        f.write("="*40 + "\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc:.4f}\n")

    print(f"\nResults saved to {SAVE_DIR}/ablation_results.txt")


if __name__ == "__main__":
    main()
