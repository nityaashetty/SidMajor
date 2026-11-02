import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json, os, sys

sys.path.append(r"D:\Major\src\datasets")
sys.path.append(r"D:\Major\src\models")

from eeg_dataset import CAUEEGDataset
from qsfe_net import QSFENet

# Config
DATASET_PATH = r"D:\Major\caueeg-dataset"
EDF_DIR      = os.path.join(DATASET_PATH, "signal", "edf")
BATCH_SIZE   = 16
EPOCHS       = 30
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_class_weights(records):
    counts = [0, 0, 0]
    for r in records:
        counts[r["class_label"]] += 1
    total = sum(counts)
    weights = [total / (3 * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in loader:
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        s3 = batch["s3"].to(device)
        s4 = batch["s4"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits, gates = model(s1, s2, s3, s4)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            s1 = batch["s1"].to(device)
            s2 = batch["s2"].to(device)
            s3 = batch["s3"].to(device)
            s4 = batch["s4"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(s1, s2, s3, s4)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

def main():
    print(f"Using device: {DEVICE}")

    with open(os.path.join(DATASET_PATH, "dementia.json")) as f:
        data = json.load(f)

    train_set = CAUEEGDataset(data["train_split"], EDF_DIR, augment=True)
    val_set   = CAUEEGDataset(data["validation_split"], EDF_DIR, augment=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model = QSFENet(num_classes=3).to(DEVICE)

    # Weighted loss to handle class imbalance
    weights = get_class_weights(data["train_split"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    os.makedirs(r"D:\Major\outputs", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       r"D:\Major\outputs\qsfe_best.pth")
            print(f"  --> Best model saved (val acc: {val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()