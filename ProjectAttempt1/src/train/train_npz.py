import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json, os, sys
from tqdm import tqdm

sys.path.append(r"D:\Major\src\datasets")
sys.path.append(r"D:\Major\src\models")

from npz_dataset import NPZDataset
from qsfe_net import QSFENet

# Config
DATASET_PATH = r"D:\Major\caueeg-dataset"
FEATURE_DIR = r"D:\Major\outputs\features_multicrop"

BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_weights(records):
    counts = [0, 0, 0]
    for r in records:
        counts[r["class_label"]] += 1
    total = sum(counts)
    weights = [total / (3 * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    loop = tqdm(loader, desc="Training", leave=False)

    for batch in loop:
        s1 = batch["s1"].to(DEVICE, non_blocking=True)
        s2 = batch["s2"].to(DEVICE, non_blocking=True)
        s3 = batch["s3"].to(DEVICE, non_blocking=True)
        s4 = batch["s4"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits, _ = model(s1, s2, s3, s4)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            s1 = batch["s1"].to(DEVICE)
            s2 = batch["s2"].to(DEVICE)
            s3 = batch["s3"].to(DEVICE)
            s4 = batch["s4"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

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

    train_set = NPZDataset(data["train_split"], FEATURE_DIR, n_crops=8)
    val_set   = NPZDataset(data["validation_split"], FEATURE_DIR, n_crops=1)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = QSFENet(num_classes=3).to(DEVICE)

    weights = get_class_weights(data["train_split"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 8
    os.makedirs(r"D:\Major\outputs", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), r"D:\Major\outputs\qsfe_npz_best.pth")
            print("Best model saved")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()