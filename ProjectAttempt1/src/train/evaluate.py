import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json, os, sys
import numpy as np
from tqdm import tqdm

sys.path.append(r"D:\Major\src\datasets")
sys.path.append(r"D:\Major\src\models")

from npz_dataset import NPZDataset
from qsfe_net import QSFENet

# Config
DATASET_PATH  = r"D:\Major\caueeg-dataset"
FEATURE_DIR   = r"D:\Major\outputs\features_multicrop"
CHECKPOINT = r"D:\Major\outputs\qsfe_npz_best.pth"
BATCH_SIZE    = 32
CLASS_NAMES   = ["Normal", "MCI", "Dementia"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_test(model, loader):
    model.eval()
    all_preds  = []
    all_labels = []
    all_gates  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            s1     = batch["s1"].to(DEVICE)
            s2     = batch["s2"].to(DEVICE)
            s3     = batch["s3"].to(DEVICE)
            s4     = batch["s4"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits, gates = model(s1, s2, s3, s4)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_gates.append(gates.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_gates  = np.concatenate(all_gates)

    return all_preds, all_labels, all_gates


def confusion_matrix_manual(preds, labels, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(labels, preds):
        cm[true][pred] += 1
    return cm


def print_confusion_matrix(cm):
    print("\nConfusion Matrix (rows=True, cols=Predicted):")
    header = f"{'':>12}" + "".join(f"{name:>12}" for name in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name:>12}" + "".join(f"{cm[i][j]:>12}" for j in range(3))
        print(row)


def compute_per_class_metrics(cm):
    metrics = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        metrics[name] = {
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "support":   int(cm[i, :].sum())
        }
    return metrics


def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT}\n")

    with open(os.path.join(DATASET_PATH, "dementia.json")) as f:
        data = json.load(f)

    test_set = NPZDataset(data["test_split"], FEATURE_DIR, n_crops=1)
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Test samples: {len(test_set)}")

    model = QSFENet(num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    print("Checkpoint loaded.\n")

    preds, labels, gates = evaluate_test(model, test_loader)

    # Overall accuracy
    accuracy = (preds == labels).mean()
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Confusion matrix
    cm = confusion_matrix_manual(preds, labels)
    print_confusion_matrix(cm)

    # Per-class metrics
    metrics = compute_per_class_metrics(cm)
    print("\nPer-Class Metrics:")
    print(f"{'Class':>12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}")
    for name, m in metrics.items():
        print(f"{name:>12}{m['precision']:>12.4f}{m['recall']:>12.4f}"
              f"{m['f1']:>12.4f}{m['support']:>12}")

    # Macro F1
    macro_f1 = np.mean([m["f1"] for m in metrics.values()])
    print(f"\nMacro F1: {macro_f1:.4f}")

    # Gate weights per class
    print("\nMean Gate Weights per Class (S1, S2, S3, S4):")
    for cls_idx, name in enumerate(CLASS_NAMES):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        mean_gates = gates[mask].mean(axis=0)
        print(f"  {name}: S1={mean_gates[0]:.3f}  S2={mean_gates[1]:.3f}"
              f"  S3={mean_gates[2]:.3f}  S4={mean_gates[3]:.3f}")


if __name__ == "__main__":
    main()