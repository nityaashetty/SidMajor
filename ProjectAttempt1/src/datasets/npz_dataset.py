import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    def __init__(self, records, feature_dir, n_crops=1):
        self.feature_dir = feature_dir
        self.n_crops = n_crops
        
        # Build flat list of (serial, crop_idx, label, age)
        self.samples = []
        for record in records:
            serial = record["serial"]
            label  = record["class_label"]
            age    = record["age"]
            for crop_idx in range(n_crops):
                path = os.path.join(
                    feature_dir, f"{serial}_crop{crop_idx}.npz"
                )
                if os.path.exists(path):
                    self.samples.append((serial, crop_idx, label, age))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        serial, crop_idx, label, age = self.samples[idx]
        path = os.path.join(
            self.feature_dir, f"{serial}_crop{crop_idx}.npz"
        )
        data = np.load(path)

        return {
            "s1": torch.tensor(data["s1"], dtype=torch.float32),
            "s2": torch.tensor(data["s2"], dtype=torch.float32),
            "s3": torch.tensor(data["s3"], dtype=torch.float32),
            "s4": torch.tensor(data["s4"], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "age": torch.tensor(age,   dtype=torch.float32)
        }