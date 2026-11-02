from feature_extraction import extract_all_features
import torch
from torch.utils.data import Dataset
import pyedflib
import numpy as np
import json
import os

EEG_CHANNELS = list(range(19))  # drop EKG (19) and Photic (20)
SAMPLE_RATE = 200
CROP_LENGTH = 2000  # 10 seconds

class CAUEEGDataset(Dataset):
    def __init__(self, records, edf_dir, augment=False):
        self.records = records
        self.edf_dir = edf_dir
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        serial = record["serial"]
        label = record["class_label"]
        age = record["age"]

        edf_path = os.path.join(self.edf_dir, f"{serial}.edf")
        signal = self._load_edf(edf_path)
        signal = self._crop(signal)
        signal = self._normalize(signal)

        if self.augment:
            signal = self._add_noise(signal)

        features = extract_all_features(signal)

        return {
            "signal": torch.tensor(signal, dtype=torch.float32),
            "s1": torch.tensor(features["s1_freq_slowing"], dtype=torch.float32),
            "s2": torch.tensor(features["s2_coherence"], dtype=torch.float32),
            "s3": torch.tensor(features["s3_complexity"], dtype=torch.float32),
            "s4": torch.tensor(features["s4_asymmetry"], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "age": torch.tensor(age, dtype=torch.float32)
        }

    def _load_edf(self, path):
        f = pyedflib.EdfReader(path)
        signals = np.zeros((21, f.getNSamples()[0]))
        for i in range(21):
            signals[i, :] = f.readSignal(i)
        f.close()
        return signals[EEG_CHANNELS, :]  # keep only 19 EEG channels

    def _crop(self, signal):
        total = signal.shape[1]
        if total <= CROP_LENGTH:
            # pad if shorter than crop length
            pad = CROP_LENGTH - total
            signal = np.pad(signal, ((0, 0), (0, pad)))
            return signal
        # random crop
        start = np.random.randint(0, total - CROP_LENGTH)
        return signal[:, start:start + CROP_LENGTH]

    def _normalize(self, signal):
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True) + 1e-8
        return (signal - mean) / std

    def _add_noise(self, signal):
        noise = np.random.normal(0, 0.01, signal.shape)
        return signal + noise


if __name__ == "__main__":
    import json
    
    dataset_path = r"D:\Major\caueeg-dataset"
    edf_dir = os.path.join(dataset_path, "signal", "edf")
    
    with open(os.path.join(dataset_path, "dementia.json")) as f:
        data = json.load(f)
    
    train_records = data["train_split"]
    
    dataset = CAUEEGDataset(train_records, edf_dir, augment=False)
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Signal shape: {sample['signal'].shape}")
    print(f"Label: {sample['label']}")
    print(f"Age: {sample['age']}")
    print(f"Signal mean: {sample['signal'].mean():.4f}")
    print(f"Signal std: {sample['signal'].std():.4f}")