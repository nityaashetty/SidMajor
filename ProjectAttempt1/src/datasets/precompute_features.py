import json, os, sys, numpy as np
from tqdm import tqdm

sys.path.append(r"D:\Major\src\datasets")
from eeg_dataset import CAUEEGDataset
from feature_extraction import extract_all_features

DATASET_PATH = r"D:\Major\caueeg-dataset"
EDF_DIR      = os.path.join(DATASET_PATH, "signal", "edf")
SAVE_DIR     = r"D:\Major\outputs\features"
os.makedirs(SAVE_DIR, exist_ok=True)

with open(os.path.join(DATASET_PATH, "dementia.json")) as f:
    data = json.load(f)

all_records = (data["train_split"] + 
               data["validation_split"] + 
               data["test_split"])

dataset = CAUEEGDataset(all_records, EDF_DIR, augment=False)

print(f"Precomputing features for {len(dataset)} samples...")

for i in tqdm(range(len(dataset))):
    serial = all_records[i]["serial"]
    save_path = os.path.join(SAVE_DIR, f"{serial}.npz")
    
    if os.path.exists(save_path):
        continue  # skip already computed
    
    sample = dataset[i]
    
    np.savez(save_path,
        s1=sample["s1"].numpy(),
        s2=sample["s2"].numpy(),
        s3=sample["s3"].numpy(),
        s4=sample["s4"].numpy(),
        label=sample["label"].numpy(),
        age=sample["age"].numpy()
    )

print("Done. All features saved to", SAVE_DIR)