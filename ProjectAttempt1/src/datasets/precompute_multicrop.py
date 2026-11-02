import json, os, sys, numpy as np
from tqdm import tqdm

sys.path.append(r"D:\Major\src\datasets")
from eeg_dataset import CAUEEGDataset

DATASET_PATH = r"D:\Major\caueeg-dataset"
EDF_DIR      = os.path.join(DATASET_PATH, "signal", "edf")
SAVE_DIR     = r"D:\Major\outputs\features_multicrop"
os.makedirs(SAVE_DIR, exist_ok=True)

N_CROPS_TRAIN = 8  # 8 crops per training patient earlier i put 5 but that didnt give validation accuracy above 57% and testing accuracy above 49 % 
N_CROPS_VALTEST = 1  # 1 fixed crop for val and test

with open(os.path.join(DATASET_PATH, "dementia.json")) as f:
    data = json.load(f)

def precompute_split(records, n_crops, split_name):
    print(f"\nProcessing {split_name} — {len(records)} patients x {n_crops} crops")
    
    for record in tqdm(records):
        serial = record["serial"]
        label  = record["class_label"]
        age    = record["age"]
        
        for crop_idx in range(n_crops):
            save_path = os.path.join(
                SAVE_DIR, f"{serial}_crop{crop_idx}.npz"
            )
            if os.path.exists(save_path):
                continue
            
            # Load and process
            import pyedflib
            edf_path = os.path.join(EDF_DIR, f"{serial}.edf")
            
            f = pyedflib.EdfReader(edf_path)
            signals = np.zeros((21, f.getNSamples()[0]))
            for i in range(21):
                signals[i, :] = f.readSignal(i)
            f.close()
            
            signal = signals[:19, :]  # drop EKG and Photic
            
            # Crop
            total = signal.shape[1]
            CROP_LENGTH = 2000
            
            if split_name == "train":
                # random crop
                start = np.random.randint(0, max(1, total - CROP_LENGTH))
            else:
                # fixed center crop
                start = (total - CROP_LENGTH) // 2
            
            signal = signal[:, start:start + CROP_LENGTH]
            
            # Normalize
            mean = signal.mean(axis=1, keepdims=True)
            std  = signal.std(axis=1, keepdims=True) + 1e-8
            signal = (signal - mean) / std
            
            # Extract features
            sys.path.append(r"D:\Major\src\datasets")
            from feature_extraction import extract_all_features
            features = extract_all_features(signal)
            
            np.savez(save_path,
                s1=features["s1_freq_slowing"],
                s2=features["s2_coherence"],
                s3=features["s3_complexity"],
                s4=features["s4_asymmetry"],
                label=np.array(label),
                age=np.array(age, dtype=np.float32)
            )

precompute_split(data["train_split"],      N_CROPS_TRAIN,   "train")
precompute_split(data["validation_split"], N_CROPS_VALTEST, "val")
precompute_split(data["test_split"],       N_CROPS_VALTEST, "test")

print("\nDone.")