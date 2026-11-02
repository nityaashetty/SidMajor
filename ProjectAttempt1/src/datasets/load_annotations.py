import json
import os

def load_dementia_annotations(dataset_path):
    json_path = os.path.join(dataset_path, "dementia.json")
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    return {
        "class_labels": data["class_label_to_name"],
        "train": data["train_split"],
        "val": data["validation_split"],
        "test": data["test_split"]
    }

if __name__ == "__main__":
    dataset_path = r"D:\Major\caueeg-dataset"
    annotations = load_dementia_annotations(dataset_path)
    
    print(f"Classes: {annotations['class_labels']}")
    print(f"Train: {len(annotations['train'])} samples")
    print(f"Val: {len(annotations['val'])} samples")
    print(f"Test: {len(annotations['test'])} samples")
    
    # Check class distribution in training set
    from collections import Counter
    train_labels = [r["class_name"] for r in annotations["train"]]
    print(f"\nTrain class distribution: {Counter(train_labels)}")
    
    val_labels = [r["class_name"] for r in annotations["val"]]
    print(f"Val class distribution: {Counter(val_labels)}")
    
    test_labels = [r["class_name"] for r in annotations["test"]]
    print(f"Test class distribution: {Counter(test_labels)}")