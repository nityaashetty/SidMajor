import json
import os

dataset_path = r"D:\Major\caueeg-dataset"
json_path = os.path.join(dataset_path, "dementia.json")

with open(json_path, "r") as f:
    data = json.load(f)

print(f"Type: {type(data)}")
print(f"Keys: {list(data.keys())}")
print(f"First key value: {data[list(data.keys())[0]]}")