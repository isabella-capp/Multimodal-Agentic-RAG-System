import json
import os


def load_dataset(json_path, base_folder):
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    for item in dataset:
        item["image_path"] = os.path.join(base_folder, item["related_images"])
    return dataset
