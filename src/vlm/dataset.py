import json
import os


def load_dataset(json_path, base_folder):
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    for item in dataset:
        item["image_path"] = os.path.join(base_folder, item["related_images"])
    return dataset


def build_record(item, prediction, retrieved_context=None):
    record = {
        "unique_id": item["unique_id"],
        "question": item["question"],
        "question_type": item["question_type"],
        "answer": item["answer"],
        "prediction": prediction,
    }
    if retrieved_context is not None:
        record["retrieved_context"] = retrieved_context
    return record


def done_ids(path):
    """Ids already written to a predictions file, so a run can resume."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["unique_id"] for line in f if line.strip()}
