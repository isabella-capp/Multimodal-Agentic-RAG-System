"""Run Qwen2.5-VL inference over Encyclopedic-VQA and save predictions as JSONL.

The run is resumable: already-predicted unique_ids are skipped on restart.
"""

import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_model import QwenVQAModel, load_dataset


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["unique_id"] for line in f if line.strip()}


def build_record(item, prediction):
    return {
        "unique_id": item["unique_id"],
        "question": item["question"],
        "question_type": item["question_type"],
        "answer": item["answer"],
        "prediction": prediction,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL inference on Encyclopedic-VQA")
    parser.add_argument("--json-path", default="/work/cvcs2026/encyclopedic/encyclopedic_test_subset.json")
    parser.add_argument("--base-folder", default="/work/cvcs2026/encyclopedic")
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(args.json_path, args.base_folder)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done_ids = load_done_ids(args.output)
    if done_ids:
        print(f"Skipping {len(done_ids)} already-predicted examples")

    model = QwenVQAModel(model_name=args.model_name)

    with open(args.output, "a", encoding="utf-8") as out:
        for item in tqdm(dataset, desc="Inference"):
            if item["unique_id"] in done_ids:
                continue

            image_path = item["image_path"]
            if os.path.exists(image_path):
                prediction = model.generate_response(image_path, item["question"])
            else:
                tqdm.write(f"missing image: {image_path}")
                prediction = None

            out.write(json.dumps(build_record(item, prediction), ensure_ascii=False) + "\n")
            out.flush()

    print(f"Done. Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
