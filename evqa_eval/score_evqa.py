"""Evaluate saved predictions with the Encyclopedic-VQA metric (Exact Match + BEM).

Reads the JSONL produced by src/vlm/run_inference.py and reports overall and
per-question-type accuracy.
"""

import os

# Must be set before TensorFlow is imported (through evaluation_utils).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
from collections import defaultdict

from tqdm import tqdm

import evaluation_utils


def read_predictions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_prediction(record, scoring_function):
    if record["prediction"] is None:
        return 0.0
    example = {
        "question": record["question"],
        "reference": record["answer"],
        "candidate": record["prediction"],
        "question_type": record["question_type"],
    }
    return scoring_function(example)


def parse_args():
    parser = argparse.ArgumentParser(description="Encyclopedic-VQA evaluation")
    parser.add_argument("--predictions", default="../outputs/predictions.jsonl")
    parser.add_argument("--output", default="../outputs/results.json")
    return parser.parse_args()


def main():
    args = parse_args()

    records = read_predictions(args.predictions)
    print(f"Loaded {len(records)} predictions from {args.predictions}")

    scoring_function = evaluation_utils.initialize_encyclopedic_vqa_evaluation_function()

    scores_by_type = defaultdict(list)
    for record in tqdm(records, desc="Scoring"):
        score = score_prediction(record, scoring_function)
        scores_by_type[record["question_type"]].append(score)

    all_scores = [s for scores in scores_by_type.values() for s in scores]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    print(f"\nOverall accuracy: {overall:.4f} (n={len(all_scores)})")
    for qtype in sorted(scores_by_type):
        scores = scores_by_type[qtype]
        print(f"  {qtype:15s}: {sum(scores) / len(scores):.4f} (n={len(scores)})")

    summary = {
        "num_examples": len(all_scores),
        "accuracy_overall": overall,
        "accuracy_by_type": {q: sum(s) / len(s) for q, s in scores_by_type.items()},
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {args.output}")


if __name__ == "__main__":
    main()
