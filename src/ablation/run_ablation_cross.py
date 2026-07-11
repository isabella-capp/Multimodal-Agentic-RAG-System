"""All-in-one ablation study for cross-encoder reranking parameters.

Loads all heavy models (Qwen VLM, EVA-CLIP retriever, cross-encoder reranker)
**once**, then iterates over a grid of (top_k, rerank_top_n) configurations.
For each configuration it runs inference on the validation set and saves both
predictions and per-config results.

Usage (SLURM):
    See scripts/run_ablation_cross.sh

Usage (interactive):
    uv run python scripts/run_ablation_cross.py --val-json <path> [options]
"""

import argparse
import itertools
import json
import os
import sys
import time

# Ensure project root is on sys.path so we can import src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from collections import defaultdict
from PIL import Image
from tqdm import tqdm

from retrieval.retriever import Retriever
from retrieval.knowledge_base import KnowledgeBase
from retrieval.reranker import CrossEncoderReranker
from vlm.qwen_model import QwenVQAModel, load_dataset
from vlm.run_inference import build_rag_prompt, build_record

BASE_FOLDER = "/work/cvcs2026/encyclopedic"


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Ablation study: top-k × rerank-top-n (cross-encoder)")

    # Paths
    p.add_argument("--val-json", default=f"{BASE_FOLDER}/encyclopedic_val_split.json",
                    help="Path to the validation split JSON.")
    p.add_argument("--base-folder", default=BASE_FOLDER)
    p.add_argument("--output-dir", default="outputs/ablation",
                    help="Directory for per-config prediction & result files.")

    # Model
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")

    # Retriever
    p.add_argument("--img-index-path", default=f"{BASE_FOLDER}/knn.index")
    p.add_argument("--img-index-json-path", default=f"{BASE_FOLDER}/knn.json")
    p.add_argument("--kb-path", default=f"{BASE_FOLDER}/encyclopedic_kb_wiki.json")
    p.add_argument("--retriever-device", default="cuda")

    # Cross-encoder
    p.add_argument("--cross-encoder-model", default="BAAI/bge-reranker-base")

    # Grid (space-separated lists)
    p.add_argument("--top-k-values", type=int, nargs="+", default=[5, 10, 20, 50, 80],
                    help="List of top-k values to test.")
    p.add_argument("--rerank-top-n-values", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30, 35],
                    help="List of rerank-top-n values to test.")

    # Debug
    p.add_argument("--debug-samples", type=int, default=1,
                    help="Print detailed trace for first N examples of each config.")
    p.add_argument("--limit", type=int, default=None,
                    help="Limit the number of validation examples (useful for testing).")

    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────

def _truncate(text: str, n: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " …"


def run_single_config(
    dataset, model, retriever, kb, reranker, top_k, rerank_top_n,
    output_path, debug_samples=1,
):
    """Run inference for a single (top_k, rerank_top_n) configuration.

    The retriever's top_k is temporarily overridden.
    Returns a list of prediction records.
    """
    # Override retriever top_k for this run
    retriever.top_k = top_k

    records = []
    debug_count = 0

    for item in tqdm(dataset, desc=f"  top_k={top_k} rerank_n={rerank_top_n}", leave=False):
        image_path = item["image_path"]

        if not os.path.exists(image_path):
            records.append(build_record(item, None))
            continue

        retrieved_context = None
        top_paragraphs = None
        prompt = item["question"]

        try:
            user_image = Image.open(image_path).convert("RGB")
            results = retriever.retrieve(user_image, item["question"])

            if results:
                pooled = []
                for r in results:
                    pooled.extend(kb.get_paragraphs_by_url(r["wiki_url"]))

                if pooled:
                    top_paragraphs = reranker.rerank(
                        item["question"], pooled, top_n=rerank_top_n
                    )
                    prompt = build_rag_prompt(item["question"], top_paragraphs)
                    retrieved_context = {
                        "wiki_url": results[0]["wiki_url"],
                        "title": results[0].get("title", ""),
                        "score": results[0].get("score"),
                        "candidates": [
                            {
                                "wiki_url": r["wiki_url"],
                                "title": r.get("title", ""),
                                "score": r.get("score"),
                            }
                            for r in results
                        ],
                        "num_paragraphs_total": len(pooled),
                        "num_paragraphs_used": len(top_paragraphs) if top_paragraphs else 0,
                    }
        except Exception as e:
            tqdm.write(f"  retrieval failed for {item['unique_id']}: {e}")

        prediction = model.generate_response(image_path, prompt)

        if debug_count < debug_samples:
            tqdm.write(f"\n  [DEBUG] {item['unique_id']} ({item['question_type']})")
            tqdm.write(f"    Q : {item['question']}")
            tqdm.write(f"    GT: {item['answer']}")
            if top_paragraphs:
                tqdm.write(f"    Paragraphs used: {len(top_paragraphs)}")
                for i, p in enumerate(top_paragraphs, 1):
                    tqdm.write(f"      [{i}] {_truncate(p)}")
            tqdm.write(f"    Pred: {prediction}")
            debug_count += 1

        record = build_record(item, prediction, retrieved_context)
        records.append(record)

    # Save predictions
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


def compute_scores_simple(records):
    """Compute accuracy using simple exact-match (case-insensitive substring).

    This is a lightweight proxy for the full BEM evaluation.
    The BEM scores are computed separately via score_evqa.py.
    """
    scores_by_type = defaultdict(list)

    for rec in records:
        pred = (rec.get("prediction") or "").strip().lower()
        gt = rec.get("answer", "").strip().lower()

        # Simple heuristic: prediction contains the answer or vice versa
        score = 1.0 if (gt in pred or pred in gt) and pred else 0.0
        scores_by_type[rec["question_type"]].append(score)

    all_scores = [s for scores in scores_by_type.values() for s in scores]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    accuracy_by_type = {
        q: sum(s) / len(s) for q, s in scores_by_type.items()
    }

    return {
        "num_examples": len(all_scores),
        "accuracy_overall": overall,
        "accuracy_by_type": accuracy_by_type,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load dataset ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Loading validation dataset …")
    dataset = load_dataset(args.val_json, args.base_folder)
    if args.limit is not None:
        dataset = dataset[:args.limit]
    print(f"Validation set: {len(dataset)} examples")

    qtypes = defaultdict(int)
    for item in dataset:
        qtypes[item["question_type"]] += 1
    for qt in sorted(qtypes):
        print(f"  {qt:20s}: {qtypes[qt]}")

    # ── Load models (ONCE) ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Loading Qwen VLM …")
    model = QwenVQAModel(model_name=args.model_name)

    print(f"\n{'='*70}")
    print("Loading EVA-CLIP retriever …")
    # Start with max top_k so FAISS is ready for all configurations
    max_top_k = max(args.top_k_values)
    retriever = Retriever(
        img_index_path=args.img_index_path,
        img_index_json_path=args.img_index_json_path,
        top_k=max_top_k,
        device=args.retriever_device,
    )
    # Force eager loading
    retriever._ensure_index()
    retriever._ensure_model()

    print(f"\n{'='*70}")
    print("Loading Knowledge Base …")
    kb = KnowledgeBase(args.kb_path)

    print(f"\n{'='*70}")
    print("Loading Cross-Encoder reranker …")
    reranker = CrossEncoderReranker(
        args.cross_encoder_model, device=args.retriever_device
    )

    # ── Grid search ──────────────────────────────────────────────────────
    grid = list(itertools.product(args.top_k_values, args.rerank_top_n_values))
    print(f"\n{'='*70}")
    print(f"Ablation grid: {len(grid)} configurations")
    for top_k, rerank_n in grid:
        print(f"  top_k={top_k:2d}  rerank_top_n={rerank_n}")
    print("=" * 70)

    all_results = {}

    for i, (top_k, rerank_n) in enumerate(grid, 1):
        config_name = f"cross_topK{top_k}_rerankN{rerank_n}"
        pred_path = os.path.join(args.output_dir, f"predictions_{config_name}.jsonl")
        result_path = os.path.join(args.output_dir, f"results_{config_name}.json")

        print(f"\n{'─'*70}")
        print(f"[{i}/{len(grid)}] Running config: {config_name}")
        print(f"  top_k={top_k}, rerank_top_n={rerank_n}")

        t0 = time.time()

        records = run_single_config(
            dataset=dataset,
            model=model,
            retriever=retriever,
            kb=kb,
            reranker=reranker,
            top_k=top_k,
            rerank_top_n=rerank_n,
            output_path=pred_path,
            debug_samples=args.debug_samples,
        )

        elapsed = time.time() - t0
        print(f"  Inference done in {elapsed:.1f}s")

        # Quick proxy scores (exact-match heuristic)
        scores = compute_scores_simple(records)
        scores["config"] = {"top_k": top_k, "rerank_top_n": rerank_n}
        scores["elapsed_seconds"] = round(elapsed, 1)

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        all_results[config_name] = scores

        print(f"  Proxy accuracy: {scores['accuracy_overall']:.4f}")
        for qt in sorted(scores["accuracy_by_type"]):
            print(f"    {qt:20s}: {scores['accuracy_by_type'][qt]:.4f}")
        print(f"  Predictions → {pred_path}")
        print(f"  Results     → {result_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}\n")

    # Sort by accuracy
    ranking = sorted(all_results.items(), key=lambda x: x[1]["accuracy_overall"], reverse=True)

    print(f"{'Config':<30s} {'Accuracy':>10s} {'Time (s)':>10s}")
    print("─" * 52)
    for name, res in ranking:
        cfg = res["config"]
        print(
            f"  top_k={cfg['top_k']:<2d} rerank_n={cfg['rerank_top_n']:<2d}"
            f"        {res['accuracy_overall']:>8.4f}"
            f"  {res['elapsed_seconds']:>8.1f}"
        )

    best_name, best_res = ranking[0]
    best_cfg = best_res["config"]
    print(f"\n★ Best config: top_k={best_cfg['top_k']}, rerank_top_n={best_cfg['rerank_top_n']}")
    print(f"  Proxy accuracy: {best_res['accuracy_overall']:.4f}")

    # Save aggregated summary
    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    summary = {
        "grid": {"top_k_values": args.top_k_values, "rerank_top_n_values": args.rerank_top_n_values},
        "best_config": best_cfg,
        "best_accuracy": best_res["accuracy_overall"],
        "results": {name: res for name, res in ranking},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nAggregated summary → {summary_path}")

    print(f"\n{'='*70}")
    print("NOTE: The proxy scores above use simple exact-match heuristics.")
    print("For official BEM scores, run the evaluation script on each predictions file:")
    print(f"  for f in {args.output_dir}/predictions_cross_*.jsonl; do")
    print(f'    uv run python evqa_eval/score_evqa.py --predictions "$f" \\')
    print(f'      --output "${{f/predictions_/results_BEM_}}"')
    print(f"  done")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
