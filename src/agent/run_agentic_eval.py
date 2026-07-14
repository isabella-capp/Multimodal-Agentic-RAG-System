"""Batch evaluation for the Multimodal Agentic RAG pipeline.

Loads all heavy models **once** (Qwen VLM, EVA-CLIP retriever, Knowledge
Base, Cross-Encoder reranker), then runs the ``MultimodalReActAgent`` over
every example in the dataset.  Saves predictions in the same JSONL format
used by the baselines so that ``evqa_eval/score_evqa.py`` can score them
directly.
"""

import argparse
import json
import os
import sys

# Ensure src/ is on sys.path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_ROOT)

from collections import defaultdict
from tqdm import tqdm

from retrieval.retriever import Retriever
from retrieval.knowledge_base import KnowledgeBase
from retrieval.reranker import CrossEncoderReranker
from vlm.qwen_model import QwenVQAModel, load_dataset
from vlm.run_inference import build_record
from agent.react_agent import MultimodalReActAgent

BASE_FOLDER = "/work/cvcs2026/encyclopedic"


def parse_args():
    p = argparse.ArgumentParser(
        description="Agentic RAG evaluation on Encyclopedic-VQA"
    )

    # Paths
    p.add_argument(
        "--json-path",
        default=f"{BASE_FOLDER}/encyclopedic_test_subset.json",
        help="Path to the test/validation JSON.",
    )
    p.add_argument("--base-folder", default=BASE_FOLDER)
    p.add_argument(
        "--output",
        default="outputs/predictions_agentic.jsonl",
        help="Output JSONL path for predictions.",
    )

    # Model
    p.add_argument(
        "--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct"
    )

    # Retriever
    p.add_argument(
        "--img-index-path", default=f"{BASE_FOLDER}/knn.index"
    )
    p.add_argument(
        "--img-index-json-path", default=f"{BASE_FOLDER}/knn.json"
    )
    p.add_argument(
        "--kb-path", default=f"{BASE_FOLDER}/encyclopedic_kb_wiki.db"
    )
    p.add_argument("--retriever-device", default="cuda")

    # Cross-encoder
    p.add_argument(
        "--cross-encoder-model", default="BAAI/bge-reranker-base"
    )

    # Agent hyper-parameters
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of FAISS neighbours for pre-retrieval.",
    )
    p.add_argument(
        "--rerank-top-n",
        type=int,
        default=5,
        help="Paragraphs returned per tool call (cross-encoder top-n).",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max tool invocations per question.",
    )

    # Debug / limits
    p.add_argument(
        "--debug-samples",
        type=int,
        default=3,
        help="Print detailed trace for first N examples.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples (useful for testing).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose LangChain agent logging.",
    )

    return p.parse_args()

#-----------------------------------------------
# Helpers
#-----------------------------------------------

def load_done_ids(path):
    """Return set of unique_ids already present in the output file."""
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["unique_id"] for line in f if line.strip()}

def write_record(out, record):
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()

#----------------------------------------------
# Debug printing
#----------------------------------------------

def print_debug_example(item, result, prediction):
    tqdm.write(f"\n{'=' * 70}")
    tqdm.write(
        f"[DEBUG] {item['unique_id']}  ({item['question_type']})"
    )
    tqdm.write(f"Q : {item['question']}")
    tqdm.write(f"GT: {item['answer']}")
    tqdm.write(
        f"Agent iterations: {result['num_iterations']}"
    )
    tqdm.write(
        f"Paragraph pool: {result['num_paragraphs_pool']}"
    )
    for i, (action, obs) in enumerate(result["intermediate_steps"], 1): 
        tqdm.write(
            f"Step {i}: Action={action.tool}, "
            f"Input={action.tool_input}"
        )
        tqdm.write(
            f"Obs={str(obs)}"
        )
    tqdm.write(f"Prediction: {prediction}")
    tqdm.write(
        f"  Time: {result['elapsed_seconds']:.1f}s"
    )
    tqdm.write("=" * 70)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'=' * 70}")
    print("Loading dataset …")
    dataset = load_dataset(args.json_path, args.base_folder)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    print(f"Dataset: {len(dataset)} examples")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done_ids = load_done_ids(args.output)
    if done_ids:
        print(f"Skipping {len(done_ids)} already-predicted examples")

    print(f"\n{'=' * 70}")
    print("Loading Qwen VLM …")
    vlm = QwenVQAModel(model_name=args.model_name)

    print(f"\n{'=' * 70}")
    print("Loading EVA-CLIP retriever …")
    retriever = Retriever(
        img_index_path=args.img_index_path,
        img_index_json_path=args.img_index_json_path,
        top_k=args.top_k,
        device=args.retriever_device,
    )
    # Eager-load so any issues surface immediately
    retriever._ensure_index()
    retriever._ensure_model()

    print(f"\n{'=' * 70}")
    print("Loading Knowledge Base …")
    kb = KnowledgeBase(args.kb_path)

    print(f"\n{'=' * 70}")
    print("Loading Cross-Encoder reranker …")
    reranker = CrossEncoderReranker(
        args.cross_encoder_model, device=args.retriever_device
    )

    # ── Build agent ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(
        f"Initialising Agentic RAG (top_k={args.top_k}, "
        f"rerank_top_n={args.rerank_top_n}, "
        f"max_iterations={args.max_iterations})"
    )
    agent = MultimodalReActAgent(
        retriever=retriever,
        kb=kb,
        reranker=reranker,
        vlm=vlm,
        top_k=args.top_k,
        top_n=args.rerank_top_n,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )

    # ── Inference loop ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Running agentic inference …")

    debug_count = 0

    with open(args.output, "a", encoding="utf-8") as out:
        for item in tqdm(dataset, desc="Agentic RAG inference"):
            if item["unique_id"] in done_ids:
                continue

            image_path = item["image_path"]

            if not os.path.exists(image_path):
                tqdm.write(f"missing image: {image_path}")
                write_record(out, build_record(item, None))
                continue

            # Run the agent
            result = agent.run(image_path, item["question"])
            prediction = result["prediction"]

            actual_paragraphs_used = sum(
                str(obs).count("[Paragraph ") for _, obs in result.get("intermediate_steps", [])
            )

            # Build retrieved_context metadata (compatible with baseline format)
            retrieved_context = None
            if result["faiss_results"]:
                retrieved_context = {
                    "wiki_url": result["faiss_results"][0]["wiki_url"],
                    "title": result["faiss_results"][0].get("title", ""),
                    "score": result["faiss_results"][0].get("score"),
                    "candidates": [
                        {
                            "wiki_url": r["wiki_url"],
                            "title": r.get("title", ""),
                            "score": r.get("score"),
                        }
                        for r in result["faiss_results"]
                    ],
                    "num_paragraphs_total": result["num_paragraphs_pool"],
                    "num_paragraphs_used": actual_paragraphs_used,
                    "agent_iterations": result["num_iterations"],
                    "agent_elapsed_seconds": result["elapsed_seconds"],
                }

            # Debug output
            if debug_count < args.debug_samples:
                print_debug_example(item, result, prediction)
                debug_count += 1

            # Save in baseline-compatible format
            record = build_record(item, prediction, retrieved_context)
            write_record(out, record)

    print(f"\nDone. Predictions saved to {args.output}")
    print(
        f"Run evaluation with:\n"
        f"  uv run python evqa_eval/score_evqa.py "
        f"--predictions {args.output}"
    )


if __name__ == "__main__":
    main()
