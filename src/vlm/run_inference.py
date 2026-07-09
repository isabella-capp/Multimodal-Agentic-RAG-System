"""Run Qwen2.5-VL inference over Encyclopedic-VQA and save predictions as JSONL.

Supports two modes:

* **Baseline** (default): The VLM answers using only the image + question.
* **With retrieval** (``--use-retrieval``): The user image is first used to
  retrieve the most similar Wikipedia article via FAISS, then the top
  paragraphs from the knowledge base are provided as context to the VLM.

The run is resumable: already-predicted unique_ids are skipped on restart.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image
from tqdm import tqdm
from retrieval.retriever import Retriever
from retrieval.knowledge_base import KnowledgeBase

from vlm.qwen_model import QwenVQAModel, load_dataset

BASE_FOLDER = "/work/cvcs2026/encyclopedic"


def load_done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["unique_id"] for line in f if line.strip()}


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


def build_rag_prompt(question: str, paragraphs: list[str]) -> str:
    """Compose a Multimodal RAG prompt: context paragraphs + question."""
    context = "\n\n".join(paragraphs)
    return (
        f"Answer the question concisely based on the provided image and the following context. "
        f"Strictly use only the information provided in the context or visible in the image.\n\n"
        f"--- CONTEXT ---\n{context}\n\n"
        f"--- QUESTION ---\n{question}\n\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL inference on Encyclopedic-VQA"
    )
    parser.add_argument(
        "--json-path",
        default=f"{BASE_FOLDER}/encyclopedic_test_subset.json",
    )
    parser.add_argument("--base-folder", default=BASE_FOLDER)
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--limit", type=int, default=None)

    # Retrieval options
    parser.add_argument(
        "--use-retrieval",
        action="store_true",
        help="Enable visual retrieval + KB context augmentation.",
    )
    parser.add_argument(
        "--img-index-path",
        default=f"{BASE_FOLDER}/knn.index",
        help="Path to the FAISS index.",
    )
    parser.add_argument(
        "--img-index-json-path",
        default=f"{BASE_FOLDER}/knn.json",
        help="Path to the FAISS index JSON mapping.",
    )
    parser.add_argument(
        "--kb-path",
        default=f"{BASE_FOLDER}/encyclopedic_kb_wiki.json",
        help="Path to the encyclopedic knowledge base JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of FAISS nearest neighbours to retrieve.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=3,
        help="Number of paragraphs to keep after reranking.",
    )
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

    retriever = None
    kb = None
    if args.use_retrieval:
        retriever = Retriever(
            img_index_path=args.img_index_path,
            img_index_json_path=args.img_index_json_path,
            top_k=args.top_k,
        )
        kb = KnowledgeBase(args.kb_path)

    with open(args.output, "a", encoding="utf-8") as out:
        for item in tqdm(dataset, desc="Inference"):
            if item["unique_id"] in done_ids:
                continue

            image_path = item["image_path"]

            if not os.path.exists(image_path):
                tqdm.write(f"missing image: {image_path}")
                out.write(
                    json.dumps(
                        build_record(item, None), ensure_ascii=False
                    )
                    + "\n"
                )
                out.flush()
                continue

            retrieved_context = None
            prompt = item["question"]

            # --- Retrieval-augmented path ---
            if retriever is not None and kb is not None:
                try:
                    user_image = Image.open(image_path).convert("RGB")
                    results = retriever.retrieve(user_image, item["question"])

                    if results:
                        best_url = results[0]["wiki_url"]
                        paragraphs = kb.get_paragraphs_by_url(best_url)

                        if paragraphs:
                            top_paragraphs = retriever.rerank_paragraphs(
                                item["question"],
                                paragraphs,
                                top_n=args.rerank_top_n,
                            )
                            prompt = build_rag_prompt(item["question"], top_paragraphs)
                            retrieved_context = {
                                "wiki_url": best_url,
                                "title": results[0].get("title", ""),
                                "score": results[0].get("score"),
                                "num_paragraphs_total": len(paragraphs),
                                "num_paragraphs_used": len(top_paragraphs),
                            }
                except Exception as e:
                    tqdm.write(f"retrieval failed for {item['unique_id']}: {e}")

            prediction = model.generate_response(image_path, prompt)

            out.write(
                json.dumps(
                    build_record(item, prediction, retrieved_context),
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

    print(f"Done. Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
