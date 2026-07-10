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
        help="Number of paragraphs to keep (after reranking, or the first N if --no-rerank). With --no-rerank, <=0 means keep all pooled paragraphs.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip paragraph reranking; use the first --rerank-top-n paragraphs directly.",
    )
    parser.add_argument(
        "--retriever-device",
        default="cuda",
        help="Device for the EVA-CLIP retriever (e.g. 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--reranker",
        choices=["clip", "cross"],
        default="clip",
        help="Paragraph reranker: 'clip' (EVA-CLIP bi-encoder) or 'cross' (cross-encoder).",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default="BAAI/bge-reranker-base",
        help="Cross-encoder model to use when --reranker cross.",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=3,
        help="Print a detailed pipeline trace for the first N processed examples.",
    )
    return parser.parse_args()


def _truncate(text: str, n: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " …"


def print_debug_example(item, retrieved_context, top_paragraphs, prediction):
    tqdm.write("\n" + "=" * 70)
    tqdm.write(f"[DEBUG] {item['unique_id']}  ({item['question_type']})")
    tqdm.write(f"  Q : {item['question']}")
    tqdm.write(f"  GT: {item['answer']}")
    if retrieved_context is not None:
        tqdm.write(
            f"  Retrieved: {retrieved_context['title']!r} "
            f"(score={retrieved_context['score']}, "
            f"paragraphs {retrieved_context['num_paragraphs_used']}/"
            f"{retrieved_context['num_paragraphs_total']})"
        )
        tqdm.write(f"             {retrieved_context['wiki_url']}")
        for i, p in enumerate(top_paragraphs or [], 1):
            tqdm.write(f"    [{i}] {_truncate(p)}")
    else:
        tqdm.write("  Retrieved: <none> (baseline prompt, question only)")
    tqdm.write(f"  Prediction: {prediction}")
    tqdm.write("=" * 70)


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
            device=args.retriever_device,
        )
        kb = KnowledgeBase(args.kb_path)

    reranker = None
    if args.use_retrieval and not args.no_rerank and args.reranker == "cross":
        from retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            args.cross_encoder_model, device=args.retriever_device
        )

    debug_count = 0

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
            top_paragraphs = None
            prompt = item["question"]

            # --- Retrieval-augmented path ---
            if retriever is not None and kb is not None:
                try:
                    user_image = Image.open(image_path).convert("RGB")
                    results = retriever.retrieve(user_image, item["question"])

                    if results:
                        # Pool paragraphs from all top-k retrieved articles.
                        pooled = []
                        for r in results:
                            pooled.extend(kb.get_paragraphs_by_url(r["wiki_url"]))

                        if pooled:
                            if args.no_rerank:
                                top_paragraphs = (
                                    pooled
                                    if args.rerank_top_n <= 0
                                    else pooled[: args.rerank_top_n]
                                )
                            elif reranker is not None:
                                top_paragraphs = reranker.rerank(
                                    item["question"],
                                    pooled,
                                    top_n=args.rerank_top_n,
                                )
                            else:
                                top_paragraphs = retriever.rerank_paragraphs(
                                    item["question"],
                                    pooled,
                                    top_n=args.rerank_top_n,
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
                                "num_paragraphs_used": len(top_paragraphs),
                            }
                except Exception as e:
                    tqdm.write(f"retrieval failed for {item['unique_id']}: {e}")

            prediction = model.generate_response(image_path, prompt)

            if debug_count < args.debug_samples:
                print_debug_example(item, retrieved_context, top_paragraphs, prediction)
                debug_count += 1

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
