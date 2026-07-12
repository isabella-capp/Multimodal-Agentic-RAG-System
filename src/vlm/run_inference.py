"""Run Qwen2.5-VL inference over Encyclopedic-VQA and save predictions as JSONL.

Supports two modes:

* **Baseline** (default): the VLM answers using only the image + question.
* **With retrieval** (``--use-retrieval``): the image retrieves the most similar
  Wikipedia article via FAISS, then the top KB paragraphs are added as context.

The run is resumable: already-predicted unique_ids are skipped on restart.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image
from tqdm import tqdm
from retrieval.retriever import Retriever
from retrieval.knowledge_base import KnowledgeBase
from vlm.arg_parser import parse_args

from vlm.qwen_model import QwenVQAModel, load_dataset

BASE_FOLDER = "/work/cvcs2026/encyclopedic"
JSON_PATH = f"{BASE_FOLDER}/encyclopedic_test_subset.json"
KB_DB_PATH = f"{BASE_FOLDER}/encyclopedic_kb_wiki.db"
IMG_INDEX_PATH = f"{BASE_FOLDER}/knn.index"
IMG_INDEX_JSON_PATH = f"{BASE_FOLDER}/knn.json"

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"
RETRIEVER_DEVICE = "cuda"

SYSTEM_PROMPT = (
    "You are a visual question answering assistant for encyclopedic questions about "
    "the entity shown in the image. Reply with a short, direct answer — a single word, "
    "entity name, or brief phrase — with no explanation and no full sentences."
)


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


def write_record(out, record):
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()


def build_rag_prompt(question, paragraphs):
    context = "\n\n".join(paragraphs)
    return (
        f"{question}\n\n"
        f"The following paragraphs may contain useful information to help answer "
        f"the question correctly:\n\n{context}\n\n"
    )


def setup_retrieval(top_k, use_cross_reranker):
    retriever = Retriever(
        IMG_INDEX_PATH, IMG_INDEX_JSON_PATH, top_k=top_k, device=RETRIEVER_DEVICE
    )
    kb = KnowledgeBase(KB_DB_PATH)
    reranker = None
    if use_cross_reranker:
        from retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(CROSS_ENCODER_MODEL, device=RETRIEVER_DEVICE)
    return retriever, kb, reranker


def build_context(
    retriever, kb, reranker, question, image_path, rerank_top_n, no_rerank
):
    """Retrieve articles for the image, pool and rerank their paragraphs.

    Returns ``(top_paragraphs, retrieved_context)``, or ``None`` when retrieval
    yields no usable paragraphs.
    """
    user_image = Image.open(image_path).convert("RGB")
    results = retriever.retrieve(user_image, question)
    if not results:
        return None

    pooled = []
    for r in results:
        pooled.extend(kb.get_paragraphs_by_url(r["wiki_url"]))
    if not pooled:
        return None

    if no_rerank:
        top_paragraphs = pooled if rerank_top_n <= 0 else pooled[:rerank_top_n]
    else:
        top_paragraphs = reranker.rerank(question, pooled, top_n=rerank_top_n)

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
    return top_paragraphs, retrieved_context


def _truncate(text, n=200):
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

    dataset = load_dataset(JSON_PATH, BASE_FOLDER)
    if args.limit is not None:
        dataset = dataset[: args.limit]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    done_ids = load_done_ids(args.output)
    if done_ids:
        print(f"Skipping {len(done_ids)} already-predicted examples")

    model = QwenVQAModel(model_name=MODEL_NAME)

    retriever = kb = reranker = None
    if args.use_retrieval:
        retriever, kb, reranker = setup_retrieval(args.top_k, not args.no_rerank)

    debug_count = 0

    with open(args.output, "a", encoding="utf-8") as out:
        for item in tqdm(dataset, desc="Inference"):
            if item["unique_id"] in done_ids:
                continue

            image_path = item["image_path"]
            if not os.path.exists(image_path):
                tqdm.write(f"missing image: {image_path}")
                write_record(out, build_record(item, None))
                continue

            retrieved_context = None
            top_paragraphs = None
            prompt = item["question"]

            if retriever is not None:
                try:
                    context = build_context(
                        retriever,
                        kb,
                        reranker,
                        item["question"],
                        image_path,
                        args.rerank_top_n,
                        args.no_rerank,
                    )
                    if context is not None:
                        top_paragraphs, retrieved_context = context
                        prompt = build_rag_prompt(item["question"], top_paragraphs)
                except Exception as e:
                    tqdm.write(f"retrieval failed for {item['unique_id']}: {e}")

            prediction = model.generate_response(
                image_path, prompt, system_prompt=SYSTEM_PROMPT
            )

            if debug_count < args.debug_samples:
                print_debug_example(item, retrieved_context, top_paragraphs, prediction)
                debug_count += 1

            write_record(out, build_record(item, prediction, retrieved_context))

    print(f"Done. Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
