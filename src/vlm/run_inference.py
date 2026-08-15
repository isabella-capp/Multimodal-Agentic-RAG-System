import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image
from tqdm import tqdm

import paths
from llm import VLMClient
from prompts import NO_RAG_PROMPT, RAG_PROMPT
from retrieval.knowledge_base import KnowledgeBase
from retrieval.retriever import Retriever
from runner import load_todo, run_batch
from vlm.arg_parser import parse_args
from vlm.dataset import build_record


def build_rag_prompt(question, paragraphs):
    return RAG_PROMPT.format(context="\n\n".join(paragraphs), question=question)


def setup_retrieval(top_k, use_cross_reranker):
    retriever = Retriever(
        paths.IMG_INDEX_PATH, paths.IMG_INDEX_JSON_PATH, top_k=top_k,
        device=paths.RETRIEVER_DEVICE, ef_search=paths.EF_SEARCH
    )
    kb = KnowledgeBase(paths.KB_PATH)
    reranker = None
    if use_cross_reranker:
        from retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(paths.CROSS_ENCODER_MODEL, device=paths.RETRIEVER_DEVICE)
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
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model = VLMClient(args.model_name, args.base_url)
    retriever = kb = reranker = None
    if args.use_retrieval:
        retriever, kb, reranker = setup_retrieval(args.top_k, not args.no_rerank)

    shown = []

    def predict(item):
        prompt = NO_RAG_PROMPT.format(question=item["question"])
        paragraphs = retrieved = None
        if retriever is not None:
            try:
                context = build_context(retriever, kb, reranker, item["question"],
                                        item["image_path"], args.rerank_top_n,
                                        args.no_rerank)
                if context is not None:
                    paragraphs, retrieved = context
                    prompt = build_rag_prompt(item["question"], paragraphs)
            except Exception as e:
                tqdm.write(f"retrieval failed for {item['unique_id']}: {e}")

        prediction = model.generate_response(item["image_path"], prompt)
        if len(shown) < args.debug_samples:
            shown.append(item["unique_id"])
            print_debug_example(item, retrieved, paragraphs, prediction)
        return build_record(item, prediction, retrieved)

    run_batch(load_todo(args.output, args.limit), predict, args.output, args.concurrency)
    print(f"Done. Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
