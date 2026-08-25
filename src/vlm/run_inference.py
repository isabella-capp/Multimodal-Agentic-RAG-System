import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

import paths
from agent.messages import image_to_data_uri
from agent.prompts import NAMING_PROMPT
from llm import VLMClient
from prompts import (NO_RAG_PROMPT, NO_RAG_PROMPT_LEGACY, RAG_PROMPT,
                     RAG_PROMPT_LEGACY)
from retrieval.knowledge_base import KnowledgeBase
from retrieval.retriever import Retriever
from runner import load_todo, run_batch
from vlm.arg_parser import parse_args
from vlm.dataset import build_record


def build_rag_prompt(question, paragraphs, legacy=False):
    template = RAG_PROMPT_LEGACY if legacy else RAG_PROMPT
    return template.format(context="\n\n".join(paragraphs), question=question)


def setup_retrieval(top_k, use_cross_reranker):
    retriever = Retriever(
        paths.IMG_INDEX_PATH, paths.IMG_INDEX_JSON_PATH, top_k=top_k,
        device=paths.RETRIEVER_DEVICE, ef_search=paths.EF_SEARCH
    )
    retriever._ensure_index()
    retriever._ensure_model()
    kb = KnowledgeBase(paths.KB_PATH)
    reranker = None
    if use_cross_reranker:
        from retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(paths.CROSS_ENCODER_MODEL, device=paths.RETRIEVER_DEVICE)
    return retriever, kb, reranker


def name_entity(model, image_path):
    """What the model thinks the image shows, as a bare Wikipedia-style name.

    Image only, no question: the name is a retrieval key, and letting the
    question leak in makes the model answer instead of naming.
    """
    resp = model.llm.invoke([
        SystemMessage(content=NAMING_PROMPT),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
        ]),
    ])
    return (resp.content if isinstance(resp.content, str) else str(resp.content)).strip()


def name_articles(kb, name, limit):
    """Articles the predicted name resolves to, as retrieval results."""
    if not name:
        return []
    return [{"wiki_url": h["wiki_url"], "title": h["title"], "score": None,
             "source": "name", "match": h["match"]}
            for h in kb.lookup_articles(name, limit=limit)]


def build_context(
    retriever, kb, reranker, question, image_path, rerank_top_n, no_rerank,
    extra_articles=()
):
    """Retrieve articles for the image, pool and rerank their paragraphs.

    ``extra_articles`` are prepended to what the image index returns, so a second
    entry point into the KB widens the pool instead of replacing it. The image
    index only reaches 40.6% recall@20 because it can only rank articles that
    have a reference image; a name resolves against the whole 2.0M-article KB.

    Returns ``(top_paragraphs, retrieved_context)``, or ``None`` when retrieval
    yields no usable paragraphs.
    """
    user_image = Image.open(image_path).convert("RGB")
    results = retriever.retrieve(user_image, question)

    seen = {r["wiki_url"] for r in results}
    results = [a for a in extra_articles if a["wiki_url"] not in seen] + results
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
                "source": r.get("source", "image"),
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
        if retrieved_context.get("predicted_name") is not None:
            named = [c["title"] for c in retrieved_context["candidates"]
                     if c.get("source") == "name"]
            tqdm.write(f"  Named: {retrieved_context['predicted_name']!r} -> {named}")
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
    if args.use_naming and not args.use_retrieval:
        raise SystemExit("--use-naming widens the retrieved pool; it needs --use-retrieval")

    model = VLMClient(args.model_name, args.base_url)
    retriever = kb = reranker = None
    if args.use_retrieval:
        retriever, kb, reranker = setup_retrieval(args.top_k, not args.no_rerank)

    shown = []

    def predict(item):
        base = NO_RAG_PROMPT_LEGACY if args.legacy_prompt else NO_RAG_PROMPT
        prompt = base.format(question=item["question"])
        paragraphs = retrieved = None
        if retriever is not None:
            try:
                name, extra = None, []
                if args.use_naming:
                    name = name_entity(model, item["image_path"])
                    extra = name_articles(kb, name, args.naming_limit)
                context = build_context(retriever, kb, reranker, item["question"],
                                        item["image_path"], args.rerank_top_n,
                                        args.no_rerank, extra)
                if context is not None:
                    paragraphs, retrieved = context
                    if args.use_naming:
                        retrieved["predicted_name"] = name
                    prompt = build_rag_prompt(item["question"], paragraphs,
                                              args.legacy_prompt)
            except Exception as e:
                tqdm.write(f"retrieval failed for {item['unique_id']}: {e}")

        prediction = model.generate_response(item["image_path"], prompt)
        if len(shown) < args.debug_samples:
            shown.append(item["unique_id"])
            print_debug_example(item, retrieved, paragraphs, prediction)
        return build_record(item, prediction, retrieved)

    run_batch(load_todo(args.output, args.limit), predict, args.output,
              args.concurrency, setting="B" if args.use_retrieval else "A",
              model=args.model_name, top_k=args.top_k, rerank_top_n=args.rerank_top_n,
              use_naming=args.use_naming, naming_limit=args.naming_limit,
              reranker=paths.CROSS_ENCODER_MODEL,
              legacy_prompt=args.legacy_prompt)
    print(f"Done. Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
