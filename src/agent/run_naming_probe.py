import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_ROOT)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from agent.evaluation.config import EvalConfig
from agent.log import setup_logging
from agent.messages import image_to_data_uri
from agent.prompts import NAMING_PROMPT
from retrieval.knowledge_base import KnowledgeBase, normalize
from vlm.dataset import load_dataset


def _norm_url(url: str) -> str:
    return (url or "").rstrip("/").lower().replace("http://", "https://")


def _name_one(llm, item: dict) -> str | None:
    try:
        resp = llm.invoke([
            SystemMessage(content=NAMING_PROMPT),
            HumanMessage(content=[
                {"type": "image_url",
                 "image_url": {"url": image_to_data_uri(item["image_path"])}},
            ]),
        ], max_tokens=32)
        return resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception:
        return None


def main():
    d = EvalConfig()
    p = argparse.ArgumentParser(description="Can the model name the entity in the image?")
    p.add_argument("--model-name", default=d.model_name)
    p.add_argument("--base-url", default=d.base_url)
    p.add_argument("--json-path", default=d.json_path)
    p.add_argument("--base-folder", default=d.base_folder)
    p.add_argument("--kb-path", default=d.kb_path)
    p.add_argument("--output", default="outputs/agentic/naming_probe.jsonl")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    logger = setup_logging()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    kb = KnowledgeBase(args.kb_path)

    dataset = [it for it in load_dataset(json_path=args.json_path,
                                         base_folder=args.base_folder)
               if os.path.exists(it["image_path"])][: args.limit]
    logger.info("Naming %d images with %s", len(dataset), args.model_name)

    llm = ChatOpenAI(model=args.model_name, base_url=args.base_url,
                     api_key=os.getenv("LLM_API_KEY", "EMPTY"), temperature=0.0)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        names = list(tqdm(pool.map(lambda it: _name_one(llm, it), dataset),
                          total=len(dataset), desc="naming"))

    stats = {"exact_name": 0, "resolved": 0, "resolved_exact": 0, "resolved_fuzzy": 0,
             "no_name": 0, "gt_in_universe": 0}
    with open(args.output, "w", encoding="utf-8") as out:
        for item, name in zip(dataset, names):
            gt_url, gt_title = _norm_url(item["wikipedia_url"]), item["wikipedia_title"]
            in_universe = bool(kb.lookup_articles(gt_title, limit=1))
            stats["gt_in_universe"] += in_universe
            if not name:
                stats["no_name"] += 1
                hits, resolved, match = [], False, None
            else:
                hits = kb.lookup_articles(name, limit=5)
                resolved = any(_norm_url(h["wiki_url"]) == gt_url for h in hits)
                match = hits[0]["match"] if hits else None
                stats["exact_name"] += normalize(name) == normalize(gt_title)
                stats["resolved"] += resolved
                if resolved and match:
                    stats[f"resolved_{match}"] += 1
            out.write(json.dumps({
                "unique_id": item.get("unique_id"),
                "gt_title": gt_title, "gt_url": item["wikipedia_url"],
                "predicted_name": name, "match": match, "resolved": resolved,
                "candidates": [h["title"] for h in hits],
            }, ensure_ascii=False) + "\n")

    n = len(dataset)
    logger.info("=" * 62)
    logger.info("model: %s   examples: %d", args.model_name, n)
    logger.info("GT title resolvable by the index      : %5.1f%%", 100 * stats["gt_in_universe"] / n)
    logger.info("predicted name == GT title (verbatim) : %5.1f%%", 100 * stats["exact_name"] / n)
    logger.info("name RESOLVES to the GT article       : %5.1f%%   <- recall of lookup_article",
                100 * stats["resolved"] / n)
    logger.info("    via exact match                   : %5.1f%%", 100 * stats["resolved_exact"] / n)
    logger.info("    via fuzzy match                   : %5.1f%%", 100 * stats["resolved_fuzzy"] / n)
    logger.info("no name produced                      : %5.1f%%", 100 * stats["no_name"] / n)
    logger.info("lookup ceiling w/ perfect naming      :  83.5%%")
    logger.info("EVA-CLIP image recall@50 (reference)  :  46.7%%")
    logger.info("=" * 62)
    logger.info("Per-example detail: %s", args.output)


if __name__ == "__main__":
    main()
