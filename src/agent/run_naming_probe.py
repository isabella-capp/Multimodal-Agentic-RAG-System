import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_ROOT)

from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

import paths
from agent.messages import image_to_data_uri
from llm import chat_model
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
        ])
        return resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="Can the model name the entity in the image?")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--output", default="outputs/agentic/naming_probe.jsonl")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kb = KnowledgeBase(paths.KB_PATH)

    dataset = [it for it in load_dataset(json_path=paths.JSON_PATH,
                                         base_folder=paths.BASE_FOLDER)
               if os.path.exists(it["image_path"])][: args.limit]
    print(f"Naming {len(dataset)} images with {args.model_name}")

    llm = chat_model(args.model_name, args.base_url, max_tokens=32)

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
    print("=" * 62)
    print(f"model: {args.model_name}   examples: {n}")
    print(f"GT title resolvable by the index      : {100 * stats['gt_in_universe'] / n:5.1f}%")
    print(f"predicted name == GT title (verbatim) : {100 * stats['exact_name'] / n:5.1f}%")
    print(f"name RESOLVES to the GT article       : {100 * stats['resolved'] / n:5.1f}%   <- recall of lookup_article")
    print(f"    via exact match                   : {100 * stats['resolved_exact'] / n:5.1f}%")
    print(f"    via fuzzy match                   : {100 * stats['resolved_fuzzy'] / n:5.1f}%")
    print(f"no name produced                      : {100 * stats['no_name'] / n:5.1f}%")
    print("lookup ceiling w/ perfect naming      :  83.5%")
    print("EVA-CLIP image recall@50 (reference)  :  46.7%")
    print("=" * 62)
    print(f"Per-example detail: {args.output}")


if __name__ == "__main__":
    main()
