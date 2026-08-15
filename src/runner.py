from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import paths
from vlm.dataset import build_record, done_ids, load_dataset


def load_todo(output: str, limit: int | None = None) -> list[dict]:
    """Examples still to predict. Output files are appended to, so a killed run
    resumes where it stopped instead of redoing everything."""
    dataset = load_dataset(paths.JSON_PATH, paths.BASE_FOLDER)
    if limit is not None:
        dataset = dataset[:limit]
    done = done_ids(output)
    todo = [it for it in dataset if it["unique_id"] not in done]
    print(f"Dataset: {len(dataset)} | already done: {len(done)} | to do: {len(todo)}")
    return todo


def run_batch(items: list[dict], predict, output: str, concurrency: int = 8) -> None:
    """Run ``predict(item) -> record`` over items, appending JSONL to ``output``.

    Shared by the baselines and the agent: only ``predict`` differs. Records are
    written as they complete, so progress survives a killed job — which also
    means the output order is not the dataset order.
    """
    def work(item):
        if not os.path.exists(item["image_path"]):
            tqdm.write(f"missing image: {item['image_path']}")
            return build_record(item, None)
        return predict(item)

    with open(output, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(work, it) for it in items]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Inference"):
                out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                out.flush()
