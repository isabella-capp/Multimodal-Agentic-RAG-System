from __future__ import annotations

import argparse

from pydantic import BaseModel

BASE_FOLDER = "/work/cvcs2026/encyclopedic"


class EvalConfig(BaseModel):
    """All knobs for one evaluation run (built from the CLI or programmatically)."""

    # Data / output
    json_path: str = f"{BASE_FOLDER}/encyclopedic_test_subset.json"
    base_folder: str = BASE_FOLDER
    output: str = "outputs/predictions_agentic.jsonl"

    # Model (served by vLLM)
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    vllm_base_url: str = "http://localhost:8000/v1"
    max_tokens: int = 512
    concurrency: int = 8

    # Retriever
    img_index_path: str = f"{BASE_FOLDER}/knn.index"
    img_index_json_path: str = f"{BASE_FOLDER}/knn.json"
    kb_path: str = f"{BASE_FOLDER}/encyclopedic_kb_wiki.db"
    retriever_device: str = "cuda"
    cross_encoder_model: str = "BAAI/bge-reranker-base"

    # Agent
    top_k: int = 20
    rerank_top_n: int = 5
    max_iterations: int = 3

    # Debug / limits
    debug_samples: int = 3
    limit: int | None = None
    verbose: bool = False

    @property
    def metrics_path(self) -> str:
        return self.output.rsplit(".", 1)[0] + ".metrics.json"

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "EvalConfig":
        d = cls()
        p = argparse.ArgumentParser(description="Agentic RAG evaluation on Encyclopedic-VQA")
        p.add_argument("--json-path", default=d.json_path)
        p.add_argument("--base-folder", default=d.base_folder)
        p.add_argument("--output", default=d.output)
        p.add_argument("--model-name", default=d.model_name)
        p.add_argument("--vllm-base-url", default=d.vllm_base_url)
        p.add_argument("--max-tokens", type=int, default=d.max_tokens)
        p.add_argument("--concurrency", type=int, default=d.concurrency)
        p.add_argument("--img-index-path", default=d.img_index_path)
        p.add_argument("--img-index-json-path", default=d.img_index_json_path)
        p.add_argument("--kb-path", default=d.kb_path)
        p.add_argument("--retriever-device", default=d.retriever_device)
        p.add_argument("--cross-encoder-model", default=d.cross_encoder_model)
        p.add_argument("--top-k", type=int, default=d.top_k)
        p.add_argument("--rerank-top-n", type=int, default=d.rerank_top_n)
        p.add_argument("--max-iterations", type=int, default=d.max_iterations)
        p.add_argument("--debug-samples", type=int, default=d.debug_samples)
        p.add_argument("--limit", type=int, default=d.limit)
        p.add_argument("--verbose", action="store_true")
        return cls(**vars(p.parse_args(argv)))
