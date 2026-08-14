from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from langchain_openai import ChatOpenAI

from retrieval.retriever import Retriever
from retrieval.knowledge_base import KnowledgeBase
from retrieval.reranker import CrossEncoderReranker
from vlm.qwen_model import load_dataset
from vlm.run_inference import build_record
from agent.rag import AgenticRAG
from agent.schemas import AgentRun
from agent.evaluation.config import EvalConfig
from agent.evaluation.metrics import MetricsCollector


def load_stack(config: EvalConfig, logger: logging.Logger):
    """Load the vLLM chat client + retrieval stack once (reusable across agents)."""
    logger.info("Connecting to %s (model %s)", config.base_url, config.model_name)
    llm = ChatOpenAI(
        model=config.model_name,
        base_url=config.base_url,
        api_key=os.getenv("LLM_API_KEY", "EMPTY"),
        max_tokens=config.max_tokens,
        temperature=0.0,
    )

    logger.info("Loading EVA-CLIP retriever …")
    retriever = Retriever(
        img_index_path=config.img_index_path,
        img_index_json_path=config.img_index_json_path,
        top_k=config.top_k,
        device=config.retriever_device,
        ef_search=config.ef_search,
    )
    retriever._ensure_index()
    retriever._ensure_model()

    logger.info("Loading Knowledge Base …")
    kb = KnowledgeBase(config.kb_path)

    logger.info("Loading Cross-Encoder reranker …")
    reranker = CrossEncoderReranker(config.cross_encoder_model, device=config.retriever_device)

    return llm, retriever, kb, reranker


def build_agent(config: EvalConfig, logger: logging.Logger) -> AgenticRAG:
    """Load the retrieval stack + chat client and wire up the agent."""
    llm, retriever, kb, reranker = load_stack(config, logger)
    return AgenticRAG(
        retriever=retriever,
        kb=kb,
        reranker=reranker,
        llm=llm,
        logger=logger,
        top_n=config.rerank_top_n,
        top_k=config.top_k,
        max_iterations=config.max_iterations,
        force_first_tool=config.force_first_tool,
    )


class AgenticEvaluator:
    """Runs the agent over the (resumable) test set with concurrent requests."""

    def __init__(self, config: EvalConfig, agent: AgenticRAG, logger: logging.Logger):
        self.config = config
        self.agent = agent
        self.metrics = MetricsCollector()
        self._logger = logger

    def run(self) -> dict:
        todo = self._load_todo()
        self._logger.info(
            "Running (concurrency=%d, top_k=%d, rerank_top_n=%d, max_iterations=%d)",
            self.config.concurrency, self.config.top_k,
            self.config.rerank_top_n, self.config.max_iterations,
        )
        debug_left = self.config.debug_samples

        with open(self.config.output, "a", encoding="utf-8") as out:
            with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
                futures = [pool.submit(self._worker, it) for it in todo]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Agentic RAG"):
                    item, run = fut.result()
                    if run is None:
                        self._logger.warning("missing image: %s", item["image_path"])
                        out.write(json.dumps(build_record(item=item, prediction=None),
                                             ensure_ascii=False) + "\n")
                        out.flush()
                        continue
                    out.write(json.dumps(self._record(item, run), ensure_ascii=False) + "\n")
                    out.flush()
                    self.metrics.record(run)
                    if debug_left > 0:
                        self._log_debug(item, run)
                        debug_left -= 1

        summary = self.metrics.save(self.config.metrics_path)
        self._logger.info("Metrics summary: %s", json.dumps(summary))
        self._logger.info("Predictions: %s | Metrics: %s", self.config.output, self.config.metrics_path)
        return summary

    def _done_ids(self) -> set[str]:
        if not os.path.exists(self.config.output):
            return set()
        with open(self.config.output, encoding="utf-8") as f:
            return {json.loads(line)["unique_id"] for line in f if line.strip()}

    def _load_todo(self) -> list[dict]:
        dataset = load_dataset(json_path=self.config.json_path, base_folder=self.config.base_folder)
        if self.config.limit is not None:
            dataset = dataset[: self.config.limit]
        done = self._done_ids()
        todo = [it for it in dataset if it["unique_id"] not in done]
        self._logger.info("Dataset: %d | already done: %d | to do: %d", len(dataset), len(done), len(todo))
        return todo

    def _worker(self, item: dict) -> tuple[dict, AgentRun | None]:
        if not os.path.exists(item["image_path"]):
            return item, None
        return item, self.agent.run(image_path=item["image_path"], question=item["question"])

    def _record(self, item: dict, run: AgentRun) -> dict:
        record = build_record(item=item, prediction=run.prediction)
        record["agent"] = {
            "tool_called": run.tool_called,
            "num_tool_calls": run.num_tool_calls,
            "elapsed_seconds": run.elapsed_seconds,
            "error": run.error,
            "steps": [
                {
                    "order": s.order,
                    "tool": s.tool,
                    "arguments": s.arguments,
                    "miss": s.miss,
                    "num_articles": s.num_articles,
                    "num_paragraphs": s.num_paragraphs,
                }
                for s in run.steps
            ],
        }
        return record

    def _log_debug(self, item: dict, run: AgentRun) -> None:
        steps = " | ".join(f"{s.order}:{s.tool}{s.arguments}" for s in run.steps) or "<no tool>"
        self._logger.info(
            "[%s] %s\n  Q: %s\n  GT: %s\n  steps: %s\n  pred: %s (%.1fs)",
            item["unique_id"], item["question_type"], item["question"],
            item["answer"], steps, run.prediction, run.elapsed_seconds,
        )
