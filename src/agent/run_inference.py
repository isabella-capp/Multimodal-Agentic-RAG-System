import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, SRC_ROOT)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tqdm import tqdm

import paths
from agent.metrics import summarise
from agent.rag import AgenticRAG
from llm import chat_model
from retrieval.knowledge_base import KnowledgeBase
from retrieval.reranker import CrossEncoderReranker
from retrieval.retriever import Retriever
from runner import load_todo, run_batch
from vlm.dataset import build_record

MAX_TOKENS = 512


def parse_args():
    p = argparse.ArgumentParser(description="Agentic RAG evaluation on Encyclopedic-VQA")
    p.add_argument("--output", default="outputs/predictions_agentic.jsonl")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rerank-top-n", type=int, default=20)
    p.add_argument("--max-iterations", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--debug-samples", type=int, default=3)
    p.add_argument("--no-force-first-tool", dest="force_first", action="store_false")
    return p.parse_args()


def build_agent(args):
    llm = chat_model(args.model_name, args.base_url, MAX_TOKENS)
    print(f"Agent model: {args.model_name} @ {args.base_url}")

    retriever = Retriever(paths.IMG_INDEX_PATH, paths.IMG_INDEX_JSON_PATH,
                          top_k=args.top_k, device=paths.RETRIEVER_DEVICE,
                          ef_search=paths.EF_SEARCH)
    retriever._ensure_index()
    retriever._ensure_model()
    kb = KnowledgeBase(paths.KB_PATH)
    reranker = CrossEncoderReranker(paths.CROSS_ENCODER_MODEL, device=paths.RETRIEVER_DEVICE)

    return AgenticRAG(llm, retriever, kb, reranker, top_n=args.rerank_top_n,
                      top_k=args.top_k, max_iterations=args.max_iterations,
                      force_first=args.force_first)


def format_trace(messages):
    """The full loop for one example: question → tool calls → results → answer."""
    def text(content):
        if isinstance(content, str):
            return content
        parts = []
        for block in content or []:
            if isinstance(block, dict):
                parts.append(block["text"] if block.get("type") == "text" else "<image>")
            else:
                parts.append(str(block))
        return " ".join(parts)

    lines = ["=" * 78]
    for m in messages:
        if isinstance(m, HumanMessage):
            lines += ["[USER]", text(m.content).strip(), ""]
        elif isinstance(m, AIMessage):
            if m.tool_calls:
                lines.append("[ASSISTANT → tool call]")
                lines += [f"  {tc['name']}(args={tc.get('args', {})})" for tc in m.tool_calls]
            else:
                lines += ["[ASSISTANT → final answer]", f"  {text(m.content).strip()}"]
            lines.append("")
        elif isinstance(m, ToolMessage):
            lines += [f"[TOOL RESULT] ({m.name})", text(m.content).strip(), ""]
    return "\n".join(lines + ["=" * 78])


def print_debug_example(item, run):
    steps = " | ".join(f"{s.order}:{s.tool}{s.arguments}" for s in run.steps) or "<no tool>"
    tqdm.write("\n" + "=" * 70)
    tqdm.write(f"[DEBUG] {item['unique_id']}  ({item['question_type']})")
    tqdm.write(f"  Q : {item['question']}")
    tqdm.write(f"  GT: {item['answer']}")
    tqdm.write(f"  steps: {steps}")
    tqdm.write(f"  Prediction: {run.prediction}  ({run.elapsed_seconds}s)")
    tqdm.write("=" * 70)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    agent = build_agent(args)

    runs, shown, traced = [], [], []

    def predict(item):
        run = agent.run(item["image_path"], item["question"])
        runs.append(run)

        if len(shown) < args.debug_samples:
            shown.append(item["unique_id"])
            print_debug_example(item, run)
        if not traced and run.tool_called and run.messages:
            traced.append(item["unique_id"])
            tqdm.write(format_trace(run.messages))

        record = build_record(item, run.prediction)
        record["agent"] = {
            "tool_called": run.tool_called,
            "num_tool_calls": len(run.steps),
            "elapsed_seconds": run.elapsed_seconds,
            "error": run.error,
            "steps": [s.as_record() for s in run.steps],
        }
        return record

    t0 = time.time()
    run_batch(load_todo(args.output, args.limit), predict, args.output, args.concurrency)

    metrics_path = args.output.rsplit(".", 1)[0] + ".metrics.json"
    metrics = summarise(runs, time.time() - t0)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics: {json.dumps(metrics)}")
    print(f"Predictions: {args.output} | Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
