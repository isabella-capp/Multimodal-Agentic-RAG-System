from __future__ import annotations

import time

from typing import Callable, Any
from PIL import Image
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.errors import GraphRecursionError

from agent.messages import build_user_message
from agent.prompts import SYSTEM_PROMPT
from agent.run import AgentRun
from agent.tools import build_tools
from retrieval.bm25 import BM25Ranker

def force_first_tool() -> Any:
    """Forces the LLM to call a tool on its very first conversational turn."""
    @wrap_model_call
    def force_first_middleware(request: Any, handler: Callable) -> Any:
        if not any(isinstance(m, ToolMessage) for m in reversed(request.messages)):
            request = request.override(tool_choice="required")
        return handler(request)

    return force_first_middleware


def remind_original_question(original_question: str) -> Any:
    """Re-injects the original question before the final answer."""
    @wrap_model_call
    def remind_question_middleware(request, handler):
        if request.messages and isinstance(request.messages[-1], ToolMessage):
            
            reminder_text = (
                f"Reminder: Keep your final answer strictly focused on the "
                f"original question: '{original_question}'"
            )
            reminder_msg = SystemMessage(content=reminder_text)
            
            # Aggiungiamo il promemoria alla fine della lista dei messaggi
            request = request.override(messages=request.messages + [reminder_msg])
            
        return handler(request)
        
    return remind_question_middleware

class AgenticRAG:
    """Runs the agentic RAG loop for one example at a time.

    Tools retrieve on demand and grow a per-example working set of articles; the
    model decides whether to enter by name or by image, and which to read.
    """

    def __init__(self, llm, retriever, kb, reranker, top_n=5, top_k=20,
                 bm25_top_m=50, max_iterations=8, force_first=True,
                 retrieval_mode: str = "bm25+reranker", rrf_k: int = 60):
        self.llm = llm
        self.retriever = retriever
        self.kb = kb
        self.reranker = reranker
        self.bm25 = BM25Ranker()
        self.top_n = top_n
        self.top_k = top_k
        self.bm25_top_m = bm25_top_m
        self.max_iterations = max_iterations
        self.force_first = force_first
        self.retrieval_mode = retrieval_mode
        self.rrf_k = rrf_k

    def _middleware(self, question: str) -> list[Any]:
        middlewares = []
        if self.force_first:
            middlewares.append(force_first_tool())
            
        # Aggiungi il promemoria per mantenere il focus
        middlewares.append(remind_original_question(question))
        
        return middlewares

    def run(self, image_path: str, question: str) -> AgentRun:
        t0 = time.time()

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return AgentRun(error=f"Image load error: {str(e)}")
        
        agent = create_agent(
            model=self.llm,
            tools=build_tools(self.retriever, self.kb, self.reranker, self.bm25,
                              image, top_n=self.top_n, top_k=self.top_k,
                              bm25_top_m=self.bm25_top_m,
                              retrieval_mode=self.retrieval_mode,
                              rrf_k=self.rrf_k),
            system_prompt=SYSTEM_PROMPT,
            middleware=self._middleware(question),
        )

        try:
            out = agent.invoke(
                {"messages": [build_user_message(image_path, question)]},
                config={"recursion_limit": 2 * self.max_iterations + 1},
            )
            run = AgentRun.from_messages(out["messages"])
            run.messages = out["messages"]
        except GraphRecursionError:
            run = AgentRun(error="recursion_limit")
        except Exception as e:
            run = AgentRun(error=str(e))

        run.elapsed_seconds = round(time.time() - t0, 2)
        return run
