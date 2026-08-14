from __future__ import annotations

import logging
import time

from PIL import Image
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

from agent.messages import build_user_message
from agent.prompts import SYSTEM_PROMPT
from agent.protocols import KnowledgeBase, Reranker, Retriever
from agent.schemas import AgentRun
from agent.tools import build_tools


def _make_force_first():
    """Middleware that requires a tool call on the first model turn, then lets the
    model decide. Later turns are left on auto, so it can refine or answer.
    """
    @wrap_model_call
    def middleware(request, handler):
        if not any(isinstance(m, ToolMessage) for m in request.messages):
            request = request.override(tool_choice="required")
        return handler(request)

    return middleware


class AgenticRAG:
    """Runs the agentic RAG loop for one example at a time.

    The retrieval stack and chat model are loaded once and reused. Tools retrieve
    on demand and grow a per-example working set of articles; the model decides
    whether to enter by name or by image, and which articles to read.
    """

    def __init__(self, retriever: Retriever, kb: KnowledgeBase, reranker: Reranker,
                 llm: BaseChatModel, logger: logging.Logger,
                 top_n: int = 20, top_k: int = 20, max_iterations: int = 8,
                 force_first_tool: bool = True):
        self.retriever = retriever
        self.kb = kb
        self.reranker = reranker
        self.llm = llm
        self.top_n = top_n
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.force_first_tool = force_first_tool
        self._logger = logger

    def run(self, image_path: str, question: str, capture_messages: bool = False) -> AgentRun:
        t0 = time.time()
        image = Image.open(image_path).convert("RGB")
        tools = build_tools(self.retriever, self.kb, self.reranker, image,
                            top_n=self.top_n, top_k=self.top_k)
        agent = create_agent(model=self.llm, tools=tools, system_prompt=SYSTEM_PROMPT,
                             middleware=[_make_force_first()] if self.force_first_tool else [])

        try:
            out = agent.invoke(
                {"messages": [build_user_message(image_path=image_path, question=question)]},
                config={"recursion_limit": 2 * self.max_iterations + 1},
            )
            run = AgentRun.from_messages(out["messages"])
            if capture_messages:
                run.raw = out["messages"]
        except GraphRecursionError:
            self._logger.warning("Agent hit the iteration limit without a final answer.")
            run = AgentRun(error="recursion_limit")
        except Exception as e:  # keep the batch running
            self._logger.warning("Agent error: %s", e)
            run = AgentRun(error=str(e))

        run.elapsed_seconds = round(time.time() - t0, 2)
        return run
