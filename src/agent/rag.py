from __future__ import annotations

import logging
import time

from PIL import Image
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

from agent.graph import build_search_graph, make_initial_state
from agent.messages import build_user_message, image_to_data_uri
from agent.prompts import RESEARCH_SYSTEM_PROMPT, XML_SYSTEM_PROMPT
from agent.protocols import KnowledgeBase, Reranker, Retriever
from agent.research import EvidenceExtractor, build_research_tools, gather_candidates
from agent.schemas import AgentRun


def _make_force_first(tool_name: str | None = None):
    """Middleware that forces a tool call on the first model turn only.

    Used exclusively by the **research** pipeline to ensure the first call is
    always ``research`` (the multimodal extractor sub-agent).  Later turns are
    left at ``auto`` so the model can refine or answer freely.
    """
    @wrap_model_call
    def middleware(request, handler):
        if not any(isinstance(m, ToolMessage) for m in request.messages):
            choice = ({"type": "function", "function": {"name": tool_name}}
                      if tool_name else "required")
            request = request.override(tool_choice=choice)
        return handler(request)

    return middleware


class AgenticRAG:
    """Runs the agentic RAG loop for one example at a time.

    Two pipelines are supported:

    ``search`` (default)
        XML-based LangGraph graph — the model outputs ``<search>``/``<answer>``
        tags; no JSON tool-calling required.  The graph handles routing, parse
        errors, and iteration limits explicitly.

    ``research``
        LangChain tool-calling agent with a multimodal ``EvidenceExtractor``
        sub-agent as the first (forced) hop, followed by optional
        ``search_paragraphs`` refine calls.
    """

    def __init__(self, retriever: Retriever, kb: KnowledgeBase, reranker: Reranker,
                 llm: BaseChatModel, logger: logging.Logger,
                 top_n: int = 5, max_iterations: int = 3,
                 force_first_tool: bool = False, pipeline: str = "search",
                 research_pool_articles: int = 50, research_extractor_articles: int = 20):
        self.retriever = retriever
        self.kb = kb
        self.reranker = reranker
        self.llm = llm
        self.top_n = top_n
        self.max_iterations = max_iterations
        self.pipeline = pipeline
        self.research_pool_articles = research_pool_articles
        self.research_extractor_articles = research_extractor_articles
        self._logger = logger
        self._extractor = EvidenceExtractor(llm, kb) if pipeline == "research" else None

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, image_path: str, question: str, capture_messages: bool = False) -> AgentRun:
        t0 = time.time()
        try:
            run = (self._run_research(image_path, question, capture_messages)
                   if self.pipeline == "research"
                   else self._run_search(image_path, question, capture_messages))
        except Exception as e:
            self._logger.warning("Agent error: %s", e)
            run = AgentRun(error=str(e))

        run.elapsed_seconds = round(time.time() - t0, 2)
        return run

    # ── search pipeline (XML + LangGraph) ────────────────────────────────────

    def _run_search(self, image_path: str, question: str,
                    capture_messages: bool) -> AgentRun:
        image = Image.open(image_path).convert("RGB")

        graph = build_search_graph(
            retriever=self.retriever,
            kb=self.kb,
            reranker=self.reranker,
            llm=self.llm,
            image=image,
            top_n=self.top_n,
            max_iterations=self.max_iterations,
        )
        initial = make_initial_state(
            system_prompt=XML_SYSTEM_PROMPT,
            image_path=image_path,
            question=question,
        )

        try:
            # Each node counts as 1 step toward the recursion limit.
            # Worst case: max_iterations searches + error retries + overhead.
            recursion_limit = 3 * self.max_iterations + 15
            result = graph.invoke(initial, config={"recursion_limit": recursion_limit})
        except GraphRecursionError:
            self._logger.warning("Search graph hit the recursion limit.")
            return AgentRun(error="recursion_limit")

        run = AgentRun.from_graph_state(result)
        if capture_messages:
            run.raw = result.get("messages", [])
        return run

    # ── research pipeline (tool-calling) ─────────────────────────────────────

    def _run_research(self, image_path: str, question: str,
                      capture_messages: bool) -> AgentRun:
        if self._extractor is None:
            raise RuntimeError("EvidenceExtractor not initialized for research pipeline.")
        
        image = Image.open(image_path).convert("RGB")
        articles = gather_candidates(self.retriever, image, self.research_pool_articles)
        data_uri = image_to_data_uri(image_path)
        tools = build_research_tools(
            self._extractor, self.reranker, self.kb, data_uri,
            articles, extractor_articles=self.research_extractor_articles,
            top_n=self.top_n,
        )
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            middleware=[_make_force_first("research")],
        )

        try:
            out = agent.invoke(
                {"messages": [build_user_message(image_path=image_path, question=question)]},
                config={"recursion_limit": 2 * self.max_iterations + 1},
            )
            run = AgentRun.from_messages(out["messages"])
            if capture_messages:
                run.raw = out["messages"]
        except GraphRecursionError:
            self._logger.warning("Research agent hit the iteration limit.")
            run = AgentRun(error="recursion_limit")

        return run