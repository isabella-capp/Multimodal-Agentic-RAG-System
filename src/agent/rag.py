from __future__ import annotations

import logging
import time

from PIL import Image
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

from agent.messages import build_user_message, image_to_data_uri
from agent.prompts import RESEARCH_SYSTEM_PROMPT, SYSTEM_PROMPT
from agent.protocols import KnowledgeBase, Reranker, Retriever
from agent.research import EvidenceExtractor, build_research_tools, gather_candidates
from agent.schemas import AgentRun
from agent.tools import build_tools


def _make_force_first(tool_name: str | None = None):
    """Middleware that forces a tool call on the first model turn, then lets the
    model decide. ``tool_name`` forces that specific tool (e.g. ``research``);
    otherwise any tool is required. Later turns are left untouched (auto), so the
    model can refine or produce the final answer.
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

    The retrieval stack and chat model are loaded once and reused. Tools retrieve
    on demand (no pre-computed pool); the model decides whether/what to search.
    With ``force_first_tool`` the first retrieval is mandatory (still agentic on
    which tool/query and on any follow-up steps).
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
        # The research pipeline always forces the first (research) call.
        self.pipeline = pipeline
        self.force_first_tool = force_first_tool or pipeline == "research"
        self.research_pool_articles = research_pool_articles
        self.research_extractor_articles = research_extractor_articles
        self._logger = logger
        self._extractor = EvidenceExtractor(llm, kb) if pipeline == "research" else None

    def run(self, image_path: str, question: str, capture_messages: bool = False) -> AgentRun:
        t0 = time.time()
        image = Image.open(image_path).convert("RGB")
        if self.pipeline == "research":
            articles = gather_candidates(self.retriever, image, self.research_pool_articles)
            data_uri = image_to_data_uri(image_path)
            tools = build_research_tools(self._extractor, self.reranker, self.kb, data_uri,
                                         articles, extractor_articles=self.research_extractor_articles,
                                         top_n=self.top_n)
            system_prompt = RESEARCH_SYSTEM_PROMPT
            middleware = [_make_force_first("research")]
        else:
            tools = build_tools(self.retriever, self.kb, self.reranker, image, top_n=self.top_n)
            system_prompt = SYSTEM_PROMPT
            middleware = [_make_force_first()] if self.force_first_tool else []
        agent = create_agent(model=self.llm, tools=tools, system_prompt=system_prompt,
                             middleware=middleware)

        try:
            out = agent.invoke(
                {"messages": [build_user_message(image_path=image_path, question=question)]},
                # one iteration ≈ an agent step + a tool step, +1 for the answer
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
