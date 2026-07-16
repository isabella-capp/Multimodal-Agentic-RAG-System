from __future__ import annotations

import logging
import time

from PIL import Image
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.errors import GraphRecursionError

from agent.messages import build_user_message
from agent.prompts import SYSTEM_PROMPT
from agent.schemas import AgentRun, ArticleHit
from agent.tools import make_search_tool


class AgenticRAG:
    """Runs the agentic RAG loop for one example at a time.

    The retrieval stack and the chat model are loaded once and reused; only the
    paragraph pool and the tool bound to it are rebuilt per query.
    """

    def __init__(self, retriever, kb, reranker, llm: BaseChatModel,
                 logger: logging.Logger, top_n: int = 5, max_iterations: int = 3):
        self.retriever = retriever
        self.kb = kb
        self.reranker = reranker
        self.llm = llm
        self.top_n = top_n
        self.max_iterations = max_iterations
        self._logger = logger

    def run(self, image_path: str, question: str) -> AgentRun:
        t0 = time.time()

        image = Image.open(image_path).convert("RGB")
        raw_articles = self.retriever.retrieve(image=image, question=question)
        articles = [
            ArticleHit(wiki_url=a["wiki_url"], title=a.get("title", ""), score=a.get("score"))
            for a in raw_articles
        ]
        pool = [
            p
            for a in raw_articles
            for p in self.kb.get_paragraphs_by_url(wiki_url=a["wiki_url"])
        ]

        tool = make_search_tool(
            reranker=self.reranker, paragraph_pool=pool, top_n=self.top_n
        )
        agent = create_agent(model=self.llm, tools=[tool], system_prompt=SYSTEM_PROMPT)

        try:
            out = agent.invoke(
                {"messages": [build_user_message(image_path=image_path, question=question)]},
                # one iteration ≈ an agent step + a tool step, +1 for the answer
                config={"recursion_limit": 2 * self.max_iterations + 1},
            )
            run = AgentRun.from_messages(
                out["messages"], articles=articles, num_paragraphs_pool=len(pool)
            )
        except GraphRecursionError:
            run = AgentRun(articles=articles, num_paragraphs_pool=len(pool),
                           error="recursion_limit")
            self._logger.warning("Agent hit the iteration limit without a final answer.")
        except Exception as e:  # keep the batch running
            run = AgentRun(articles=articles, num_paragraphs_pool=len(pool), error=str(e))
            self._logger.warning("Agent error: %s", e)

        run.elapsed_seconds = round(time.time() - t0, 2)
        return run
