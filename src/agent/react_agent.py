"""Multimodal ReAct Agent — orchestrates visual retrieval + agentic reasoning.

This module ties together the existing retrieval stack (``Retriever``,
``KnowledgeBase``, ``CrossEncoderReranker``) with a LangChain ReAct agent
that uses ``QwenVQAModel`` as its reasoning backbone.

Architecture (Visual-First)
---------------------------
1. **Pre-retrieval (Stage 0)** — executed *once* per query, *outside* the
   agent loop:
   - Encode the user image with EVA-CLIP.
   - Search FAISS for the ``top_k`` nearest articles.
   - Extract *all* paragraphs from those articles via the Knowledge Base.
   → produces a fixed ``paragraph_pool``.

2. **Agent loop (ReAct)** — the LLM can call ``search_paragraphs`` up to
   ``max_iterations`` times.  Each call runs the cross-encoder reranker on
   ``paragraph_pool`` with a *new* textual query chosen by the agent.

3. **Final answer** — the agent generates a concise, grounded response.

No existing files in ``src/retrieval/`` or ``src/vlm/`` are modified.
"""

from __future__ import annotations

import time
from typing import Any

from PIL import Image
from langchain.agents import AgentExecutor, create_react_agent

from agent.qwen_langchain import QwenLangChainLLM
from agent.tools import make_search_tool
from agent.prompts import REACT_PROMPT
from retrieval.retriever import Retriever
from retrieval.knowledge_base import KnowledgeBase
from retrieval.reranker import CrossEncoderReranker


class MultimodalReActAgent:
    """End-to-end multimodal agentic RAG system.

    Parameters
    ----------
    retriever : Retriever
        EVA-CLIP + FAISS retriever (already loaded).
    kb : KnowledgeBase
        Encyclopedic knowledge base (already loaded).
    reranker : CrossEncoderReranker
        Cross-encoder paragraph reranker (already loaded).
    vlm : Any
        ``QwenVQAModel`` instance (already loaded).
    top_k : int
        Number of FAISS nearest neighbours for the visual pre-retrieval.
    top_n : int
        Number of paragraphs the cross-encoder returns per tool call.
    max_iterations : int
        Maximum number of tool invocations the agent is allowed before
        being forced to produce a final answer.
    verbose : bool
        If ``True``, LangChain prints every agent step to stdout.
    """

    def __init__(
        self,
        retriever: Retriever,
        kb: KnowledgeBase,
        reranker: CrossEncoderReranker,
        vlm: Any,
        top_k: int = 20,
        top_n: int = 5,
        max_iterations: int = 3,
        verbose: bool = False,
    ):
        self.retriever = retriever
        self.kb = kb
        self.reranker = reranker
        self.vlm = vlm
        self.top_k = top_k
        self.top_n = top_n
        self.max_iterations = max_iterations
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Pre-retrieval (Stage 0)                                              #
    # ------------------------------------------------------------------ #

    def _pre_retrieve(
        self, image_path: str, question: str
    ) -> tuple[list[dict], list[str]]:
        """Visual retrieval + KB paragraph extraction.

        Returns
        -------
        faiss_results : list[dict]
            The raw FAISS results (wiki_url, title, score, …).
        paragraph_pool : list[str]
            All paragraphs from the top-k articles.
        """
        saved_top_k = self.retriever.top_k
        self.retriever.top_k = self.top_k
        try:
            user_image = Image.open(image_path).convert("RGB")
            faiss_results = self.retriever.retrieve(user_image, question)
        finally:
            self.retriever.top_k = saved_top_k

        paragraph_pool: list[str] = []
        for r in faiss_results:
            paragraph_pool.extend(
                self.kb.get_paragraphs_by_url(r["wiki_url"])
            )

        return faiss_results, paragraph_pool

    # ------------------------------------------------------------------ #
    # Agent execution                                                      #
    # ------------------------------------------------------------------ #

    def run(self, image_path: str, question: str) -> dict[str, Any]:
        """Execute the full agentic pipeline for a single example.

        Parameters
        ----------
        image_path : str
            Absolute path to the user query image.
        question : str
            The user question.

        Returns
        -------
        dict with keys:
            ``prediction`` — the agent's final answer (str or None).
            ``intermediate_steps`` — list of (action, observation) tuples.
            ``faiss_results`` — the pre-retrieval FAISS results.
            ``num_paragraphs_pool`` — size of the paragraph pool.
            ``num_iterations`` — how many tool calls the agent made.
            ``elapsed_seconds`` — wall-clock time for the full pipeline.
        """
        t0 = time.time()

        # Stage 0: visual pre-retrieval
        faiss_results, paragraph_pool = self._pre_retrieve(
            image_path, question
        )

        # Build the LangChain components for this query
        search_tool = make_search_tool(
            self.reranker, paragraph_pool, self.top_n
        )
        llm = QwenLangChainLLM(vlm=self.vlm, image_path=image_path)

        agent = create_react_agent(
            llm=llm,
            tools=[search_tool],
            prompt=REACT_PROMPT,
        )
        executor = AgentExecutor(
            agent=agent,
            tools=[search_tool],
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            verbose=self.verbose,
        )

        # Run the agent
        try:
            result = executor.invoke({"input": question})
            prediction = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])
        except Exception as e:
            prediction = None
            intermediate_steps = []
            print(f"  Agent error: {e}")

        elapsed = time.time() - t0

        return {
            "prediction": prediction,
            "intermediate_steps": intermediate_steps,
            "faiss_results": faiss_results,
            "num_paragraphs_pool": len(paragraph_pool),
            "num_iterations": len(intermediate_steps),
            "elapsed_seconds": round(elapsed, 2),
        }
