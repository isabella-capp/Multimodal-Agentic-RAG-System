from __future__ import annotations

from pydantic import BaseModel, Field


class ArticleHit(BaseModel):
    """One article returned by the visual retriever (FAISS neighbour)."""

    wiki_url: str
    title: str = ""
    score: float | None = None


class RetrievedContext(BaseModel):
    """Baseline-compatible retrieval metadata stored with each prediction.

    Serialised (``model_dump``) into the ``retrieved_context`` field of the
    output JSONL so ``evqa_eval/score_evqa.py`` and the notebooks can read it.
    """

    wiki_url: str
    title: str = ""
    score: float | None = None
    candidates: list[ArticleHit] = Field(default_factory=list)
    num_paragraphs_total: int = 0
    num_paragraphs_used: int = 0
    agent_tool_calls: int = 0
    agent_elapsed_seconds: float = 0.0
