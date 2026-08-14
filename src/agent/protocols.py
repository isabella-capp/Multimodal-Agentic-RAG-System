from __future__ import annotations

from typing import Protocol

from PIL import Image


class Retriever(Protocol):
    """Visual + text retrieval over the article index."""

    def retrieve(self, image: Image.Image, question: str | None = None) -> list[dict]:
        """Retrieve candidate articles matching the image (visual channel).

        Each dict has ``wiki_url``, ``title``, ``image_path``, ``score``.
        """
        ...

    def retrieve_by_text(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve candidate articles matching a text query (text channel)."""
        ...


class KnowledgeBase(Protocol):
    """Read-only access to articles, their paragraphs, and name lookup."""

    def get_paragraphs_by_url(self, wiki_url: str) -> list[str]:
        """Return the section paragraphs for a Wikipedia URL, in order."""
        ...

    def lookup_articles(self, name: str, limit: int = 5) -> list[dict]:
        """Articles whose title matches ``name`` (exact/alias, then fuzzy)."""
        ...


class Reranker(Protocol):
    """Cross-encoder paragraph reranker."""

    def rerank(self, query: str, paragraphs: list[str], top_n: int = 5) -> list[str]:
        """Return the ``top_n`` paragraphs most relevant to the query."""
        ...
