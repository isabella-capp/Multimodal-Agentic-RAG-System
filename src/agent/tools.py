from __future__ import annotations

from langchain_core.tools import tool
from PIL import Image

from agent.protocols import KnowledgeBase, Reranker, Retriever


def _format(paragraphs: list[str]) -> str:
    return "\n\n".join(f"[Paragraph {i + 1}] {p}" for i, p in enumerate(paragraphs))


def build_tools(retriever: Retriever, kb: KnowledgeBase, reranker: Reranker,
                image: Image.Image, top_n: int = 5):
    """On-demand image-grounded retrieval tool bound to the query image.

    The visual candidate pool is memoised (the image never changes); the tool
    reranks it by the agent's query and returns the top paragraphs. The text-side
    channel was removed: the cross-modal text→image path is broken (misaligned
    encode_text) and never returned relevant articles.
    """
    visual: dict = {}

    def _visual_pool() -> list[str]:
        if "pool" not in visual:
            articles = retriever.retrieve(image=image, question=None)
            visual["pool"] = [
                p for a in articles for p in kb.get_paragraphs_by_url(wiki_url=a["wiki_url"])
            ]
        return visual["pool"]

    @tool
    def search_by_image(query: str) -> str:
        """Retrieve facts from Wikipedia articles that match the IMAGE, ranked by your query.

        Use this first — it grounds on the entity actually shown in the image.
        """
        pool = _visual_pool()
        if not pool:
            return "No articles found for this image."
        results = reranker.rerank(query, pool, top_n=top_n)
        return _format(results) if results else "No relevant paragraphs found."

    return [search_by_image]
