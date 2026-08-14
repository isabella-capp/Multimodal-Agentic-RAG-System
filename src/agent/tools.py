from __future__ import annotations

from langchain_core.tools import tool
from PIL import Image

from agent.protocols import KnowledgeBase, Reranker, Retriever


def _format(paragraphs: list[str]) -> str:
    return "\n\n".join(f"[Paragraph {i + 1}] {p}" for i, p in enumerate(paragraphs))


def build_tools(retriever: Retriever, kb: KnowledgeBase, reranker: Reranker,
                image: Image.Image,
                top_n: int = 20, top_k: int = 20, lookup_limit: int = 5):
    """Retrieval tools for one query image, over a working set the agent grows.

    Two independent ways in — by name (string lookup, the only channel that can
    beat the ~47% ceiling of the image embedding) and by image — plus a reader
    that runs the cross-encoder over a chosen article's full text.
    """
    working: dict[str, str] = {}

    def _register(articles: list[dict]) -> None:
        for a in articles:
            working[a["title"]] = a["wiki_url"]

    @tool
    def lookup_article(name: str) -> str:
        """Find a Wikipedia article by the NAME of the entity shown in the image.

        Use this first whenever you can name what you see (species, landmark,
        building, artwork, ...). Pass the name alone, e.g. "Northern cardinal".
        """
        hits = kb.lookup_articles(name, limit=lookup_limit)
        if not hits:
            return (f"No article found for '{name}'. Try a different name, "
                    f"or use search_by_image.")
        _register(hits)
        return "\n".join(f"- {h['title']}" for h in hits)

    @tool
    def search_by_image() -> str:
        """List the Wikipedia articles whose reference images resemble this image.

        Use when you cannot name the entity, or to check which article the image
        actually matches. Returns titles only — read one with `read_article`.
        """
        articles = retriever.search_index(retriever.encode_image(image), top_k=top_k)
        if not articles:
            return "No articles found for this image."
        _register(articles)
        return "\n".join(f"- {a['title']}" for a in articles)

    @tool
    def read_article(title: str, query: str) -> str:
        """Read the passages of one article that best match `query`.

        `title` must be one returned by `lookup_article` or `search_by_image`.
        Pass the question, or the specific fact you still need, as `query`.
        """
        url = working.get(title)
        if url is None:
            hits = kb.lookup_articles(title, limit=1)
            if not hits:
                return (f"Unknown article '{title}'. Find it with lookup_article "
                        f"or search_by_image first.")
            url = hits[0]["wiki_url"]
        paragraphs = kb.get_paragraphs_by_url(wiki_url=url)
        if not paragraphs:
            return f"No text available for '{title}'."
        results = reranker.rerank(query, paragraphs, top_n=top_n)
        return _format(results) if results else "No relevant paragraphs found."

    return [lookup_article, search_by_image, read_article]
