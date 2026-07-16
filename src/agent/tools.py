from __future__ import annotations

from langchain_core.tools import tool

from retrieval.reranker import CrossEncoderReranker


def make_search_tool(
    reranker: CrossEncoderReranker,
    paragraph_pool: list[str],
    top_n: int,
):
    """Create a ``search_paragraphs`` tool bound to a fixed paragraph pool.

    Parameters
    ----------
    reranker : CrossEncoderReranker
        The cross-encoder reranker (already loaded).
    paragraph_pool : list[str]
        All paragraphs extracted from the FAISS top-k articles for this
        query.  This list is **fixed** for the lifetime of a single agent
        session — the agent cannot change it.
    top_n : int
        Number of paragraphs to return per tool invocation (the optimal
        value found during the ablation study).
    """

    @tool
    def search_paragraphs(query: str) -> str:
        """Search the knowledge base for paragraphs relevant to a query.

        Use this tool when you need factual information to answer the
        user's question about the image.  You can rephrase or refine
        your query to find different paragraphs each time.

        Args:
            query: A natural-language search query describing what
                   information you are looking for.
        """
        if not paragraph_pool:
            return "No paragraphs available in the knowledge base for this image."

        results = reranker.rerank(query, paragraph_pool, top_n=top_n)

        if not results:
            return "No relevant paragraphs found for this query."

        formatted = "\n\n".join(
            f"[Paragraph {i + 1}] {p}" for i, p in enumerate(results)
        )
        return (
            f"Found {len(results)} relevant paragraphs:\n\n{formatted}"
        )

    return search_paragraphs
