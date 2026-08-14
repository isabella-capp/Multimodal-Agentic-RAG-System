from __future__ import annotations

from langchain_core.tools import tool
from PIL import Image
from pydantic import BaseModel, Field

from agent.protocols import KnowledgeBase, Reranker, Retriever


def _format(paragraphs: list[str]) -> str:
    """Format a list of paragraphs as a numbered block for the model."""
    return "\n\n".join(f"[Paragraph {i + 1}] {p}" for i, p in enumerate(paragraphs))


class SearchInput(BaseModel):
    query: str = Field(
        description="The query MUST be highly specific and informative. Include the relevant context from the user's question and clearly express what information is being sought."
    )

class SubmitInput(BaseModel):
    answer: str = Field(
        description="The final short answer (1-4 words). NEVER a full sentence. NEVER restate the question."
    )


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

    # 2. Assegniamo lo schema al tool di ricerca
    @tool(args_schema=SearchInput)
    def search_by_image(query: str) -> str:
        """Retrieve relevant facts from Wikipedia articles about the entity shown
        in the IMAGE.

        ALWAYS call this tool (or submit_final_answer) — never reply in free text.
        Use this to retrieve evidence BEFORE you are ready to answer, or when
        the previous results are insufficient. Grounds on the entity actually
        shown in the image.
        """
        pool = _visual_pool()
        if not pool:
            return "No articles found for this image."
        results = reranker.rerank(query, pool, top_n=top_n)
        return _format(results) if results else "No relevant paragraphs found."
    
    # 3. Assegniamo lo schema al tool di invio risposta
    @tool(args_schema=SubmitInput)
    def submit_final_answer(answer: str) -> str:
        """Submit your FINAL answer to the user's question and END the search loop.

        Call this ONLY when you have enough retrieved evidence to answer confidently.
        Do NOT call this tool if you still need more evidence — call search_by_image instead.
        """
        # Returning the sentinel value lets the agent loop detect termination.
        return f"__FINAL__:{answer}"

    return [search_by_image, submit_final_answer]