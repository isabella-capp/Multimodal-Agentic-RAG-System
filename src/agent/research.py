from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from PIL import Image

from agent.prompts import EXTRACTOR_SYSTEM_PROMPT
from agent.protocols import KnowledgeBase, Reranker, Retriever
from agent.tools import _format

# How many candidate neighbours to pull before de-duplicating to unique articles,
# and how much of each article to show the extractor (keeps the prompt within the
# context budget: ~20 articles × ~900 chars ≈ 5k tokens + the image).
_NEIGHBOURS = 160
_ARTICLE_CHARS = 900


class EvidenceExtractor:
    """Sub-agent: reads the top articles retrieved for an image and reports the
    key evidence for the question in prose. It is multimodal — it sees the image
    and uses it to pick the right article among visually similar candidates (an
    LLM re-ranker), replacing the cross-encoder as the first-pass reader.
    """

    def __init__(self, llm: BaseChatModel, kb: KnowledgeBase, max_tokens: int = 384):
        self.llm = llm
        self.kb = kb
        self.max_tokens = max_tokens

    def _articles_block(self, articles: list[dict]) -> str:
        blocks = []
        for i, art in enumerate(articles):
            text = " ".join(self.kb.get_paragraphs_by_url(wiki_url=art["wiki_url"]))
            if not text:
                continue
            blocks.append(f"[Article {i + 1}] {art['title']}\n{text[:_ARTICLE_CHARS]}")
        return "\n\n".join(blocks)

    def extract(self, image_data_uri: str, focus: str, articles: list[dict]) -> str:
        block = self._articles_block(articles)
        if not block:
            return "No relevant evidence found."
        user = HumanMessage(content=[
            {"type": "text", "text": f"QUESTION: {focus}\n\nCANDIDATE ARTICLES:\n{block}"},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ])
        resp = self.llm.invoke(
            [SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT), user],
            max_tokens=self.max_tokens,
        )
        return resp.content if isinstance(resp.content, str) else str(resp.content)


def gather_candidates(retriever: Retriever, image: Image.Image, n: int) -> list[dict]:
    """Top-n unique articles retrieved for the image (shared by both tools)."""
    articles = retriever.search_index(retriever.encode_image(image), top_k=_NEIGHBOURS)
    return articles[:n]


def build_research_tools(extractor: EvidenceExtractor, reranker: Reranker, kb: KnowledgeBase,
                         image_data_uri: str, articles: list[dict],
                         extractor_articles: int = 20, top_n: int = 5):
    """Two tools over the image-retrieved article pool. The full pool (e.g. 50
    articles) feeds the cheap cross-encoder refine; the extractor reads only the
    top ``extractor_articles`` (e.g. 20) to stay within the LLM context budget.

    - ``research``          — forced first: multimodal extractor → evidence.
    - ``search_paragraphs`` — optional refine: cross-encoder over the whole pool's
      paragraphs for a specific query, for a second hop when evidence is thin.
    """
    pool = [p for a in articles for p in kb.get_paragraphs_by_url(wiki_url=a["wiki_url"])]
    extractor_arts = articles[:extractor_articles]

    @tool
    def research(focus: str) -> str:
        """Gather evidence about the entity in the IMAGE from Wikipedia.

        Pass what you need to find out (e.g. the question). Returns the key facts
        found in the retrieved articles — answer the question using this evidence.
        """
        return extractor.extract(image_data_uri, focus, extractor_arts)

    @tool
    def search_paragraphs(query: str) -> str:
        """Search the SAME retrieved articles for specific paragraphs matching your query.

        Use this only if the evidence from `research` is missing what you need:
        it returns the paragraphs most relevant to `query` from those articles.
        """
        if not pool:
            return "No paragraphs available for this image."
        results = reranker.rerank(query, pool, top_n=top_n)
        return _format(results) if results else "No relevant paragraphs found."

    return [research, search_paragraphs]
