from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from retrieval.knowledge_base import normalize
from retrieval.fusion import rank_paragraphs

# Maps the agent-facing retrieval_mode names to fusion strategies.
_MODE_TO_STRATEGY = {
    "bm25+reranker": "bm25_bge",   # current default — behaviour unchanged
    "reranker":      "bge",
    "rrf":           "rrf",
}

class LookupArticleInput(BaseModel):
    name: str = Field(..., description="The exact name of the entity, person, or object to look up on Wikipedia.")

class SearchParagraphsInput(BaseModel):
    query: str = Field(..., description="A short, highly focused keyword phrase (e.g. 'Arabidopsis lyrata outcrossing') to find specific information. Do not use full sentences or questions.")

class ReadArticleInput(BaseModel):
    title: str = Field(..., description="The EXACT title of the Wikipedia article, exactly as it appeared in previous tool results.")
    query: str = Field(..., description="The keyword phrase to search for inside this specific article.")

class SearchByImageNameInput(BaseModel):
    name: str = Field(..., description="The name of the entity you identified in the image "
                     "(e.g. 'Golden Gate Bridge', 'Tiliqua rugosa'). "
                     "Used by GroundingDINO to locate and crop the subject.")

@dataclass
class Candidate:
    """One article in the working set, with its provenance."""

    title: str
    wiki_url: str
    sources: set[str] = field(default_factory=set)   # "image" | "lookup"
    image_score: float | None = None


def _format(paragraphs: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[Paragraph {i + 1} — {title}] {text}"
                       for i, (title, text) in enumerate(paragraphs))


def build_tools(retriever, kb, reranker, bm25, image,
                top_n=5, top_k=20, bm25_top_m=50, lookup_limit=5,
                retrieval_mode: str = "reranker", rrf_k: int = 60,
                grounder=None, visual_mode: str = "image_only"):
    """Retrieval tools for one query image, over a working set the agent grows.

    Two ways into the KB -- by image (EVA-CLIP/FAISS) and by name -- and two
    ways to read: search across all candidates, or read one article deeply.
    The working set tracks how each article was found (provenance) but this
    information is kept internal; tool outputs are unchanged.

    retrieval_mode controls the paragraph-level pipeline:
      "reranker"       -- all paragraphs go directly to the cross-encoder (default)
      "bm25+reranker"  -- BM25 pre-filter -> cross-encoder
      "rrf"            -- BM25 + BGE independent rankings -> Reciprocal Rank Fusion

    visual_mode controls how search_by_image retrieves candidates:
      "image_only"  -- full image only; no name argument needed (default, backward-compat).
      "crop_only"   -- GroundingDINO crop only; agent must pass entity name.
      "both"        -- full image + crop, fused with RRF; agent must pass entity name.
    grounder must be a Grounder instance when visual_mode != 'image_only'.
    """
    candidates: dict[str, Candidate] = {}   # keyed by wiki_url
    tried: set[str] = set()
    tried_raw: set[str] = set()
    cache: dict = {}

    def _register_image(articles: list[dict]) -> None:
        for a in articles:
            url = a["wiki_url"]
            if url in candidates:
                candidates[url].sources.add("image")
            else:
                candidates[url] = Candidate(
                    title=a["title"],
                    wiki_url=url,
                    sources={"image"},
                    image_score=a.get("score"),
                )

    def _register_lookup(articles: list[dict]) -> None:
        for a in articles:
            url = a["wiki_url"]
            if url in candidates:
                candidates[url].sources.add("lookup")
            else:
                candidates[url] = Candidate(
                    title=a["title"],
                    wiki_url=url,
                    sources={"lookup"},
                    image_score=None,
                )

    def _image_candidates() -> list[dict]:
        if "full" not in cache:
            cache["full"] = retriever.search_index(
                retriever.encode_image(image), top_k=top_k
            )
            _register_image(cache["full"])
        return cache["full"]

    def _crop_candidates(name: str) -> list[dict] | None:
        """Return crop-based FAISS results for *name*, or None if detection fails."""
        if name not in cache.setdefault("crop", {}):
            try:
                cropped = grounder.crop(image, name)  # type: ignore[union-attr]
            except Exception as exc:
                cache["crop"][name] = (None, str(exc))
                return None
            if cropped is None:
                cache["crop"][name] = (None, "not detected")
                return None
            emb = retriever.encode_image(cropped)
            results = retriever.search_index(emb, top_k=top_k * 2)
            cache["crop"][name] = (results, None)
        results, err = cache["crop"][name]
        return results  # None when detection failed

    def _rrf_visual(full: list[dict], crop: list[dict], k: int = 60) -> list[dict]:
        """RRF merge of two EVA-CLIP FAISS result lists."""
        scores: dict[str, float] = {}
        by_url: dict[str, dict] = {}
        for rank, a in enumerate(full, 1):
            url = a["wiki_url"]
            scores[url] = scores.get(url, 0.0) + 1.0 / (k + rank)
            by_url.setdefault(url, a)
        for rank, a in enumerate(crop, 1):
            url = a["wiki_url"]
            scores[url] = scores.get(url, 0.0) + 1.0 / (k + rank)
            by_url.setdefault(url, a)
        ranked = sorted(scores, key=lambda u: scores[u], reverse=True)
        return [by_url[u] for u in ranked[:top_k]]

    def _pool() -> list[tuple[str, str]]:
        """Every paragraph of every article in the working set, tagged with title."""
        _image_candidates()
        key = tuple(sorted((c.wiki_url, c.title) for c in candidates.values()))
        if cache.get("pool_key") != key:
            cache["pool"] = [
                (c.title, p)
                for c in candidates.values()
                for p in kb.get_paragraphs_by_url(wiki_url=c.wiki_url)
            ]
            cache["pool_key"] = key
        return cache["pool"]

    @tool(args_schema=LookupArticleInput)
    def lookup_article(name: str) -> str:
        """Add Wikipedia articles matching an entity name to the candidate pool.
        
        Use this when you want to explore a specific entity by name, or if the 
        image search didn't give good results. Calling this multiple times with 
        DIFFERENT names expands your search pool.
        """
        if normalize(name) in tried:
            return (f"You already tried '{name}'. Names tried so far: "
                    f"{', '.join(sorted(tried_raw))}. Give a different one — a "
                    f"more specific name, the common name instead of the "
                    f"scientific one, or another candidate entirely.")
        tried.add(normalize(name)); tried_raw.add(name)
        hits = kb.lookup_articles(name, limit=lookup_limit)
        if not hits:
            return (f"No article found for '{name}'. Try a different name, "
                    f"or use search_by_image to see what the image matches.")
        _register_lookup(hits)
        return ("Added to the search pool:\n"
                + "\n".join(f"- {h['title']}" for h in hits))

    # --- search_by_image: mode-dispatched -----------------------------------
    # Three visual_mode values control schema and behaviour:
    #   image_only  -- no name arg, full image only (default, backward-compat)
    #   crop_only   -- name required, GroundingDINO crop only
    #   both        -- name required, full + crop fused with RRF

    if visual_mode == "image_only":
        @tool
        def search_by_image() -> str:
            """Find candidate Wikipedia articles whose reference images are visually
            similar to the input image (EVA-CLIP / FAISS).

            Use this as the first retrieval step.  Do not assume the top result is
            correct -- use lookup_article and search_paragraphs to verify.
            All returned articles are added to the working set automatically.
            """
            articles = _image_candidates()
            if not articles:
                return "No articles found for this image."
            return "\n".join(f"{i:2d}. {a['title']}   (visual match {a['score']:.3f})"
                              for i, a in enumerate(articles, 1))

    elif visual_mode == "crop_only":
        @tool(args_schema=SearchByImageNameInput)
        def search_by_image(name: str) -> str:
            """Find candidate Wikipedia articles by retrieving with a GroundingDINO
            crop of the identified subject rather than the full image.

            Provide the entity name you identified (``name``): GroundingDINO will
            locate that subject in the image, crop and upscale it, and retrieve
            visually similar articles from the FAISS index using the crop embedding.
            All returned articles are added to the working set automatically.
            """
            crop = _crop_candidates(name)
            if crop is None:
                reason = (cache.get("crop") or {}).get(name, (None, "unknown"))[1]
                return (f"No '{name}' detected in the image ({reason}). "
                        f"Try a different, more specific name.")
            _register_image(crop)
            if not crop:
                return "No articles found for the cropped image."
            return "\n".join(f"{i:2d}. {a['title']}   (visual match {a['score']:.3f})"
                              for i, a in enumerate(crop[:top_k], 1))

    else:  # visual_mode == "both"
        @tool(args_schema=SearchByImageNameInput)
        def search_by_image(name: str) -> str:
            """Find candidate Wikipedia articles by fusing full-image and
            subject-crop retrieval with Reciprocal Rank Fusion.

            Provide the entity name you identified (``name``): GroundingDINO
            locates that subject in the image, crops it, and EVA-CLIP retrieves
            candidates from both the full image and the crop.  The two ranked
            lists are merged with RRF so high-scoring articles from either query
            float to the top.  All returned articles are added to the working set.
            """
            full = _image_candidates()
            crop = _crop_candidates(name)
            if crop is None:
                # Fall back to full-image results with a warning.
                reason = (cache.get("crop") or {}).get(name, (None, "unknown"))[1]
                articles = full
                note = f"  [GroundingDINO: '{name}' not detected ({reason}); full-image only]"
            else:
                articles = _rrf_visual(full, crop)
                note = ""
            _register_image(articles)
            if not articles:
                return "No articles found."
            return "\n".join(f"{i:2d}. {a['title']}   (visual match {a['score']:.3f})"
                              for i, a in enumerate(articles, 1)) + note


    @tool(args_schema=ReadArticleInput)
    def read_article(title: str, query: str) -> str:
        """Extract relevant text passages from one specific candidate article.
        
        Use this ONLY when you want to deeply inspect a single article that you 
        have already discovered. It returns the most relevant paragraphs from that article.
        """
        cand = next((c for c in candidates.values() if c.title == title), None)
        url = cand.wiki_url if cand else None
        if url is None:
            hits = kb.lookup_articles(title, limit=1)
            if not hits:
                return (f"Unknown article '{title}'. Find it with lookup_article "
                        f"or search_by_image first.")
            url = hits[0]["wiki_url"]
            _register_lookup(hits)
        paragraphs = kb.get_paragraphs_by_url(wiki_url=url)
        if not paragraphs:
            return f"No text available for '{title}'."
        strategy = _MODE_TO_STRATEGY.get(retrieval_mode, "bge")
        results = rank_paragraphs(
            query, paragraphs, strategy=strategy, top_k=top_n,
            bm25_top_m=bm25_top_m, bm25_ranker=bm25, reranker=reranker,
            rrf_k=rrf_k,
        )
        return _format([(title, p) for p in results]) if results else \
            "No relevant paragraphs found."

    @tool(args_schema=SearchParagraphsInput)
    def search_paragraphs(query: str) -> str:
        """Search for relevant text passages across ALL currently loaded candidate articles.
        
        Use this to quickly gather facts and compare information across all the articles 
        in your pool simultaneously. Each returned passage will tell you which article it came from.
        """
        pool = _pool()
        if not pool:
            return "No candidate articles available for this image."
        by_text = {text: title for title, text in pool}
        texts = [text for _, text in pool]
        strategy = _MODE_TO_STRATEGY.get(retrieval_mode, "bge")
        best = rank_paragraphs(
            query, texts, strategy=strategy, top_k=top_n,
            bm25_top_m=bm25_top_m, bm25_ranker=bm25, reranker=reranker,
            rrf_k=rrf_k,
        )
        return _format([(by_text.get(p, "?"), p) for p in best]) if best else \
            "No relevant paragraphs found."

    return [lookup_article, search_by_image, search_paragraphs, read_article]
