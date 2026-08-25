from __future__ import annotations

from langchain_core.tools import tool

from retrieval.knowledge_base import normalize


def _format(paragraphs: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[Paragraph {i + 1} — {title}] {text}"
                       for i, (title, text) in enumerate(paragraphs))


def build_tools(retriever, kb, reranker, image, top_n=20, top_k=20, lookup_limit=5):
    """Retrieval tools for one query image, over a working set the agent grows.

    Two ways in — by name (string lookup, the only channel that can beat the
    ~47% ceiling of the image embedding) and by image — and two ways to read:
    one chosen article, or every candidate at once.

    That second reader exists because committing to a single article is where
    the agent lost to plain RAG: it read the right one only 15.6% of the time
    against 41.3% for a pooled rerank, and was far more accurate than RAG
    whenever it did (0.656 vs 0.528). Pooling keeps that accuracy without
    requiring the entity to be identified correctly first.
    """
    working: dict[str, str] = {}
    tried: set[str] = set()
    tried_raw: set[str] = set()
    cache: dict = {}

    def _register(articles: list[dict]) -> None:
        for a in articles:
            working[a["title"]] = a["wiki_url"]

    def _image_candidates() -> list[dict]:
        if "articles" not in cache:
            cache["articles"] = retriever.search_index(retriever.encode_image(image),
                                                       top_k=top_k)
            _register(cache["articles"])
        return cache["articles"]

    def _pool() -> list[tuple[str, str]]:
        """Every paragraph of every article seen so far, tagged with its title."""
        _image_candidates()
        key = tuple(sorted(working.items()))
        if cache.get("pool_key") != key:
            cache["pool"] = [(title, p) for title, url in key
                             for p in kb.get_paragraphs_by_url(wiki_url=url)]
            cache["pool_key"] = key
        return cache["pool"]

    @tool
    def lookup_article(name: str) -> str:
        """Add to the search pool the article that NAME refers to.

        Pass the name alone, e.g. "Northern cardinal". Whatever it resolves to
        joins the articles that `search_paragraphs` searches — so a second call
        with a DIFFERENT name widens the search rather than replacing it.

        This is the only tool that can bring in an article the image index does
        not have. The image index only covers articles that carry a photograph,
        and reaches the right one 40.6% of the time; names reach 84.2% of the
        gold articles, but one guess resolves correctly only 11.6% of the time
        with this model. So the way to use it is several times, with genuinely
        different names — not once.
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
        _register(hits)
        return ("Added to the search pool:\n"
                + "\n".join(f"- {h['title']}" for h in hits))

    @tool
    def search_by_image() -> str:
        """List the Wikipedia articles whose reference images resemble this image.

        Use when you cannot name the entity, or to check which article the image
        actually matches. Returns titles only — read them with `read_article` or
        `search_paragraphs`.
        """
        articles = _image_candidates()
        if not articles:
            return "No articles found for this image."
        return "\n".join(f"{i:2d}. {a['title']}   (visual match {a['score']:.3f})"
                          for i, a in enumerate(articles, 1))

    @tool
    def read_article(title: str, query: str) -> str:
        """Go deeper into ONE article, after `search_paragraphs` showed which.

        Prefer `search_paragraphs` for the first read: it covers every candidate.
        Use this to pull more from a single article once you know which one holds
        the answer. `title` must be one you have already seen.
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
        return _format([(title, p) for p in results]) if results else \
            "No relevant paragraphs found."

    @tool
    def search_paragraphs(query: str) -> str:
        """Search the passages of EVERY candidate article at once — read this first.

        The normal way to gather evidence. Pass the question. Each passage is
        labelled with the article it came from, so you find the answer and the
        entity that owns it together, without having to pick the right article
        beforehand.
        """
        pool = _pool()
        if not pool:
            return "No candidate articles available for this image."
        by_text = {text: title for title, text in pool}
        best = reranker.rerank(query, [text for _, text in pool], top_n=top_n)
        return _format([(by_text.get(p, "?"), p) for p in best]) if best else \
            "No relevant paragraphs found."

    return [lookup_article, search_by_image, search_paragraphs, read_article]
