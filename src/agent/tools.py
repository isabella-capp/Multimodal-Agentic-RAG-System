from __future__ import annotations

from langchain_core.tools import tool

from retrieval.knowledge_base import normalize


def _format(paragraphs: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"[Paragraph {i + 1} — {title}] {text}"
                       for i, (title, text) in enumerate(paragraphs))


def build_tools(retriever, kb, reranker, image, top_n=20, top_k=20,
                lookup_limit=5, show_candidates=10):
    """Retrieval tools for one query image, over a working set the agent grows.

    Two tools, and no pooled search: reranking every candidate's paragraphs at
    once is how baseline B works, and giving the agent that shortcut collapsed it
    into B — 978 of 1000 examples made a single pooled call with the question
    verbatim, which is B's algorithm with extra steps.

    The whole gap to the 235B is entity choice, not reasoning: both answer ~0.14
    when they read the wrong article and ~0.7 when they read the right one, but
    the 235B reads the right one 40.1% of the time against 15.6%. And 40.1% is
    exactly the image index's recall@20 — the big model extracts everything the
    ranking offers, the small one throws half of it away by trusting its own
    name. So the ranking, with its scores, is what `identify` leads with.
    """
    working: dict[str, str] = {}
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
    def identify(name: str = "") -> str:
        """The articles the IMAGE itself retrieves, best match first.

        This ranking is the strongest evidence available about what the picture
        shows — stronger than your own impression of it. Pass your guess as
        `name` if you have one and it will be located in the list; leave it empty
        to just see the ranking.
        """
        candidates = _image_candidates()
        lines = [f"{i:2d}. {a['title']}   (visual match {a['score']:.3f})"
                 for i, a in enumerate(candidates[:show_candidates], 1)]
        listing = "\n".join(lines)

        note = ""
        if name:
            wanted = normalize(name)
            hit = next((i for i, a in enumerate(candidates, 1)
                        if normalize(a["title"]) == wanted), None)
            if hit:
                note = f"\n\nYour guess '{name}' is #{hit} in this ranking."
            else:
                resolved = kb.lookup_articles(name, limit=1)
                if resolved:
                    _register(resolved)
                    note = (f"\n\nYour guess '{resolved[0]['title']}' is a real article but "
                            f"the image does not retrieve it at all. The ranking above "
                            f"is the better evidence.")
                else:
                    note = f"\n\nNo article is named '{name}'."
        return f"Articles this image matches, best first:\n{listing}{note}"

    @tool
    def read_article(title: str, query: str) -> str:
        """Read the passages of ONE candidate article that match `query`.

        `title` must be one of the candidates `identify` listed. Read the top
        candidate first; if what comes back does not match the picture, read the
        next one down. Pass the question as `query`.
        """
        url = working.get(title)
        if url is None:
            hits = kb.lookup_articles(title, limit=1)
            if not hits:
                return (f"Unknown article '{title}'. Use `identify` and read one "
                        f"of the candidates it lists.")
            url = hits[0]["wiki_url"]
        paragraphs = kb.get_paragraphs_by_url(wiki_url=url)
        if not paragraphs:
            return f"No text available for '{title}'."
        results = reranker.rerank(query, paragraphs, top_n=top_n)
        return _format([(title, p) for p in results]) if results else \
            "No relevant paragraphs found."

    return [identify, read_article]
