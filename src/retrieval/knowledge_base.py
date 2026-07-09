import json


class KnowledgeBase:
    """Encyclopedic knowledge base backed by a JSON file.

    Supports two JSON layouts:

    * **dict** - ``{wiki_url: {section_texts: [...], ...}, ...}``
    * **list** - ``[{retrieval: [{url, section_texts, ...}], wikipedia_url, ...}, ...]``
      (the format of ``example_kb.json`` / Encyclopedic-VQA dataset entries).

    In both cases an internal ``url -> article`` index is built so that
    lookups by Wikipedia URL are O(1).
    """

    def __init__(self, kb_path: str):
        """
        Parameters
        ----------
        kb_path : str
            Path to the encyclopedic KB JSON file.
        """
        self.kb_path = kb_path
        self._url_index: dict[str, dict] = {}
        self._load_kb()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_kb(self):
        print(f"Loading Knowledge Base from {self.kb_path} …")
        with open(self.kb_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict):
            # Format: {url: {section_texts: [...], section_titles: [...], ...}}
            self._url_index = raw
        elif isinstance(raw, list):
            # Format: list of dataset entries (example_kb.json style)
            self._build_index_from_list(raw)
        else:
            raise ValueError(
                f"Unsupported KB format: expected dict or list, got {type(raw)}"
            )

        print(f"Knowledge Base loaded: {len(self._url_index)} articles indexed.")

    def _build_index_from_list(self, entries: list[dict]):
        """Build a url→article index from a list of dataset entries.

        Each entry may contain a ``retrieval`` list with objects that have
        ``url``, ``section_texts``, ``section_titles``, etc.  We also
        fall back to ``wikipedia_url`` if present.
        """
        for entry in entries:
            # Index from the `retrieval` blocks (primary source of article data)
            for ret_block in entry.get("retrieval", []):
                url = ret_block.get("url")
                if url and url not in self._url_index:
                    self._url_index[url] = ret_block

            # Also index the top-level wikipedia_url → whole entry
            top_url = entry.get("wikipedia_url")
            if top_url and top_url not in self._url_index:
                self._url_index[top_url] = entry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_paragraphs_by_url(self, wiki_url: str) -> list[str]:
        """Return non-empty section texts for a given Wikipedia URL."""
        article = self._url_index.get(wiki_url)
        if not article:
            return []

        section_texts = article.get("section_texts", [])
        return [p for p in section_texts if p and p.strip()]

    def get_article_by_url(self, wiki_url: str) -> dict | None:
        """Return the full article dict for a given Wikipedia URL.

        The dict typically contains ``section_texts``, ``section_titles``,
        ``title``, ``url``, ``image_urls``, etc.  Returns ``None`` if the
        URL is not found.
        """
        return self._url_index.get(wiki_url)

    def __len__(self) -> int:
        return len(self._url_index)

    def __contains__(self, wiki_url: str) -> bool:
        return wiki_url in self._url_index