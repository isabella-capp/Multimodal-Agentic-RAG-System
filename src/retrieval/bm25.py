from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Lower-case, strip accents, split on non-alphanumeric runs."""
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", t.lower())


class BM25Ranker:
    """Lexical candidate filter using BM25Okapi (rank_bm25).

    Built on demand over the paragraphs supplied to each call — no persistent
    index.  Intended as the first stage before BGE cross-encoder reranking:
    BM25 cheaply prunes the pool from O(working-set) to a manageable top-M
    before the expensive model pass.
    """

    def rank(
        self,
        query: str,
        paragraphs: list[str],
        top_m: int,
        force_sort: bool = False,
    ) -> list[str]:
        """Return up to *top_m* paragraphs ranked by BM25 score.

        If the pool is already small enough (≤ top_m) the paragraphs are
        returned as-is so the caller never loses coverage, unless
        *force_sort* is True, in which case the full pool is always sorted
        (required e.g. for RRF which needs the complete BM25 ordering).
        """
        if not paragraphs:
            return []
        if len(paragraphs) <= top_m and not force_sort:
            return paragraphs

        tokenized = [_tokenize(p) for p in paragraphs]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(_tokenize(query))
        order = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)
        return [paragraphs[i] for i in order[:top_m]]
