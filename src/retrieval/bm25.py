from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Lower-case, strip accents, split on non-alphanumeric runs."""
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", t.lower())


class BM25Ranker:
    """DEPRECATED. Lexical candidate filter using BM25Okapi (rank_bm25).

    Built on demand over the paragraphs supplied to each call — no persistent
    index. Intended as the first stage before BGE cross-encoder reranking: BM25
    cheaply prunes the pool from O(working-set) to a manageable top-M before the
    expensive model pass.

    Measured, it does not pay. At matched top_n=20 the cross-encoder alone
    scores 0.380 and fusing it with this by RRF 0.374; as a pre-filter it gives
    0.366 against 0.369 without, and top_m 50 against 20 changes nothing
    (0.366/0.367). Every difference sits inside the ~0.3 point run-to-run noise.

    The reason is visible in the construction: an index built over the ~120
    paragraphs of one call has no usable IDF, and IDF is the whole point of
    BM25 — with 120 documents there is no telling a rare term from a common one.
    Lexical retrieval that does work is ``KnowledgeBase.search_articles_by_text``,
    where BM25 runs over all 14.1M paragraphs and adds 12.3 points of article
    coverage.

    Kept so the `bm25`, `bm25_bge` and `rrf` strategies still run; do not build
    on it.
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
