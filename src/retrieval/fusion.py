"""Unified paragraph-level retrieval API.

A single :func:`rank_paragraphs` entry-point dispatches to BM25, BGE
cross-encoder, or Reciprocal Rank Fusion depending on the chosen *strategy*.
All retrieval consumers (agentic RAG tools, static RAG baseline, offline recall
evaluator) call this function so that ranking logic lives in one place.

Strategies
----------
``"bm25"``
    BM25 lexical ranking; no BGE call.  Fast, CPU-only.
``"bge"``
    BGE cross-encoder over the full paragraph pool; no BM25 pre-filter.
``"bm25_bge"``
    BM25 top-M pre-filter → BGE top-K.
    **This is the current default pipeline.**  Calling with
    ``strategy="bm25_bge"``, ``bm25_top_m=50``, ``top_k=5`` is identical
    to the previous hard-coded behaviour and preserves all existing results.
``"rrf"``
    BM25 and BGE are run **independently** over the full paragraph pool, then
    fused with Reciprocal Rank Fusion.  Both rankings see the same pool with
    no pre-filtering.
"""

from __future__ import annotations

from retrieval.bm25 import BM25Ranker
from retrieval.reranker import CrossEncoderReranker

STRATEGIES = ("bm25", "bge", "bm25_bge", "rrf")


def rrf_score(rankings: list[list[str]], rrf_k: int = 60) -> list[str]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion.

    For each paragraph *p*::

        score(p) = Σ_i  1 / (rrf_k + rank_i(p))

    where the sum is over each input ranking that contains *p* and rank is
    1-based.  Paragraphs absent from a ranking contribute zero from that
    ranking.  The result is sorted by descending score.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, p in enumerate(ranking, 1):
            scores[p] = scores.get(p, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda p: scores[p], reverse=True)


def rank_paragraphs(
    query: str,
    paragraphs: list[str],
    strategy: str,
    top_k: int,
    *,
    bm25_top_m: int = 50,
    bm25_ranker: BM25Ranker | None = None,
    reranker: CrossEncoderReranker | None = None,
    rrf_k: int = 60,
) -> list[str]:
    """Return the *top_k* most relevant paragraphs using the chosen strategy.

    Parameters
    ----------
    query:
        Retrieval query (question text or keyword phrase).
    paragraphs:
        Full paragraph pool to rank.  All strategies see the same pool.
    strategy:
        One of ``"bm25"``, ``"bge"``, ``"bm25_bge"``, ``"rrf"``.
    top_k:
        Maximum number of paragraphs to return.
    bm25_top_m:
        BM25 candidate pool size used by ``"bm25_bge"`` before BGE reranking.
        Ignored for ``"bm25"``, ``"bge"``, and ``"rrf"``.
    bm25_ranker:
        Required when strategy ∈ {``"bm25"``, ``"bm25_bge"``, ``"rrf"``}.
    reranker:
        Required when strategy ∈ {``"bge"``, ``"bm25_bge"``, ``"rrf"``}.
    rrf_k:
        RRF smoothing constant (default 60).
    """
    if not paragraphs:
        return []
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; valid: {STRATEGIES}")

    if strategy == "bm25":
        assert bm25_ranker is not None, \
            "bm25_ranker required for strategy='bm25'"
        return bm25_ranker.rank(query, paragraphs, top_m=top_k)

    if strategy == "bge":
        assert reranker is not None, \
            "reranker required for strategy='bge'"
        return reranker.rerank(query, paragraphs, top_n=top_k)

    if strategy == "bm25_bge":
        assert bm25_ranker is not None and reranker is not None, \
            "both bm25_ranker and reranker required for strategy='bm25_bge'"
        bm25_pool = bm25_ranker.rank(query, paragraphs, top_m=bm25_top_m)
        return reranker.rerank(query, bm25_pool, top_n=top_k)

    # strategy == "rrf"
    assert bm25_ranker is not None and reranker is not None, \
        "both bm25_ranker and reranker required for strategy='rrf'"
    bm25_all = bm25_ranker.rank(
        query, paragraphs, top_m=len(paragraphs), force_sort=True
    )
    bge_all = reranker.rerank(
        query, paragraphs, top_n=len(paragraphs), force_sort=True
    )
    fused = rrf_score([bm25_all, bge_all], rrf_k=rrf_k)
    return fused[:top_k]
