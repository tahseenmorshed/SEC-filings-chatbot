"""Reciprocal-rank fusion: merge independently-ranked lists by rank, not score.

BM25 scores and cosine similarities live on incomparable scales; RRF sidesteps the
problem by using only each document's *position* in each list:

    fused(d) = sum over lists containing d of  1 / (K + rank_of_d_in_list)

K (default 60, from Cormack et al. 2009) flattens the top of the curve so a method
must *bury* a document — not merely rank it #4 instead of #1 — before its vote stops
counting. Documents absent from a list simply contribute nothing from it, which is
what makes evidence gating upstream (BM25 refusing to rank zero-score docs) compose
cleanly: a query with no term overlap degrades gracefully to embeddings-only.

Fused scores are NOT confidence probabilities. They mean "ranked highly by one or
both methods" and their ceiling is fixed by the formula; downstream layers must not
treat them as calibrated relevance.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

# Chosen by a transparent grid sweep on the gold set (see README). The classic
# K=60 is too flat at this corpus size: chunks corroborated at mediocre ranks
# outvoted single-method #1 hits ("swamping"). K=5 sharpens top-rank votes so a
# method's confident #1 survives fusion.
DEFAULT_RRF_K = 5


def rrf_fuse(
    ranked_lists: Sequence[Sequence[int]], *, k: int = DEFAULT_RRF_K
) -> list[tuple[int, float]]:
    """Fuse ranked lists of doc indices into one ranking.

    Returns (doc_index, fused_score) sorted by descending score, ties broken by
    ascending doc index — deterministic by construction.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
