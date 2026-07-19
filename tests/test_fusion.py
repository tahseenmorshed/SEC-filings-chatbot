"""RRF fusion: the worked example, corroboration, ties, and graceful degradation."""

from __future__ import annotations

import pytest

from filings_analyst.retrieval.fusion import rrf_fuse


def test_worked_example_corroboration_wins():
    # Doc 20 is #2 in one list and #1 in the other; doc 10 is #1 in one list only.
    fused = rrf_fuse([[10, 20, 30], [20, 40]], k=60)
    order = [doc for doc, _ in fused]
    assert order == [20, 10, 40, 30]
    scores = dict(fused)
    assert scores[20] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[10] == pytest.approx(1 / 61)


def test_single_strong_vote_beats_two_deeply_buried_ones():
    # With K=60 the curve is deliberately flat: two mid-list votes (ranks ~31/~35)
    # legitimately beat a single #1. A doc must be buried past rank ~65 in BOTH
    # lists before one #1 outweighs it: 1/131 + 1/135 < 1/61.
    fused = rrf_fuse(
        [[1] + list(range(100, 169)) + [2], list(range(200, 274)) + [2]], k=60
    )
    scores = dict(fused)
    assert scores[2] == pytest.approx(1 / 131 + 1 / 135)  # ranks 71 and 75
    assert scores[1] > scores[2]


def test_two_moderate_votes_beat_one_top_vote():
    # The flip side of K=60 flattening, asserted explicitly: ranks 31+35 in both
    # lists outweigh a lone #1 (1/91 + 1/95 > 1/61).
    fused = rrf_fuse([[1] + list(range(100, 129)) + [2], list(range(200, 234)) + [2]], k=60)
    scores = dict(fused)
    assert scores[2] > scores[1]


def test_ties_break_by_doc_id():
    fused = rrf_fuse([[7], [3]], k=60)  # both get exactly 1/61
    assert [doc for doc, _ in fused] == [3, 7]


def test_empty_lists_degrade_gracefully():
    assert rrf_fuse([[], []]) == []
    # BM25 found nothing → embeddings-only ranking passes through in order.
    fused = rrf_fuse([[], [5, 6, 7]], k=60)
    assert [doc for doc, _ in fused] == [5, 6, 7]
