"""Tokenizer and BM25: domain tokens survive; scores match hand-computed fixtures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from filings_analyst.retrieval.bm25 import BM25, tokenize


# -- tokenizer ------------------------------------------------------------------
def test_domain_identifiers_survive_tokenization():
    assert tokenize("Form 10-K filed") == ["form", "10-k", "filed"]
    assert tokenize("BB6 launched in 2025") == ["bb6", "launched", "in", "2025"]
    assert tokenize("$4.2 billion") == ["4.2", "billion"]
    assert tokenize("Item 1A. Risk Factors") == ["item", "1a", "risk", "factors"]


def test_tokenizer_lowercases_and_normalizes_unicode():
    assert tokenize("NEUTRON") == ["neutron"]
    assert tokenize("“Neutron”") == ["neutron"]  # curly quotes stripped
    assert tokenize("a b") == ["a", "b"]  # non-breaking space


def test_tokenizer_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize("—…!!") == []


# -- BM25 scoring, hand-computed -----------------------------------------------
def test_scores_match_hand_computed_values():
    docs = [
        tokenize("neutron rocket development"),
        tokenize("electron rocket launch"),
        tokenize("satellite services revenue"),
    ]
    bm25 = BM25(docs)  # k1=1.5, b=0.75; all lens 3, avg 3

    # "neutron": df=1, N=3 → idf = ln((3-1+0.5)/(1+0.5) + 1) = ln(8/3)
    # d0: tf=1, len=avg → denom = 1 + 1.5 = 2.5 → score = idf * 2.5/2.5 = idf
    scores = bm25.scores(["neutron"])
    assert scores[0] == pytest.approx(math.log(8 / 3), abs=1e-9)
    assert scores[1] == 0.0 and scores[2] == 0.0

    # "rocket": df=2 → idf = ln((3-2+0.5)/(2+0.5) + 1) = ln(1.6); d0 == d1
    scores = bm25.scores(["rocket"])
    assert scores[0] == pytest.approx(math.log(1.6), abs=1e-9)
    assert scores[0] == scores[1]


def test_rare_terms_outweigh_common_ones():
    docs = [tokenize(t) for t in [
        "rocket neutron", "rocket electron", "rocket photon", "rocket rutherford",
    ]]
    bm25 = BM25(docs)
    # "rocket" appears everywhere (low idf); "neutron" once (high idf).
    assert bm25.scores(["neutron"])[0] > bm25.scores(["rocket"])[0]


def test_length_normalization_prefers_concise_docs():
    docs = [
        tokenize("apple banana"),
        tokenize("apple banana cherry date egg fig grape hat igloo jug"),
    ]
    scores = BM25(docs).scores(["apple"])
    assert scores[0] > scores[1]


def test_term_frequency_saturates():
    docs = [
        tokenize("apple apple apple banana banana banana"),
        tokenize("apple banana banana banana banana banana"),
    ]
    scores = BM25(docs).scores(["apple"])
    assert scores[0] > scores[1]  # more mentions → higher…
    assert scores[0] < 3 * scores[1]  # …but strongly sub-linear


# -- top(): evidence gating, determinism, filters --------------------------------
def test_zero_score_docs_are_never_ranked():
    docs = [tokenize("neutron rocket"), tokenize("satellite revenue")]
    bm25 = BM25(docs)
    assert bm25.top(tokenize("neutron"), n=10) == [(0, pytest.approx(bm25.scores(["neutron"])[0]))]
    assert bm25.top(tokenize("zebra"), n=10) == []  # no evidence, no votes


def test_ties_break_by_doc_index():
    docs = [tokenize("rocket alpha"), tokenize("rocket beta")]
    top = BM25(docs).top(["rocket"], n=2)
    assert [i for i, _ in top] == [0, 1]


def test_eligible_mask_excludes_before_ranking():
    docs = [tokenize("neutron rocket"), tokenize("neutron engine")]
    bm25 = BM25(docs)
    mask = np.array([False, True])
    assert [i for i, _ in bm25.top(["neutron"], n=5, eligible=mask)] == [1]
