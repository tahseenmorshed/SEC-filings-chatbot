"""Lexical search: domain tokenizer + BM25 (Okapi), implemented from the formula.

Why hand-rolled rather than a library: the formula is ~30 lines of published math
(testable against hand-computed fixtures), while the part that actually decides
correctness in this domain — tokenization — must be custom anyway. Naive word
splitting destroys the most valuable tokens in SEC text: "10-K" must not become
"10" + "K", "BB6" and "bge-small"-style identifiers must survive intact.

Tokenizer decisions (each deliberate):
* lowercase ("Neutron" == "neutron");
* hyphens/dots *inside* alphanumeric runs stay ("10-k", "bb6", "4.2");
* no stopword list — BM25's IDF already down-weights common words smoothly;
* no stemming — embeddings cover word-form variation (that's the hybrid's division
  of labor); BM25 stays exact;
* queries pass through the same unicode normalization as chunk text, so a phrase
  pasted from a filing matches itself.

BM25 in one breath: a chunk scores higher when it contains the query's words, where
rare words count more than common ones (IDF), repeated words saturate rather than
scale linearly (k1), and long documents are penalized so they can't win on bulk (b).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ..parsing.extract import normalize_text

# Alphanumeric runs, optionally joined by internal hyphens/dots: "10-k", "4.2", "bb6".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

# Standard Okapi parameters (term-frequency saturation, length normalization).
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """Canonical tokenization for both indexed documents and queries."""
    return _TOKEN_RE.findall(normalize_text(text).lower())


class BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenized documents."""

    def __init__(
        self,
        docs_tokens: list[list[str]],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(docs_tokens)
        self._doc_len = np.array([len(d) for d in docs_tokens], dtype=np.float64)
        self._avg_len = float(self._doc_len.mean()) if self.n_docs else 0.0

        # Postings: token -> (doc indices, term frequencies). Document frequency for
        # IDF falls out of the postings length.
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, tokens in enumerate(docs_tokens):
            for token, tf in Counter(tokens).items():
                postings.setdefault(token, []).append((i, tf))
        self._postings = {
            t: (
                np.array([i for i, _ in plist], dtype=np.int64),
                np.array([tf for _, tf in plist], dtype=np.float64),
            )
            for t, plist in postings.items()
        }
        # IDF with the +1 inside the log (Lucene-style): never negative, smooth.
        self._idf = {
            t: math.log((self.n_docs - len(idx) + 0.5) / (len(idx) + 0.5) + 1.0)
            for t, (idx, _) in self._postings.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 score of every document for the query (0.0 = no term overlap)."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        if not self.n_docs:
            return out
        for token in query_tokens:
            entry = self._postings.get(token)
            if entry is None:
                continue
            doc_idx, tf = entry
            denom = tf + self.k1 * (
                1.0 - self.b + self.b * self._doc_len[doc_idx] / self._avg_len
            )
            out[doc_idx] += self._idf[token] * tf * (self.k1 + 1.0) / denom
        return out

    def top(
        self,
        query_tokens: list[str],
        n: int,
        *,
        eligible: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        """Top-n (doc_index, score) with evidence gating and deterministic ties.

        Evidence gating: documents scoring 0.0 (no query-term overlap) are never
        ranked — an arbitrary ordering of hundreds of zero-score documents would
        inject noise votes into fusion. ``eligible`` is an optional boolean mask
        (filters are applied *before* ranking, so ranks are computed among eligible
        documents only).
        """
        s = self.scores(query_tokens)
        if eligible is not None:
            s = np.where(eligible, s, 0.0)
        idx = np.nonzero(s > 0.0)[0]
        # Deterministic: sort by (-score, doc index).
        order = sorted(idx.tolist(), key=lambda i: (-s[i], i))[:n]
        return [(i, float(s[i])) for i in order]
