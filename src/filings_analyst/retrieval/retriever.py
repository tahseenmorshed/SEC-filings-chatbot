"""The Retriever: query → hybrid-ranked chunks with full provenance + diagnostics.

Flow per query (no routing, no staged cutoffs — see design notes in the package
docstring): apply filters to get the eligible set → BM25 ranks eligible chunks it has
term evidence for → embeddings rank all eligible chunks (per-window cosine,
max-pooled per chunk) → RRF merges the two rankings → top-k returned with per-method
diagnostics (so "why did this rank #1?" is always answerable, and the future
answering layer gets the raw signals refusal calibration will need).

Try it::

    uv run python -m filings_analyst.retrieval.retriever "Neutron status" --k 5
    uv run python -m filings_analyst.retrieval.retriever "launch plans" --ticker ASTS
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bm25 import BM25, tokenize
from .encoders import Encoder
from .fusion import DEFAULT_RRF_K, rrf_fuse
from .index import Index

# Each method contributes its top-N candidates to fusion (distinct from returned k).
DEFAULT_FUSION_DEPTH = 50
DEFAULT_K = 8

Method = Literal["fused", "bm25", "embed"]


@dataclass(frozen=True)
class RetrievedChunk:
    """One result: the verbatim chunk record plus ranking diagnostics."""

    record: dict  # exactly as stored in the index (text byte-identical to source)
    rank: int  # 1-based, in the returned list
    score: float  # fused RRF score (or method score for single-method searches)
    bm25_rank: int | None  # None = no term evidence for this chunk
    bm25_score: float | None
    embed_rank: int | None
    embed_score: float | None

    @property
    def chunk_id(self) -> str:
        return self.record["chunk_id"]


class Retriever:
    """Hybrid search over a loaded, verified Index."""

    def __init__(self, index: Index, encoder: Encoder) -> None:
        self.index = index
        self.encoder = encoder
        self._bm25 = BM25(
            [tokenize(t) for t in index.enriched],
            k1=index.manifest["params"]["k1"],
            b=index.manifest["params"]["b"],
        )

    # -- filtering --------------------------------------------------------------
    def _eligible_mask(
        self, ticker: str | None, form: str | None, section: str | None
    ) -> np.ndarray:
        mask = np.ones(len(self.index.records), dtype=bool)
        for i, r in enumerate(self.index.records):
            if ticker is not None and r["ticker"] != ticker.upper():
                mask[i] = False
            elif form is not None and r["form"] != form.upper():
                mask[i] = False
            elif section is not None and r["section_key"] != section.lower():
                mask[i] = False
        return mask

    # -- per-method rankings ------------------------------------------------------
    def _bm25_ranking(
        self, query: str, mask: np.ndarray, depth: int
    ) -> list[tuple[int, float]]:
        return self._bm25.top(tokenize(query), depth, eligible=mask)

    def _embed_ranking(
        self, query: str, mask: np.ndarray, depth: int
    ) -> list[tuple[int, float]]:
        q = self.encoder.encode_query(query)
        window_sims = self.index.embeddings @ q  # (n_windows,)
        # Max-pool windows per chunk: a chunk's score is its best window's score, so
        # text beyond the model's reading window is never invisible.
        chunk_scores = np.full(len(self.index.records), -np.inf, dtype=np.float64)
        np.maximum.at(chunk_scores, self.index.window_map, window_sims.astype(np.float64))
        chunk_scores[~mask] = -np.inf
        eligible_idx = np.nonzero(np.isfinite(chunk_scores))[0]
        order = sorted(eligible_idx.tolist(), key=lambda i: (-chunk_scores[i], i))[:depth]
        return [(i, float(chunk_scores[i])) for i in order]

    # -- public API -----------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        k: int = DEFAULT_K,
        ticker: str | None = None,
        form: str | None = None,
        section: str | None = None,
        method: Method = "fused",
        fusion_depth: int = DEFAULT_FUSION_DEPTH,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[RetrievedChunk]:
        """Search the corpus. Deterministic: same inputs → same ranked results.

        Long queries are allowed; the embedding side reads the model's window (~512
        tokens) while BM25 sees every term. Fused scores are rank-based, NOT
        calibrated confidence.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k < 1:
            raise ValueError("k must be >= 1")

        mask = self._eligible_mask(ticker, form, section)
        if not mask.any():
            return []  # filter matched nothing — empty result, not an error

        bm25_ranked = self._bm25_ranking(query, mask, fusion_depth)
        embed_ranked = self._embed_ranking(query, mask, fusion_depth)
        bm25_pos = {doc: (pos + 1, score) for pos, (doc, score) in enumerate(bm25_ranked)}
        embed_pos = {doc: (pos + 1, score) for pos, (doc, score) in enumerate(embed_ranked)}

        if method == "bm25":
            fused = [(doc, score) for doc, score in bm25_ranked]
        elif method == "embed":
            fused = [(doc, score) for doc, score in embed_ranked]
        else:
            fused = rrf_fuse(
                [[doc for doc, _ in bm25_ranked], [doc for doc, _ in embed_ranked]],
                k=rrf_k,
            )

        results = []
        for rank, (doc, score) in enumerate(fused[:k], start=1):
            b = bm25_pos.get(doc)
            e = embed_pos.get(doc)
            results.append(
                RetrievedChunk(
                    record=self.index.records[doc],
                    rank=rank,
                    score=score,
                    bm25_rank=b[0] if b else None,
                    bm25_score=b[1] if b else None,
                    embed_rank=e[0] if e else None,
                    embed_score=e[1] if e else None,
                )
            )
        return results


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid search over parsed chunks")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--ticker")
    parser.add_argument("--form")
    parser.add_argument("--section")
    parser.add_argument("--method", choices=["fused", "bm25", "embed"], default="fused")
    args = parser.parse_args(argv)

    from .encoders import FastEmbedEncoder
    from .index import load_index

    encoder = FastEmbedEncoder()
    index = load_index(expected_model_id=encoder.model_id)
    retriever = Retriever(index, encoder)
    results = retriever.search(
        args.query, k=args.k, ticker=args.ticker, form=args.form,
        section=args.section, method=args.method,
    )
    if not results:
        print("No results (filters matched 0 chunks).")
        return 0
    for r in results:
        rec = r.record
        diag = (
            f"bm25 #{r.bm25_rank}" if r.bm25_rank else "bm25 —",
            f"embed #{r.embed_rank}" if r.embed_rank else "embed —",
        )
        print(
            f"#{r.rank} [{rec['ticker']} {rec['form']} {rec['section_key']}] "
            f"{rec['chunk_id']}  (fused {r.score:.4f}; {diag[0]}, {diag[1]})"
        )
        text = rec["text"]
        print(f"    {text[:220]}{'...' if len(text) > 220 else ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
