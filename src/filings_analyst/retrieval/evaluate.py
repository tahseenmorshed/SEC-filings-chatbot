"""Gold-set evaluation: the permanent regression harness for retrieval quality.

Reads ``eval/gold_set.jsonl`` (committed to the repo; labels found by text-grepping
known answers, never by running the retriever — otherwise labels inherit the
system's blind spots). Reports recall@5 / recall@10 / MRR@10 for BM25-only,
embeddings-only, and fused, plus the non-gated diagnostics (off-corpus decoy score
separation, near-duplicate crowding).

Exit code enforces the hard bars: fused recall@5 ≥ 0.85 AND fused recall@10 = 1.0
AND fused recall@5 ≥ each single method.

Run::

    uv run python -m filings_analyst.retrieval.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

GOLD_SET_PATH = Path("eval/gold_set.jsonl")

RECALL5_BAR = 0.85
CROWDING_SIM = 0.97


def _first_hit_rank(results, acceptable: set[str]) -> int | None:
    for r in results:
        if r.chunk_id in acceptable:
            return r.rank
    return None


def _evaluate_method(retriever, gold: list[dict], method: str) -> dict:
    ranks = []
    misses = []
    for q in gold:
        filters = q.get("filters", {})
        results = retriever.search(q["query"], k=10, method=method, **filters)
        rank = _first_hit_rank(results, set(q["acceptable"]))
        ranks.append(rank)
        if rank is None or rank > 5:
            misses.append((q["qid"], rank))
    n = len(gold)
    return {
        "recall@5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "recall@10": sum(1 for r in ranks if r is not None) / n,
        "mrr@10": sum(1 / r for r in ranks if r is not None) / n,
        "misses": misses,
        "ranks": ranks,
    }


def _chunk_similarity(index, i: int, j: int) -> float:
    """Max cosine between any window of chunk i and any window of chunk j."""
    wi = np.nonzero(index.window_map == i)[0]
    wj = np.nonzero(index.window_map == j)[0]
    sims = index.embeddings[wi] @ index.embeddings[wj].T
    return float(sims.max())


def main() -> int:
    from .encoders import FastEmbedEncoder
    from .index import load_index
    from .retriever import Retriever

    entries = [
        json.loads(line)
        for line in GOLD_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold = [e for e in entries if e["type"] != "off_corpus"]
    decoys = [e for e in entries if e["type"] == "off_corpus"]

    encoder = FastEmbedEncoder()
    index = load_index(expected_model_id=encoder.model_id)
    retriever = Retriever(index, encoder)

    print(f"Gold set: {len(gold)} questions + {len(decoys)} off-corpus decoys\n")

    reports = {m: _evaluate_method(retriever, gold, m) for m in ("bm25", "embed", "fused")}

    print(f"{'method':8} {'recall@5':>9} {'recall@10':>10} {'mrr@10':>8}")
    for m, rep in reports.items():
        print(f"{m:8} {rep['recall@5']:>9.3f} {rep['recall@10']:>10.3f} {rep['mrr@10']:>8.3f}")

    fused = reports["fused"]
    if fused["misses"]:
        print("\nFused misses (rank > 5 or absent from top-10):")
        by_id = {q["qid"]: q for q in gold}
        for qid, rank in fused["misses"]:
            q = by_id[qid]
            print(f"  {qid} rank={rank}: {q['query']!r}")
            for m in ("bm25", "embed"):
                r = reports[m]["ranks"][gold.index(q)]
                print(f"      {m}: rank={r}")

    # Rescue directions: questions where one method missed top-5 but fused hit.
    rescued_by_embed = rescued_by_bm25 = 0
    for i in range(len(gold)):
        f, b, e = (reports[m]["ranks"][i] for m in ("fused", "bm25", "embed"))
        if f is not None and f <= 5:
            if b is None or b > 5:
                rescued_by_embed += 1
            if e is None or e > 5:
                rescued_by_bm25 += 1
    print(f"\nRescues into fused top-5: by embeddings {rescued_by_embed}, by BM25 {rescued_by_bm25}")

    # Off-corpus decoys: report fused top-1 scores vs gold (refusal-layer input).
    if decoys:
        gold_top1 = []
        for q in gold:
            res = retriever.search(q["query"], k=1, **q.get("filters", {}))
            gold_top1.append(res[0].score if res else 0.0)
        decoy_top1 = []
        for q in decoys:
            res = retriever.search(q["query"], k=1)
            decoy_top1.append(res[0].score if res else 0.0)
        print(
            f"Decoy fused top-1 score: median {np.median(decoy_top1):.4f} "
            f"(gold median {np.median(gold_top1):.4f}) — reported, not gated"
        )

    # Near-duplicate crowding in fused top-5 (reported, not gated).
    crowded = 0
    id_of = {r["chunk_id"]: i for i, r in enumerate(index.records)}
    for q in gold:
        results = retriever.search(q["query"], k=5, **q.get("filters", {}))
        idxs = [id_of[r.chunk_id] for r in results]
        if any(
            _chunk_similarity(index, a, b) >= CROWDING_SIM
            for x, a in enumerate(idxs)
            for b in idxs[x + 1 :]
        ):
            crowded += 1
    print(f"Queries with near-duplicate pair (cos ≥ {CROWDING_SIM}) in top-5: {crowded}/{len(gold)}")

    # Determinism spot check.
    q0 = gold[0]["query"]
    same = [
        (r.chunk_id, r.score) for r in retriever.search(q0, k=5)
    ] == [(r.chunk_id, r.score) for r in retriever.search(q0, k=5)]
    print(f"Determinism spot check: {'OK' if same else 'FAILED'}")

    ok = (
        fused["recall@5"] >= RECALL5_BAR
        and fused["recall@10"] == 1.0
        and fused["recall@5"] >= reports["bm25"]["recall@5"]
        and fused["recall@5"] >= reports["embed"]["recall@5"]
        and same
    )
    print(f"\nHARD BARS: {'ALL PASS' if ok else 'FAILED'} "
          f"(fused recall@5 ≥ {RECALL5_BAR}, recall@10 = 1.0, fused ≥ both singles)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
