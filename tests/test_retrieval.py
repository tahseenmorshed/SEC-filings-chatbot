"""Index + Retriever end-to-end with a controllable fake encoder (no model, no network).

The KeywordEncoder gives texts sharing tokens a high cosine — enough semantic
structure to test window pooling, fusion behavior, and graceful degradation
deterministically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from filings_analyst.retrieval.bm25 import tokenize
from filings_analyst.retrieval.index import (
    StaleIndexError,
    build_index,
    load_index,
    split_windows,
)
from filings_analyst.retrieval.retriever import Retriever


class KeywordEncoder:
    """Deterministic fake: vector = normalized *binary* bag of hashed tokens.

    Binary (presence, not counts) so a repeated sentence can't inflate a window's
    norm and drown its distinctive tokens; dim 512 keeps hash collisions from
    dominating short documents.
    """

    model_id = "fake-keyword-encoder"
    dim = 512

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in set(tokenize(text)):
            slot = int(hashlib.sha256(tok.encode()).hexdigest()[:8], 16) % self.dim
            v[slot] = 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else np.eye(self.dim, dtype=np.float32)[0]

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def _record(chunk_id, ticker, text, *, form="10-K", section="item_1", title="Business"):
    return {
        "chunk_id": chunk_id, "ticker": ticker, "cik": "0000000001", "form": form,
        "filing_date": "2026-01-01", "report_date": "2025-12-31",
        "accession_number": "0000000001-26-000001", "source_path": "x.htm",
        "source_sha256": "0" * 64, "encoding": "utf-8", "section_key": section,
        "section_title": title, "order": 0, "n_blocks": 1, "oversized": False,
        "char_start": 0, "char_end": len(text), "text": text,
    }


LONG_TAIL_TEXT = ("The satellite program continues on schedule. " * 60) + \
    "Final tail disclosure: the zzztail milestone was reached."  # tail > window 1


@pytest.fixture
def store(tmp_path) -> Path:
    root = tmp_path / "store"
    recs = {
        "AAA": [
            _record("AAA-00000", "AAA", "The Neutron rocket remains in development."),
            _record("AAA-00001", "AAA", "We operate a satellite launch cadence business.",
                    form="10-Q", section="item_1a", title="Risk Factors"),
            _record("AAA-00002", "AAA", LONG_TAIL_TEXT, section="item_7", title="MD&A"),
        ],
        "BBB": [
            _record("BBB-00000", "BBB", "Revenue growth was driven by data center demand."),
        ],
    }
    for ticker, records in recs.items():
        chunks_dir = root / ticker / "chunks"
        chunks_dir.mkdir(parents=True)
        (chunks_dir / "10-K_acc.chunks.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
    facts = root / "AAA" / "facts"
    facts.mkdir(parents=True)
    (facts / "companyfacts.json").write_text(
        json.dumps({"entityName": "Alpha Aerospace, Inc."}), encoding="utf-8"
    )
    return root


@pytest.fixture
def retriever(store, tmp_path) -> Retriever:
    encoder = KeywordEncoder()
    build_index(encoder, store_root=store, index_dir=tmp_path / "index")
    index = load_index(index_dir=tmp_path / "index", store_root=store,
                       expected_model_id=encoder.model_id)
    return Retriever(index, encoder)


# -- provenance and enrichment ----------------------------------------------------
def test_result_text_is_byte_identical_to_source(retriever, store):
    (r,) = retriever.search("Neutron", k=1)
    source_line = next(
        json.loads(line)
        for line in (store / "AAA/chunks/10-K_acc.chunks.jsonl").read_text().splitlines()
        if json.loads(line)["chunk_id"] == r.chunk_id
    )
    assert r.record["text"] == source_line["text"]
    assert r.record == source_line  # full record, not just text


def test_enrichment_is_searchable_but_never_leaks(retriever):
    # "Alpha Aerospace" appears ONLY in the entity name, never in chunk text.
    results = retriever.search("Alpha Aerospace", k=3, method="bm25")
    assert results and all(r.record["ticker"] == "AAA" for r in results)
    assert all("Alpha Aerospace" not in r.record["text"] for r in results)


# -- filters ----------------------------------------------------------------------
def test_ticker_filter_is_absolute(retriever):
    results = retriever.search("revenue growth data center", k=8, ticker="AAA")
    assert results and all(r.record["ticker"] == "AAA" for r in results)


def test_unknown_filter_returns_empty_not_error(retriever):
    assert retriever.search("anything", ticker="ZZZ") == []
    assert retriever.search("anything", form="8-K") == []


def test_form_and_section_filters(retriever):
    results = retriever.search("satellite", k=8, form="10-Q")
    assert results and all(r.record["form"] == "10-Q" for r in results)
    results = retriever.search("satellite", k=8, section="item_1a")
    assert results and all(r.record["section_key"] == "item_1a" for r in results)


# -- staleness ---------------------------------------------------------------------
def test_modified_chunk_file_fails_load(store, tmp_path):
    encoder = KeywordEncoder()
    build_index(encoder, store_root=store, index_dir=tmp_path / "index")
    path = store / "AAA/chunks/10-K_acc.chunks.jsonl"
    path.write_text(path.read_text() + "\n", encoding="utf-8")  # any byte change
    with pytest.raises(StaleIndexError, match="changed"):
        load_index(index_dir=tmp_path / "index", store_root=store)


def test_encoder_model_mismatch_fails_load(store, tmp_path):
    build_index(KeywordEncoder(), store_root=store, index_dir=tmp_path / "index")
    with pytest.raises(StaleIndexError, match="encoder"):
        load_index(index_dir=tmp_path / "index", store_root=store,
                   expected_model_id="some-other-model")


# -- determinism ---------------------------------------------------------------------
def test_same_query_same_results(retriever):
    a = [(r.chunk_id, r.score) for r in retriever.search("satellite launch", k=5)]
    b = [(r.chunk_id, r.score) for r in retriever.search("satellite launch", k=5)]
    assert a == b


# -- degenerate inputs ---------------------------------------------------------------
def test_empty_query_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("")
    with pytest.raises(ValueError):
        retriever.search("   ")


def test_huge_query_does_not_crash(retriever):
    assert retriever.search("satellite " * 1000, k=3)


def test_k_larger_than_corpus_returns_all(retriever):
    results = retriever.search("the satellite rocket revenue", k=50)
    assert 0 < len(results) <= 4


def test_no_term_overlap_degrades_to_embeddings_only(retriever):
    # Query shares no token with any document → BM25 has no evidence.
    results = retriever.search("qqqword wwwword", k=3)
    assert all(r.bm25_rank is None for r in results)


# -- windows (the truncation fix) -----------------------------------------------------
def test_split_windows_covers_whole_text():
    text = "x" * 5000
    windows = split_windows(text, size=1600, stride=1300)
    assert windows[0] == text[:1600]
    assert sum(len(w) > 0 for w in windows) == len(windows)
    # Reconstruct coverage: last window must reach the end of the text.
    assert text[-100:] in windows[-1]
    assert split_windows("short") == ["short"]


def test_query_matching_only_a_chunks_tail_is_retrievable(retriever):
    # "zzztail" appears ONLY past the first embedding window of the long chunk.
    (top,) = retriever.search("zzztail milestone", k=1, method="embed")
    assert top.record["chunk_id"] == "AAA-00002"


# -- fusion behavior end-to-end -------------------------------------------------------
def test_corroborated_chunk_outranks_single_signal(retriever):
    # "Neutron rocket development" matches AAA-00000 both lexically and semantically.
    results = retriever.search("Neutron rocket development", k=3)
    assert results[0].chunk_id == "AAA-00000"
    assert results[0].bm25_rank == 1 and results[0].embed_rank == 1
