"""Answering layer: verification gate, refusal paths, scoping — all offline.

A fake drafter stands in for the LLM, so every path (including the ones that depend
on the model misbehaving) is exercised deterministically with no API key.
"""

from __future__ import annotations

import pytest

from filings_analyst.answering.answerer import Answerer
from filings_analyst.answering.schema import AnswerResponse, DraftAnswer, DraftClaim
from filings_analyst.answering.scoping import scope_ticker
from filings_analyst.answering.verify import canonical, verify_quote

ENTITY_NAMES = {
    "NVDA": "NVIDIA CORP",
    "RKLB": "Rocket Lab USA, Inc.",
    "ASTS": "AST SpaceMobile, Inc.",
}

CHUNK_TEXT = (
    "As of the end of fiscal year 2026, we had approximately 42,000 employees in 38 "
    "countries; 31,000 were engaged in research and development."
)


def _record(chunk_id="NVDA-00020", text=CHUNK_TEXT, ticker="NVDA"):
    return {
        "chunk_id": chunk_id, "ticker": ticker, "form": "10-K",
        "filing_date": "2026-02-25", "report_date": "2026-01-25",
        "accession_number": "0001045810-26-000021", "source_path": "NVDA/x.htm",
        "source_sha256": "a" * 64, "encoding": "utf-8", "section_key": "item_1",
        "section_title": "Business", "order": 0, "n_blocks": 1, "oversized": False,
        "char_start": 100, "char_end": 250, "text": text,
    }


class FakeResult:
    def __init__(self, record, rank=1):
        self.record = record
        self.rank = rank
        self.score = 0.5
        self.bm25_rank = 1
        self.bm25_score = 2.5
        self.embed_rank = 1
        self.embed_score = 0.8

    @property
    def chunk_id(self):
        return self.record["chunk_id"]


class FakeRetriever:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, question, *, k, ticker=None):
        self.calls.append({"question": question, "k": k, "ticker": ticker})
        return self._results


class FakeDrafter:
    model_id = "fake-model"

    def __init__(self, draft):
        self._draft = draft
        self.called = False

    def draft(self, question, chunks):
        self.called = True
        return self._draft


def _answerer(results, draft):
    return Answerer(
        FakeRetriever(results), FakeDrafter(draft), entity_names=ENTITY_NAMES
    )


# -- verification ----------------------------------------------------------------
def test_verbatim_quote_verifies():
    chunks = {"c1": {"text": CHUNK_TEXT}}
    assert verify_quote("approximately 42,000 employees in 38 countries", "c1", chunks).ok


def test_unicode_and_whitespace_drift_is_forgiven_but_words_are_not():
    chunks = {"c1": {"text": "The Company’s backlog — as defined — grew."}}
    # Straight apostrophe + hyphen versions of the same words: verifies.
    assert verify_quote("The Company's backlog - as defined - grew.", "c1", chunks).ok
    # A changed word: rejected.
    assert not verify_quote("The Company's backlog - as defined - shrank.", "c1", chunks).ok


def test_fabricated_quote_rejected():
    chunks = {"c1": {"text": CHUNK_TEXT}}
    result = verify_quote("we had approximately 99,000 employees worldwide", "c1", chunks)
    assert not result.ok and result.reason == "quote_not_found"


def test_short_quote_rejected():
    chunks = {"c1": {"text": CHUNK_TEXT}}
    result = verify_quote("42,000", "c1", chunks)
    assert not result.ok and result.reason == "quote_too_short"


def test_unknown_chunk_rejected():
    result = verify_quote("approximately 42,000 employees in 38 countries", "nope", {})
    assert not result.ok and result.reason == "unknown_chunk"


def test_canonical_is_deterministic_and_idempotent():
    s = "  “Curly”  —  text ’s  "
    assert canonical(s) == canonical(s)
    assert canonical(canonical(s)) == canonical(s)


# -- pipeline: happy path ---------------------------------------------------------
def test_verified_claims_reach_response_with_provenance():
    draft = DraftAnswer(
        answerable=True,
        answer="NVIDIA has about 42,000 employees.",
        claims=[DraftClaim(
            statement="NVIDIA had ~42,000 employees at the end of FY2026.",
            chunk_id="NVDA-00020",
            quote="we had approximately 42,000 employees in 38 countries",
        )],
    )
    response = _answerer([FakeResult(_record())], draft).answer(
        "How many employees does NVIDIA have?"
    )
    assert not response.refused
    (claim,) = response.claims
    assert claim.source_sha256 == "a" * 64
    assert claim.char_start == 100 and claim.char_end == 250
    assert claim.accession_number == "0001045810-26-000021"
    assert response.dropped_claims == 0
    assert response.model == "fake-model"


def test_response_round_trips_as_json():
    draft = DraftAnswer(answerable=True, answer="x", claims=[DraftClaim(
        statement="s", chunk_id="NVDA-00020",
        quote="approximately 42,000 employees in 38 countries")])
    response = _answerer([FakeResult(_record())], draft).answer("q")
    restored = AnswerResponse.model_validate_json(response.model_dump_json())
    assert restored == response


# -- pipeline: the gate -----------------------------------------------------------
def test_fabricated_claim_dropped_but_good_one_kept():
    draft = DraftAnswer(
        answerable=True, answer="x",
        claims=[
            DraftClaim(statement="good", chunk_id="NVDA-00020",
                       quote="31,000 were engaged in research and development"),
            DraftClaim(statement="bad", chunk_id="NVDA-00020",
                       quote="we doubled headcount to 84,000 employees overnight"),
        ],
    )
    response = _answerer([FakeResult(_record())], draft).answer("q")
    assert not response.refused
    assert [c.statement for c in response.claims] == ["good"]
    assert response.dropped_claims == 1


def test_all_claims_failing_becomes_refusal():
    draft = DraftAnswer(
        answerable=True, answer="confident nonsense",
        claims=[DraftClaim(statement="bad", chunk_id="NVDA-00020",
                           quote="a sentence that appears nowhere in the chunk")],
    )
    response = _answerer([FakeResult(_record())], draft).answer("q")
    assert response.refused and response.refusal_reason == "all_claims_unverified"
    assert response.claims == [] and response.answer is None
    assert response.dropped_claims == 1


def test_citing_unretrieved_chunk_is_dropped():
    draft = DraftAnswer(
        answerable=True, answer="x",
        claims=[DraftClaim(statement="s", chunk_id="SOME-OTHER-CHUNK",
                           quote="approximately 42,000 employees in 38 countries")],
    )
    response = _answerer([FakeResult(_record())], draft).answer("q")
    assert response.refused and response.refusal_reason == "all_claims_unverified"


# -- pipeline: refusals ------------------------------------------------------------
def test_empty_retrieval_refuses_without_calling_llm():
    drafter = FakeDrafter(DraftAnswer(answerable=True))
    answerer = Answerer(FakeRetriever([]), drafter, entity_names=ENTITY_NAMES)
    response = answerer.answer("anything")
    assert response.refused and response.refusal_reason == "retrieval_empty"
    assert drafter.called is False
    assert response.model is None


def test_model_refusal_is_honored():
    draft = DraftAnswer(answerable=False, refusal_reason="chunks are about rockets, not fruit")
    response = _answerer([FakeResult(_record())], draft).answer("Apple's revenue?")
    assert response.refused and response.refusal_reason == "model_refusal"
    assert response.refusal_detail == "chunks are about rockets, not fruit"


# -- scoping -----------------------------------------------------------------------
def test_scoping_matches_company_names_and_tickers():
    assert scope_ticker("How many people work at Rocket Lab?", ENTITY_NAMES) == "RKLB"
    assert scope_ticker("what does NVIDIA say about Blackwell", ENTITY_NAMES) == "NVDA"
    assert scope_ticker("ASTS launch plans", ENTITY_NAMES) == "ASTS"
    assert scope_ticker("AST SpaceMobile satellites", ENTITY_NAMES) == "ASTS"


def test_scoping_returns_none_when_ambiguous_or_absent():
    assert scope_ticker("compare NVIDIA and Rocket Lab", ENTITY_NAMES) is None
    assert scope_ticker("what is the revenue trend?", ENTITY_NAMES) is None


def test_scoping_filter_is_passed_to_retrieval():
    drafter = FakeDrafter(DraftAnswer(answerable=False, refusal_reason="n/a"))
    retriever = FakeRetriever([FakeResult(_record())])
    Answerer(retriever, drafter, entity_names=ENTITY_NAMES).answer(
        "How many people work at Rocket Lab?"
    )
    assert retriever.calls[0]["ticker"] == "RKLB"
