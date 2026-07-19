"""HTTP API: contract fidelity, validation, and introspection — offline via fakes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from filings_analyst.answering.answerer import Answerer
from filings_analyst.answering.schema import AnswerResponse, DraftAnswer, DraftClaim
from filings_analyst.api.app import AppState, HealthResponse, create_app
from tests.test_answering import (
    CHUNK_TEXT,
    ENTITY_NAMES,
    FakeDrafter,
    FakeResult,
    FakeRetriever,
    _record,
)


def _client(draft: DraftAnswer, results=None) -> TestClient:
    results = results if results is not None else [FakeResult(_record())]
    retriever = FakeRetriever(results)
    # /search passes form/section/method; widen the fake's signature.
    retriever.search = lambda q, *, k, ticker=None, form=None, section=None, method="fused": results  # type: ignore[method-assign]
    answerer = Answerer(
        FakeRetriever(results), FakeDrafter(draft), entity_names=ENTITY_NAMES
    )
    state = AppState(
        answerer=answerer,
        retriever=retriever,
        records_by_id={r.chunk_id: r.record for r in results},
        health=HealthResponse(
            status="ok", chunks=len(results), tickers=sorted(ENTITY_NAMES),
            embedding_model="fake-embed", answer_model="fake-model",
        ),
    )
    return TestClient(create_app(state))


GOOD_DRAFT = DraftAnswer(
    answerable=True,
    answer="About 42,000 employees.",
    claims=[DraftClaim(
        statement="NVIDIA had ~42,000 employees.",
        chunk_id="NVDA-00020",
        quote="approximately 42,000 employees in 38 countries",
    )],
)


def test_health():
    with _client(GOOD_DRAFT) as client:
        data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["tickers"] == ["ASTS", "NVDA", "RKLB"]


def test_ask_returns_exact_answer_response_schema():
    with _client(GOOD_DRAFT) as client:
        response = client.post("/ask", json={"question": "How many employees?"})
    assert response.status_code == 200
    # The wire payload must validate as the answering layer's own schema —
    # byte-level proof the API adds no reshaping.
    parsed = AnswerResponse.model_validate(response.json())
    assert not parsed.refused
    (claim,) = parsed.claims
    assert claim.source_sha256 == "a" * 64
    assert claim.char_start == 100


def test_ask_refusal_travels_over_http():
    draft = DraftAnswer(answerable=False, refusal_reason="not in the chunks")
    with _client(draft) as client:
        data = client.post("/ask", json={"question": "Apple revenue?"}).json()
    assert data["refused"] is True
    assert data["refusal_reason"] == "model_refusal"
    assert data["refusal_detail"] == "not in the chunks"


@pytest.mark.parametrize("body", [
    {"question": ""},
    {"question": "   "},
    {"question": "ok", "k": 0},
    {"question": "ok", "k": 999},
    {},
])
def test_ask_validation_rejects_bad_input(body):
    with _client(GOOD_DRAFT) as client:
        assert client.post("/ask", json=body).status_code == 422


def test_search_returns_hits_with_verbatim_records():
    with _client(GOOD_DRAFT) as client:
        data = client.post("/search", json={"query": "employees", "k": 3}).json()
    (hit,) = data["hits"]
    assert hit["rank"] == 1
    assert hit["record"]["text"] == CHUNK_TEXT  # verbatim chunk record on the wire


def test_chunk_lookup_and_404():
    with _client(GOOD_DRAFT) as client:
        ok = client.get("/chunks/NVDA-00020")
        missing = client.get("/chunks/NOPE-99999")
    assert ok.status_code == 200 and ok.json()["text"] == CHUNK_TEXT
    assert missing.status_code == 404
