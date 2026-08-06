"""Check logic: pure-function tests with a fake client — no network, no target."""

from __future__ import annotations

from harness.checks import (
    _canonical,
    check_citation_integrity,
    check_expected_facts,
    check_filter_scope,
    check_refusal,
)

CHUNK_TEXT = "As of December 31, 2025, we had over 2,600 full-time permanent employees worldwide."


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, chunks: dict[str, dict]):
        self._chunks = chunks
        self.fetched: list[str] = []

    def get_chunk(self, chunk_id: str) -> FakeResponse:
        self.fetched.append(chunk_id)
        if chunk_id in self._chunks:
            return FakeResponse(200, self._chunks[chunk_id])
        return FakeResponse(404)


def _response(claims, retrieved_ids):
    return {
        "refused": False,
        "answer": "x",
        "claims": claims,
        "retrieved": [{"chunk_id": cid} for cid in retrieved_ids],
    }


def _claim(chunk_id="RKLB-1", quote="we had over 2,600 full-time permanent employees",
           sha="a" * 64, start=10, end=90):
    return {"chunk_id": chunk_id, "quote": quote, "source_sha256": sha,
            "char_start": start, "char_end": end, "statement": "s"}


def _chunk(sha="a" * 64, text=CHUNK_TEXT):
    return {"source_sha256": sha, "text": text}


# -- canonical ---------------------------------------------------------------------
def test_canonical_folds_unicode_and_whitespace():
    assert _canonical("The  Company’s “plan” — grew") == 'The Company\'s "plan" - grew'
    assert _canonical(_canonical("a  b")) == _canonical("a  b")


# -- citation integrity -------------------------------------------------------------
def test_verbatim_claim_passes_end_to_end():
    client = FakeClient({"RKLB-1": _chunk()})
    results = check_citation_integrity(client, _response([_claim()], ["RKLB-1"]))
    assert [r.ok for r in results] == [True]
    assert client.fetched == ["RKLB-1"]  # independently fetched, not trusted


def test_fabricated_quote_fails():
    client = FakeClient({"RKLB-1": _chunk()})
    results = check_citation_integrity(
        client, _response([_claim(quote="we had over 9,999 part-time employees")], ["RKLB-1"])
    )
    assert results[0].check == "quote_verbatim_in_chunk" and not results[0].ok


def test_unicode_drift_forgiven_in_quote():
    client = FakeClient({"RKLB-1": _chunk(text="The Company’s employees — worldwide grew.")})
    results = check_citation_integrity(
        client, _response([_claim(quote="The Company's employees - worldwide grew.")], ["RKLB-1"])
    )
    assert results[0].ok


def test_citing_unretrieved_chunk_fails_without_fetch():
    client = FakeClient({"RKLB-1": _chunk()})
    results = check_citation_integrity(client, _response([_claim()], ["OTHER-2"]))
    assert results[0].check == "cites_retrieved_chunk" and not results[0].ok
    assert client.fetched == []


def test_sha_mismatch_fails():
    client = FakeClient({"RKLB-1": _chunk(sha="b" * 64)})
    results = check_citation_integrity(client, _response([_claim()], ["RKLB-1"]))
    assert results[0].check == "sha_matches" and not results[0].ok


def test_unfetchable_chunk_fails():
    client = FakeClient({})
    results = check_citation_integrity(client, _response([_claim()], ["RKLB-1"]))
    assert results[0].check == "chunk_fetchable" and not results[0].ok


def test_short_quote_fails():
    client = FakeClient({"RKLB-1": _chunk()})
    results = check_citation_integrity(client, _response([_claim(quote="2,600")], ["RKLB-1"]))
    assert results[0].check == "quote_min_length" and not results[0].ok


def test_malformed_offsets_fail():
    client = FakeClient({"RKLB-1": _chunk()})
    results = check_citation_integrity(
        client, _response([_claim(start=90, end=10)], ["RKLB-1"])
    )
    assert results[0].check == "offsets_well_formed" and not results[0].ok


# -- expected facts -----------------------------------------------------------------
def test_expected_facts_found_in_answer_or_claims():
    response = {"answer": "About 2,600 people.", "claims": [
        {"statement": "", "quote": "over 2,600 full-time permanent employees"}]}
    results = check_expected_facts(response, ["2,?600", "full-time"])
    assert all(r.ok for r in results)


def test_expected_fact_missing_fails():
    results = check_expected_facts({"answer": "no idea", "claims": []}, ["42,?000"])
    assert not results[0].ok


# -- refusal ------------------------------------------------------------------------
def test_refusal_pass_and_reason_pin():
    response = {"refused": True, "refusal_reason": "model_refusal"}
    assert all(r.ok for r in check_refusal(response, "model_refusal"))
    assert not check_refusal(response, "retrieval_empty")[1].ok


def test_non_refusal_fails_refusal_check():
    assert not check_refusal({"refused": False}, None)[0].ok


# -- filter scope -------------------------------------------------------------------
def test_filter_scope_flags_wrong_ticker():
    response = {"hits": [
        {"record": {"ticker": "ASTS", "form": "10-K", "section_key": "item_1a"}},
        {"record": {"ticker": "NVDA", "form": "10-K", "section_key": "item_1a"}},
    ]}
    results = check_filter_scope(response, expected_ticker="ASTS",
                                 expected_form=None, expected_section=None)
    assert [r.ok for r in results] == [True, False]
