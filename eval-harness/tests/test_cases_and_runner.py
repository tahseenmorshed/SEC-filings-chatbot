"""Case loading + runner logic with a fully faked client — no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.cases import CorrectnessCase, RefusalCase, load_case, load_cases
from harness.runner import run_case

CASES_DIR = Path(__file__).resolve().parent.parent / "cases"


# -- the committed case corpus itself ------------------------------------------------
def test_all_committed_cases_load_with_unique_ids():
    cases = load_cases(CASES_DIR)
    assert len(cases) >= 15
    kinds = {type(c).__name__ for c in cases}
    assert kinds == {"CorrectnessCase", "RefusalCase", "FilterCase", "RobustnessCase"}


def test_unknown_type_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("type: nonsense\nid: x\n")
    with pytest.raises(ValueError, match="unknown or missing case type"):
        load_case(p)


def test_bad_fields_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("type: correctness\nid: x\nnot_a_field: 1\n")
    with pytest.raises(ValueError, match="bad fields"):
        load_case(p)


def test_duplicate_ids_rejected(tmp_path):
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text(
            "type: refusal\nid: same\nquestion: q\ncategory: off_corpus\n"
        )
    with pytest.raises(ValueError, match="Duplicate case ids"):
        load_cases(tmp_path)


# -- runner with a fake client -------------------------------------------------------
CHUNK = {"source_sha256": "a" * 64,
         "text": "we had over 2,600 full-time permanent employees worldwide"}

GOOD_ANSWER = {
    "question": "q", "refused": False, "refusal_reason": None, "refusal_detail": None,
    "answer": "About 2,600 people.", "dropped_claims": 0, "ticker_filter": "RKLB",
    "track": "narrative", "model": "m",
    "claims": [{
        "statement": "s", "quote": "over 2,600 full-time permanent employees",
        "chunk_id": "RKLB-1", "ticker": "RKLB", "form": "10-K",
        "filing_date": "2026-02-26", "accession_number": "x", "section_key": "item_1",
        "section_title": "Business", "char_start": 1, "char_end": 99,
        "source_path": "p", "source_sha256": "a" * 64,
    }],
    "retrieved": [{"chunk_id": "RKLB-1", "ticker": "RKLB", "form": "10-K",
                   "section_key": "item_1", "rank": 1, "fused_score": 0.3,
                   "bm25_rank": 1, "embed_rank": 1}],
}

REFUSAL = {"question": "q", "refused": True, "refusal_reason": "model_refusal",
           "refusal_detail": "not in chunks", "answer": None, "claims": [],
           "dropped_claims": 0, "retrieved": [], "ticker_filter": None,
           "track": "narrative", "model": "m"}


class FakeHTTP:
    def __init__(self, ask_payload):
        self._ask = ask_payload

    class R:
        def __init__(self, code, payload):
            self.status_code, self._p = code, payload

        def json(self):
            return self._p

    def ask(self, question, k=None):
        return self.R(200, self._ask)

    def get_chunk(self, chunk_id):
        return self.R(200, CHUNK) if chunk_id == "RKLB-1" else self.R(404, {})

    def search(self, query, **kwargs):
        return self.R(200, {"query": query, "method": "fused", "hits": []})


def test_correctness_case_passes_with_grounded_answer():
    case = CorrectnessCase(id="c", question="q", expected_facts=["2,?600"])
    outcome = run_case(FakeHTTP(GOOD_ANSWER), case)
    assert outcome.hard_pass, [c for c in outcome.checks if not c.ok]


def test_correctness_case_fails_on_refusal():
    case = CorrectnessCase(id="c", question="q", expected_facts=["2,?600"])
    outcome = run_case(FakeHTTP(REFUSAL), case)
    assert not outcome.hard_pass


def test_correctness_fails_on_schema_drift():
    broken = {**GOOD_ANSWER, "refused": "not-a-bool"}
    case = CorrectnessCase(id="c", question="q", expected_facts=["2,?600"])
    outcome = run_case(FakeHTTP(broken), case)
    assert not outcome.hard_pass
    assert any(c.check == "wire_schema" and not c.ok for c in outcome.checks)


def test_refusal_case_passes_on_refusal_and_fails_on_answer():
    case = RefusalCase(id="r", question="q", category="off_corpus")
    assert run_case(FakeHTTP(REFUSAL), case).hard_pass
    assert not run_case(FakeHTTP(GOOD_ANSWER), case).hard_pass


def test_citation_integrity_enforced_even_when_facts_match():
    tampered = {**GOOD_ANSWER,
                "claims": [{**GOOD_ANSWER["claims"][0],
                            "quote": "we had over 2,600 part-time interns"}]}
    case = CorrectnessCase(id="c", question="q", expected_facts=["2,?600"])
    outcome = run_case(FakeHTTP(tampered), case)
    assert not outcome.hard_pass
    assert any(c.check == "quote_verbatim_in_chunk" and not c.ok for c in outcome.checks)
