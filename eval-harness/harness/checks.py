"""Check logic: pure functions, response dict in, a CheckResult out.

Citation integrity is the one check applied to *every* non-refused /ask response,
regardless of case type — it is the harness's independent enforcement of the target's
central claim ("every quote is verified"), re-checked here rather than trusted.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .client import HarnessClient


def _canonical(text: str) -> str:
    """Whitespace-collapsed, unicode-punctuation-folded comparison form.

    Independently reimplemented (not imported) — deliberately mirrors the target's
    own canonicalization rules only insofar as they're the documented, reasonable
    behavior (fold curly quotes/dashes, collapse whitespace); this harness does not
    trust the target's implementation of that rule, only its documented contract.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", " ": " ",
    }))
    return " ".join(text.split())


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    check: str
    detail: str = ""


@dataclass
class CaseOutcome:
    case_id: str
    case_type: str
    hard_pass: bool
    checks: list[CheckResult] = field(default_factory=list)
    latency_s: float = 0.0
    raw: dict | None = None  # last response body, for debugging


# ---------------------------------------------------------------------------------
# Citation integrity — applied to every non-refused /ask response.
# ---------------------------------------------------------------------------------

MIN_QUOTE_CHARS = 15


def check_citation_integrity(client: HarnessClient, response: dict) -> list[CheckResult]:
    """Re-verify every claim independently via GET /chunks/{id} — trust nothing."""
    results: list[CheckResult] = []
    retrieved_ids = {r["chunk_id"] for r in response.get("retrieved", [])}

    for i, claim in enumerate(response.get("claims", [])):
        label = f"claim[{i}] ({claim.get('chunk_id')})"

        if len(claim.get("quote", "").strip()) < MIN_QUOTE_CHARS:
            results.append(CheckResult(False, "quote_min_length", label))
            continue

        if claim["chunk_id"] not in retrieved_ids:
            results.append(CheckResult(
                False, "cites_retrieved_chunk",
                f"{label}: chunk not in this response's own `retrieved` list",
            ))
            continue

        chunk_resp = client.get_chunk(claim["chunk_id"])
        if chunk_resp.status_code != 200:
            results.append(CheckResult(
                False, "chunk_fetchable",
                f"{label}: GET /chunks/{claim['chunk_id']} -> {chunk_resp.status_code}",
            ))
            continue
        chunk = chunk_resp.json()

        if chunk.get("source_sha256") != claim.get("source_sha256"):
            results.append(CheckResult(False, "sha_matches", label))
            continue

        if claim.get("char_start") is None or claim.get("char_end") is None:
            results.append(CheckResult(False, "offsets_present", label))
            continue
        if not (0 <= claim["char_start"] < claim["char_end"]):
            results.append(CheckResult(False, "offsets_well_formed", label))
            continue

        if _canonical(claim["quote"]) not in _canonical(chunk.get("text", "")):
            results.append(CheckResult(
                False, "quote_verbatim_in_chunk",
                f"{label}: quote not found in the fetched chunk's text",
            ))
            continue

        results.append(CheckResult(True, "citation_integrity", label))

    return results


# ---------------------------------------------------------------------------------
# Correctness / robustness — expected facts as case-insensitive regex, ALL required.
# ---------------------------------------------------------------------------------


def check_expected_facts(response: dict, expected_facts: list[str]) -> list[CheckResult]:
    haystack = " ".join(
        [response.get("answer") or ""]
        + [c.get("statement", "") + " " + c.get("quote", "") for c in response.get("claims", [])]
    )
    results = []
    for pattern in expected_facts:
        found = re.search(pattern, haystack, re.IGNORECASE) is not None
        results.append(CheckResult(found, "expected_fact", pattern))
    return results


# ---------------------------------------------------------------------------------
# Refusal — the response must decline, with an optional reason-code check.
# ---------------------------------------------------------------------------------


def check_refusal(response: dict, expected_reason: str | None) -> list[CheckResult]:
    results = [CheckResult(response.get("refused") is True, "refused")]
    if expected_reason is not None and response.get("refused"):
        results.append(CheckResult(
            response.get("refusal_reason") == expected_reason,
            "refusal_reason",
            f"expected {expected_reason!r}, got {response.get('refusal_reason')!r}",
        ))
    return results


# ---------------------------------------------------------------------------------
# Filter — every hit's record must match the requested scope.
# ---------------------------------------------------------------------------------


def check_filter_scope(
    response: dict,
    *,
    expected_ticker: str | None,
    expected_form: str | None,
    expected_section: str | None,
) -> list[CheckResult]:
    results = []
    for i, hit in enumerate(response.get("hits", [])):
        rec = hit["record"]
        if expected_ticker is not None:
            results.append(CheckResult(
                rec.get("ticker") == expected_ticker, "hit_ticker_matches",
                f"hit[{i}]: expected {expected_ticker}, got {rec.get('ticker')}",
            ))
        if expected_form is not None:
            results.append(CheckResult(
                rec.get("form") == expected_form, "hit_form_matches",
                f"hit[{i}]: expected {expected_form}, got {rec.get('form')}",
            ))
        if expected_section is not None:
            results.append(CheckResult(
                rec.get("section_key") == expected_section, "hit_section_matches",
                f"hit[{i}]: expected {expected_section}, got {rec.get('section_key')}",
            ))
    return results
