"""Mechanical claim verification — the gate nothing unverified passes.

A claim survives only if:

1. its cited chunk is one of the chunks actually retrieved for this question
   (the model cannot cite evidence it was never shown);
2. its quote is long enough to be meaningful (a three-word fragment could "verify"
   almost any statement); and
3. the quote appears inside the cited chunk's text under **canonical comparison** —
   whitespace runs collapsed and unicode punctuation variants (curly quotes, en/em
   dashes) folded to ASCII, case preserved. This forgives the transcription drift
   LLMs exhibit (a curly apostrophe becoming straight) without ever forgiving a
   changed word, number, or meaning.

Verification is pure and deterministic: same inputs, same verdict, no model involved.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..parsing.extract import normalize_text

# Unicode punctuation variants folded to ASCII for comparison only (displayed text
# keeps the original). Covers what LLMs most often transcribe differently.
_CANON_TRANSLATION = str.maketrans({
    "’": "'",  # right single quote
    "‘": "'",  # left single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "—": "-",  # em dash
    "–": "-",  # en dash
    " ": " ",  # non-breaking space (normalize_text also catches this)
})

MIN_QUOTE_CHARS = 15


def canonical(text: str) -> str:
    """Comparison form: whitespace-normalized, punctuation-folded, case preserved."""
    return normalize_text(text).translate(_CANON_TRANSLATION)


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    reason: str | None = None  # "unknown_chunk" | "quote_too_short" | "quote_not_found"


def verify_quote(quote: str, chunk_id: str, chunks_by_id: dict[str, dict]) -> VerificationResult:
    """Verify one drafted claim's evidence against the retrieved chunks."""
    chunk = chunks_by_id.get(chunk_id)
    if chunk is None:
        return VerificationResult(False, "unknown_chunk")
    if len(quote.strip()) < MIN_QUOTE_CHARS:
        return VerificationResult(False, "quote_too_short")
    if canonical(quote) not in canonical(chunk["text"]):
        return VerificationResult(False, "quote_not_found")
    return VerificationResult(True)
