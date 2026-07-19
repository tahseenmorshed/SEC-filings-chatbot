"""Schemas for the answering layer.

Two distinct families, deliberately separated:

* **Draft** models — what the LLM produces, enforced by the API's structured-output
  guarantee (valid JSON by construction, so verification only judges truthfulness,
  never format). These stay minimal: statement + chunk id + verbatim quote.
* **Response** models — the external contract the HTTP API and evaluation harness
  consume. Claims here are *verified* claims: they carry the full provenance chain
  (chunk id → char offsets → source file → sha256) attached by our code, not the LLM.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------------
# Draft (LLM-facing) — kept flat and strict for structured-output enforcement.
# ---------------------------------------------------------------------------------


class DraftClaim(BaseModel):
    statement: str = Field(description="One factual statement, in plain language.")
    chunk_id: str = Field(description="The id of the chunk this statement rests on.")
    quote: str = Field(
        description=(
            "A verbatim quote from that chunk (copied exactly, at least one full "
            "clause) that supports the statement."
        )
    )


class DraftAnswer(BaseModel):
    answerable: bool = Field(
        description="False if the provided chunks cannot answer the question."
    )
    refusal_reason: str | None = Field(
        default=None,
        description="When answerable is false: one sentence on why (e.g. the chunks "
        "cover different companies or topics than the question asks about).",
    )
    answer: str | None = Field(
        default=None,
        description="When answerable: a direct answer to the question, one short "
        "paragraph, using only information present in the claims.",
    )
    claims: list[DraftClaim] = Field(
        default_factory=list,
        description="Every factual statement the answer relies on, each with its "
        "supporting verbatim quote. No claim, no statement.",
    )


# ---------------------------------------------------------------------------------
# Response (API-facing) — the harness contract.
# ---------------------------------------------------------------------------------

RefusalReason = Literal[
    "retrieval_empty",  # filters/search produced no chunks; LLM never called
    "model_refusal",  # the model judged the chunks insufficient
    "all_claims_unverified",  # every drafted claim failed mechanical verification
]


class VerifiedClaim(BaseModel):
    """A claim that passed verification, with the full provenance chain."""

    statement: str
    quote: str
    chunk_id: str
    ticker: str
    form: str
    filing_date: str
    accession_number: str
    section_key: str
    section_title: str
    char_start: int  # chunk's span in the decoded source document
    char_end: int
    source_path: str  # relative to the filing store
    source_sha256: str  # sha of the raw source file the offsets index into


class RetrievedChunkInfo(BaseModel):
    """What retrieval surfaced (diagnostics for the harness; not all get cited)."""

    chunk_id: str
    ticker: str
    form: str
    section_key: str
    rank: int
    fused_score: float
    bm25_rank: int | None
    embed_rank: int | None


class AnswerResponse(BaseModel):
    """The complete, external answer contract."""

    question: str
    refused: bool
    refusal_reason: RefusalReason | None = None
    refusal_detail: str | None = None  # model's own wording, when it refused
    answer: str | None = None
    claims: list[VerifiedClaim] = Field(default_factory=list)
    dropped_claims: int = 0  # drafted claims that failed verification (reported)
    retrieved: list[RetrievedChunkInfo] = Field(default_factory=list)
    ticker_filter: str | None = None  # scoping decision, for transparency
    track: Literal["narrative"] = "narrative"  # XBRL facts track reserved for later
    model: str | None = None  # None when the LLM was never called
