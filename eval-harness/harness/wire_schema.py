"""Independent redefinition of the target API's wire contract.

Deliberately re-derived from the target's documented behavior (README/ARCHITECTURE),
never imported from its source. If the live API's shape has drifted from what a
consumer would reasonably expect, validating every response against *this* file is
what catches it — validating against the target's own model would only prove the
target agrees with itself.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class VerifiedClaim(BaseModel):
    statement: str
    quote: str
    chunk_id: str
    ticker: str
    form: str
    filing_date: str
    accession_number: str
    section_key: str
    section_title: str
    char_start: int
    char_end: int
    source_path: str
    source_sha256: str


class RetrievedChunkInfo(BaseModel):
    chunk_id: str
    ticker: str
    form: str
    section_key: str
    rank: int
    fused_score: float
    bm25_rank: int | None = None
    embed_rank: int | None = None


class AnswerResponse(BaseModel):
    question: str
    refused: bool
    refusal_reason: (
        Literal["retrieval_empty", "model_refusal", "all_claims_unverified"] | None
    ) = None
    refusal_detail: str | None = None
    answer: str | None = None
    claims: list[VerifiedClaim] = []
    dropped_claims: int = 0
    retrieved: list[RetrievedChunkInfo] = []
    ticker_filter: str | None = None
    track: str = "narrative"
    model: str | None = None


class SearchHit(BaseModel):
    rank: int
    fused_score: float
    bm25_rank: int | None = None
    bm25_score: float | None = None
    embed_rank: int | None = None
    embed_score: float | None = None
    record: dict


class SearchResponse(BaseModel):
    query: str
    method: str
    hits: list[SearchHit]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    chunks: int
    tickers: list[str]
    embedding_model: str
    answer_model: str
