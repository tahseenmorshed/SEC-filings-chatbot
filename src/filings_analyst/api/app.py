"""FastAPI application for the Grounded Filings Analyst.

Endpoints (all JSON):

* ``GET  /health``      — liveness + corpus/model summary.
* ``POST /ask``         — the main contract: question → AnswerResponse (verified
  claims with provenance, refusal flags). Exactly the answering layer's schema —
  the API never reshapes it, so the two layers cannot drift.
* ``POST /search``      — raw hybrid retrieval with per-method diagnostics, so the
  harness can evaluate retrieval independently of answering.
* ``GET  /chunks/{id}`` — fetch one chunk record verbatim, so the harness can
  re-verify any cited quote/offsets itself.

Run::

    uv run python -m filings_analyst.api.serve            # binds 127.0.0.1:8000

Startup fails fast if the index is stale (chunks re-parsed since the index build) —
serving stale offsets would violate the provenance guarantee.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from ..answering.answerer import Answerer
from ..answering.schema import AnswerResponse
from ..retrieval.retriever import Retriever


# ---------------------------------------------------------------------------------
# Request/response models (the /ask response model is AnswerResponse, unmodified).
# ---------------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(max_length=2000)
    k: int = Field(default=8, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class SearchRequest(BaseModel):
    query: str = Field(max_length=2000)
    k: int = Field(default=8, ge=1, le=50)
    ticker: str | None = None
    form: str | None = None
    section: str | None = None
    method: Literal["fused", "bm25", "embed"] = "fused"

    @field_validator("query")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class SearchHit(BaseModel):
    rank: int
    fused_score: float
    bm25_rank: int | None
    bm25_score: float | None
    embed_rank: int | None
    embed_score: float | None
    record: dict  # the verbatim chunk record (text byte-identical to the store)


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


# ---------------------------------------------------------------------------------
# App wiring
# ---------------------------------------------------------------------------------


@dataclass
class AppState:
    """Everything the endpoints need, built once (or injected by tests)."""

    answerer: Answerer
    retriever: Retriever
    records_by_id: dict[str, dict]
    health: HealthResponse = field(default=None)  # type: ignore[assignment]


def _build_production_state() -> AppState:
    from ..answering.answerer import OpusDrafter
    from ..config import DEFAULT_ANSWER_MODEL
    from ..retrieval.encoders import FastEmbedEncoder
    from ..retrieval.index import load_index

    encoder = FastEmbedEncoder()
    index = load_index(expected_model_id=encoder.model_id)  # stale check: fail fast
    retriever = Retriever(index, encoder)
    answerer = Answerer(
        retriever, OpusDrafter(), entity_names=index.entity_names
    )
    return AppState(
        answerer=answerer,
        retriever=retriever,
        records_by_id={r["chunk_id"]: r for r in index.records},
        health=HealthResponse(
            status="ok",
            chunks=len(index.records),
            tickers=sorted(index.entity_names),
            embedding_model=index.manifest["model_id"],
            answer_model=DEFAULT_ANSWER_MODEL,
        ),
    )


def create_app(state: AppState | None = None) -> FastAPI:
    """App factory. Tests inject a fake AppState; production builds at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ctx = state if state is not None else _build_production_state()
        yield

    app = FastAPI(
        title="Grounded Filings Analyst",
        description="RAG over SEC filings with machine-verified citations and "
        "first-class refusal. Every claim traces to exact character offsets in a "
        "raw filing.",
        lifespan=lifespan,
    )

    def ctx(request: Request) -> AppState:
        return request.app.state.ctx

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return ctx(request).health

    @app.post("/ask", response_model=AnswerResponse)
    def ask(body: AskRequest, request: Request) -> AnswerResponse:
        return ctx(request).answerer.answer(body.question, k=body.k)

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        results = ctx(request).retriever.search(
            body.query,
            k=body.k,
            ticker=body.ticker,
            form=body.form,
            section=body.section,
            method=body.method,
        )
        return SearchResponse(
            query=body.query,
            method=body.method,
            hits=[
                SearchHit(
                    rank=r.rank,
                    fused_score=r.score,
                    bm25_rank=r.bm25_rank,
                    bm25_score=r.bm25_score,
                    embed_rank=r.embed_rank,
                    embed_score=r.embed_score,
                    record=r.record,
                )
                for r in results
            ],
        )

    @app.get("/chunks/{chunk_id}")
    def get_chunk(chunk_id: str, request: Request) -> dict:
        record = ctx(request).records_by_id.get(chunk_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown chunk_id {chunk_id!r}")
        return record

    return app
