"""The Answerer: orchestrates scope → retrieve → draft → verify → respond.

The LLM (a ``Drafter``) is injected behind a one-method protocol, so the entire
pipeline — including every verification and refusal path — is testable offline with a
fake drafter and no API key. The production drafter uses the Anthropic API with
structured outputs (the response is schema-valid by construction).

Try it::

    uv run python -m filings_analyst.answering.answerer "How many people work at Rocket Lab?"
"""

from __future__ import annotations

import sys
from typing import Protocol

from ..config import DEFAULT_ANSWER_MODEL
from ..retrieval.retriever import Retriever
from .schema import (
    AnswerResponse,
    DraftAnswer,
    RetrievedChunkInfo,
    VerifiedClaim,
)
from .verify import verify_quote

DEFAULT_K = 8

# The hard rules. Kept stable (cache-friendly) and unambiguous: the model's only
# evidence universe is the chunks in the user turn.
SYSTEM_PROMPT = """\
You are a filings analyst answering questions about SEC filings.

You will receive a question and a set of numbered chunks quoted from real SEC filings.
The chunks are your ONLY source of truth. Rules, in order of importance:

1. Never state anything the chunks do not support. No outside knowledge, no guesses,
   no filling gaps — even when you are confident you know the answer.
2. If the chunks cannot answer the question, set answerable=false and explain briefly
   in refusal_reason. Refusing is a correct outcome, not a failure.
3. Every factual statement in your answer must appear in `claims`, each citing the
   chunk_id it rests on and a supporting quote copied VERBATIM from that chunk —
   exact characters, at least one full clause. Quotes are machine-checked; any
   altered quote causes the claim to be discarded.
4. Answer the question directly and concisely. Prefer fewer, well-supported claims
   over many weak ones.
"""


class Drafter(Protocol):
    """What the pipeline requires of an LLM backend."""

    model_id: str

    def draft(self, question: str, chunks: list[dict]) -> DraftAnswer: ...


class OpusDrafter:
    """Production drafter: Anthropic structured outputs against the pinned model."""

    def __init__(self, model_id: str = DEFAULT_ANSWER_MODEL) -> None:
        import anthropic  # lazy: offline tests never import the SDK

        self.model_id = model_id
        self._client = anthropic.Anthropic()

    def draft(self, question: str, chunks: list[dict]) -> DraftAnswer:
        blocks = []
        for c in chunks:
            blocks.append(
                f"[{c['chunk_id']}] {c['ticker']} {c['form']} "
                f"filed {c['filing_date']} — {c['section_title']}\n{c['text']}"
            )
        user_message = (
            f"Question: {question}\n\n"
            f"Chunks:\n\n" + "\n\n---\n\n".join(blocks)
        )
        response = self._client.messages.parse(
            model=self.model_id,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            output_format=DraftAnswer,
        )
        if response.parsed_output is None:
            # Safety-classifier refusal or truncation: treat as a model refusal
            # rather than crash — the caller maps this to a refusal response.
            return DraftAnswer(
                answerable=False,
                refusal_reason=f"model returned no parseable answer "
                f"(stop_reason={response.stop_reason})",
            )
        return response.parsed_output


class Answerer:
    """The grounded answering pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        drafter: Drafter,
        *,
        entity_names: dict[str, str],
        k: int = DEFAULT_K,
    ) -> None:
        self.retriever = retriever
        self.drafter = drafter
        self.entity_names = entity_names
        self.k = k

    def answer(self, question: str, *, k: int | None = None) -> AnswerResponse:
        from .scoping import scope_ticker

        ticker = scope_ticker(question, self.entity_names)
        results = self.retriever.search(question, k=k or self.k, ticker=ticker)

        retrieved_info = [
            RetrievedChunkInfo(
                chunk_id=r.chunk_id,
                ticker=r.record["ticker"],
                form=r.record["form"],
                section_key=r.record["section_key"],
                rank=r.rank,
                fused_score=r.score,
                bm25_rank=r.bm25_rank,
                embed_rank=r.embed_rank,
            )
            for r in results
        ]

        if not results:
            # No evidence universe at all — refuse without spending an LLM call.
            return AnswerResponse(
                question=question,
                refused=True,
                refusal_reason="retrieval_empty",
                ticker_filter=ticker,
            )

        chunks = [
            {
                "chunk_id": r.chunk_id,
                "ticker": r.record["ticker"],
                "form": r.record["form"],
                "filing_date": r.record["filing_date"],
                "section_title": r.record["section_title"],
                "text": r.record["text"],
            }
            for r in results
        ]
        draft = self.drafter.draft(question, chunks)

        if not draft.answerable:
            return AnswerResponse(
                question=question,
                refused=True,
                refusal_reason="model_refusal",
                refusal_detail=draft.refusal_reason,
                retrieved=retrieved_info,
                ticker_filter=ticker,
                model=self.drafter.model_id,
            )

        # The gate: only claims that pass mechanical verification are ever attached.
        records_by_id = {r.chunk_id: r.record for r in results}
        verified: list[VerifiedClaim] = []
        dropped = 0
        for claim in draft.claims:
            result = verify_quote(claim.quote, claim.chunk_id, records_by_id)
            if not result.ok:
                dropped += 1
                continue
            rec = records_by_id[claim.chunk_id]
            verified.append(
                VerifiedClaim(
                    statement=claim.statement,
                    quote=claim.quote,
                    chunk_id=claim.chunk_id,
                    ticker=rec["ticker"],
                    form=rec["form"],
                    filing_date=rec["filing_date"],
                    accession_number=rec["accession_number"],
                    section_key=rec["section_key"],
                    section_title=rec["section_title"],
                    char_start=rec["char_start"],
                    char_end=rec["char_end"],
                    source_path=rec["source_path"],
                    source_sha256=rec["source_sha256"],
                )
            )

        if not verified:
            # The model answered but nothing survived verification — the grounded
            # promise says we refuse rather than show unbacked prose.
            return AnswerResponse(
                question=question,
                refused=True,
                refusal_reason="all_claims_unverified",
                dropped_claims=dropped,
                retrieved=retrieved_info,
                ticker_filter=ticker,
                model=self.drafter.model_id,
            )

        return AnswerResponse(
            question=question,
            refused=False,
            answer=draft.answer,
            claims=verified,
            dropped_claims=dropped,
            retrieved=retrieved_info,
            ticker_filter=ticker,
            model=self.drafter.model_id,
        )


def build_default_answerer(k: int = DEFAULT_K) -> Answerer:
    """Wire the production pipeline (loaded index, real retriever, Opus drafter)."""
    from ..retrieval.encoders import FastEmbedEncoder
    from ..retrieval.index import load_index

    encoder = FastEmbedEncoder()
    index = load_index(expected_model_id=encoder.model_id)
    retriever = Retriever(index, encoder)
    return Answerer(
        retriever,
        OpusDrafter(),
        entity_names=index.entity_names,
        k=k,
    )


def _main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m filings_analyst.answering.answerer \"question\"",
              file=sys.stderr)
        return 2
    answerer = build_default_answerer()
    response = answerer.answer(" ".join(argv))

    if response.refused:
        print(f"REFUSED ({response.refusal_reason})")
        if response.refusal_detail:
            print(f"  {response.refusal_detail}")
        return 0
    print(response.answer)
    print()
    for i, c in enumerate(response.claims, 1):
        print(f"[{i}] {c.statement}")
        print(f"    \"{c.quote}\"")
        print(f"    — {c.ticker} {c.form} ({c.filing_date}), {c.section_title}, "
              f"{c.source_path} chars {c.char_start}–{c.char_end}")
    if response.dropped_claims:
        print(f"\n({response.dropped_claims} claim(s) failed verification and were dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
