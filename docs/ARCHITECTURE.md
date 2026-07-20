# Grounded Filings Analyst — Architecture & Engineering Record

> The README is the user manual. This document is the engineering record: what was
> built, why each decision went the way it did, what every file does, what every test
> guards, where the gaps are, and how a request flows through the system. It will be
> extended after the external evaluation harness is built.

---

## 1. Product description

The Grounded Filings Analyst answers natural-language questions about SEC filings
(10-K/10-Q narrative and, later, XBRL financial facts) with a hard property: **it
never presents an ungrounded statement.** Every claim in every answer is backed by a
verbatim quote that our own code has mechanically verified against a chunk of a real
filing; every chunk carries exact character offsets into a raw source document stored
byte-for-byte as the SEC served it; the document's sha256 seals the chain. When the
system cannot ground an answer, it refuses — with a machine-readable reason — rather
than guesses.

It is a portfolio project whose second purpose is demonstrating production-grade
engineering judgment around LLMs: measured guarantees over vibes, deterministic
components wherever possible, right-sized tooling (no vector DB at 679 chunks, no
framework where 15 lines of arithmetic suffice), and honest documentation of gaps.

A separate, language-agnostic **evaluation harness** (not yet built) will test the
system black-box over its HTTP API.

## 2. The provenance chain — the one guarantee, layer by layer

```
answer → claim → verbatim quote → chunk → char offsets → raw file → sha256
```

| Link | Enforced by | Mechanism |
|---|---|---|
| raw file is exactly what SEC served | ingestion | stored byte-for-byte, sha256 recorded at fetch; parse refuses on hash mismatch |
| offsets reproduce chunk text | parsing | round-trip invariant checked for 100% of blocks + chunks at build time; build aborts on any failure |
| search serves current chunks | retrieval | index manifest fingerprints source chunk files; load refuses on drift |
| retrieved text is verbatim | retrieval | records copied line-for-line; enriched search representation never leaks into results (tested) |
| quote actually in cited chunk | answering | canonical substring verification, pure code, no LLM; unverified claims dropped; empty → refusal |
| external verifiability | api | `/chunks/{id}` returns verbatim records so any client can re-run the quote check |

The one deliberately non-deterministic component is the LLM draft. The guarantee is
therefore *not* "same answer every time" — it is "**never an unverified statement**",
and that gate is deterministic.

## 3. Design decision log

Numbered, with the alternatives considered and why the choice went the way it did.

### Ingestion

1. **Single SEC client chokepoint.** All EDGAR traffic flows through `SECClient`
   (User-Agent, throttle, cache, retry). Alternative: per-module HTTP calls.
   Rejected because SEC etiquette failures (silent IP blocks) must be structurally
   impossible, not conventions.
2. **Spacing rate limiter, not token bucket.** Requests are forced ≥1/8s apart, so
   the instantaneous rate can never burst past the limit even at startup. A token
   bucket permits exactly the bursts we must avoid. Clock/sleep injectable → tests
   run on a virtual clock.
3. **Cache stores raw bytes verbatim** (no decode/normalize), keyed by URL hash,
   atomic writes. This is the provenance anchor: later offsets are only meaningful
   against unmodified bytes.
4. **Store ≠ cache.** `data/cache/` is opaque transport dedupe; `data/store/` is the
   browsable, ticker-keyed corpus later stages read. Bytes land in both on purpose.
5. **Primary document by SEC designation.** A filing is a folder of dozens of
   sub-documents; we take the SEC's own `primaryDocument` field rather than guessing.
6. **Exact form match** (`10-K`, not `10-K/A`): amendments supersede parts of the
   original — reconciling that is a parsing/answering concern, deferred; at fetch we
   want the canonical original.
7. **Fail-loud User-Agent validation at construction** — the empty-default trick
   makes the actionable message fire even when the env var is entirely unset.

### Parsing

8. **`html.parser` streaming events, not BeautifulSoup/lxml.** Tree-building
   libraries extract clean text but discard source positions — the one thing we
   cannot lose. The stdlib streaming parser reports each token's position; we do our
   own (small) assembly. Zero new dependencies.
9. **"Next event ends the previous one" offset bookkeeping.** Entities decode to
   fewer chars than their raw spelling (`&#8220;` → one char), so arithmetic on
   decoded lengths drifts. All offset math stays in raw coordinates.
10. **Chunk provenance = raw envelope `(char_start, char_end)` + deterministic
    canonical extraction.** Alternatives: per-character index maps (heavy) or
    offsets into a derived clean document (weakens provenance to a file we made up).
    The invariant `canonical_strip(raw[start:end]) == text` is self-validating: any
    bookkeeping bug anywhere fails verification loudly at build time.
11. **Suppression stack** for `ix:header` (168 KB of XBRL machine data in a real
    filing), `ix:hidden`, `display:none`, script/style. Inline suppression does not
    split blocks (hidden spans mid-sentence just vanish).
12. **Section detection via weighted longest-increasing-subsequence.** "Item 1A"
    appears in the TOC, cross-references, and the real heading. Candidates
    (block-start-anchored, short, non-table) compete as a maximum-weight chain in
    canonical item order — a false early heading would block 22 later true ones, so
    the all-true chain wins automatically. Crucially: **section labels are metadata,
    not provenance** — a mislabel can never corrupt an offset.
13. **Chunking packs whole blocks only** (target ~3000 chars, never across
    sections). No mid-block splitting: that would need offset math on normalized
    text. Empirically the largest real paragraph is ~2.2k chars, so splitting logic
    would be speculative complexity.
14. **Tables excluded from chunks** (preserved at block level). Linearized cells
    pollute retrieval; numbers-of-record belong to the XBRL track.
15. **Deterministic outputs, no timestamps** in chunk files → re-runs are
    byte-identical and diffable.

### Retrieval

16. **Hybrid BM25 + embeddings, always both, rank-fused.** Each covers the other's
    blind spot (exact identifiers vs. paraphrase). No query routing/classification —
    both run on every query and RRF lets whichever has signal win. Staged
    retrieve-then-rerank rejected: whatever stage 1 drops, stage 2 never sees.
17. **Hand-rolled BM25 (~100 lines) over `rank_bm25`.** The real correctness risk is
    tokenization (keeping `10-K`, `BB6` intact), which must be custom anyway; the
    formula is published math testable against hand-computed fixtures. No stopwords
    (IDF already down-weights), no stemming (embeddings cover morphology).
18. **Local pinned ONNX embeddings** (bge-small-en-v1.5 via fastembed) over API
    embeddings or torch: free, offline, reproducible, no key in the retrieval path,
    ~140 MB total footprint vs. ~1.5 GB for torch.
19. **No vector database.** 679 chunks × 384 dims ≈ 1 MB; exact brute-force numpy is
    sub-millisecond and *exact*. A vector DB here would add dependencies and
    approximation for zero benefit.
20. **Enriched index representation, verbatim citation text.** Filings say "we",
    never "ASTS", so the *indexed* text is prefixed with ticker/company/form/section
    while the *stored and cited* text stays untouched. Leakage is blocked by test.
21. **Multi-window embeddings with max-pooling.** The model reads ~512 tokens but
    median chunk ≈ 650 tokens — a discovered correctness issue: without windows the
    tail third of most chunks is invisible to semantic search. Chunks embed as 1–5
    overlapping windows (header on each); a chunk scores as its best window.
22. **Window geometry 800/600 and RRF K=5 chosen by transparent grid sweep** on the
    gold set, reported in full (no silent tuning). K=60 (the literature default)
    demonstrably swamped single-method #1 hits at this corpus size.
23. **Evidence gating**: BM25 only ranks chunks with score > 0 — ranking hundreds of
    zero-score ties would inject noise votes into fusion. Paraphrase queries degrade
    gracefully to embeddings-only.
24. **Deterministic everything**: ties break by chunk id; same query → identical
    ranking, every run (tested).
25. **Gold-set discipline**: labels found by text-grepping known answers, never by
    running the retriever (else labels inherit the system's blind spots). Label
    corrections during iteration were made only after reading the chunks, and are
    documented in the gold-set notes.
26. **Stale-index refusal**: manifest records sha256 of every source chunk file;
    load fails with an actionable message on any drift. Serving stale offsets would
    silently break the provenance chain — the failure mode this project exists to
    prevent.

### Answering

27. **Paraphrase claims, verbatim quotes.** Forcing fully-verbatim answers reads
    robotically; free paraphrase judged by a second LLM puts a fuzzy judge at the
    gate. Middle path: claims phrased naturally, each carrying a verbatim quote, and
    only the quote is machine-verified.
28. **Canonical quote comparison forgives transcription drift, never meaning.**
    Whitespace collapsed, curly quotes/dashes folded to ASCII, case preserved. A
    changed word or number always fails (tested: "grew"→"shrank"). Minimum quote
    length (15 chars) prevents trivial fragments from "verifying" anything.
29. **Structured outputs, not JSON-by-politeness.** The API enforces the draft
    schema, so malformed-JSON failure modes don't exist; verification only ever
    judges truthfulness.
30. **Refusal as first-class outcome** with reason codes (`retrieval_empty` — LLM
    never called; `model_refusal` — model's own judgment, wording preserved;
    `all_claims_unverified` — the gate stripped everything).
31. **Deterministic scoping** (company-name matching from index entity names)
    instead of LLM query parsing: one company matched → ticker filter; ambiguous →
    no filter, enrichment carries identity.
32. **The LLM is injected behind a one-method protocol** (`Drafter`), so the entire
    pipeline including every refusal path tests offline with no key. Model pinned in
    config (`claude-opus-4-8`).

### API

33. **Thin by construction**: `/ask` serves the answering layer's `AnswerResponse`
    *unmodified* (a test validates the wire payload against that very schema), so the
    layers cannot drift.
34. **`/chunks/{id}` exists for independent verification** — an external harness can
    re-check any quote with nothing but HTTP.
35. **App factory + injectable state** for offline tests; production state builds
    once at startup, and a stale index fails the boot.

## 4. File map

### `src/filings_analyst/` — application code

| File | Purpose / key exports |
|---|---|
| `config.py` | Env-driven `Settings` (User-Agent validated fail-loud), URL builders for the three SEC endpoints (with their inconsistent CIK/accession formats encoded once), default tickers, pinned answer model, default paths |
| `edgar/throttle.py` | `RateLimiter` — min-interval spacing, thread-safe, injectable clock/sleep |
| `edgar/cache.py` | `DiskCache` — verbatim byte cache keyed by URL hash, sidecar metadata, sharded dirs; `atomic_write_bytes` (crash-safe writes, reused by later stages) |
| `edgar/client.py` | `SECClient` — the chokepoint: cache-first, throttled, UA-enforced, retry/backoff honoring `Retry-After`; `get_json`/`get_bytes`; `SECRequestError` |
| `edgar/resolver.py` | `CikResolver` — ticker → 10-digit zero-padded CIK from the SEC directory, through the client; `pad_cik`; CLI prints target tickers |
| `edgar/filings.py` | `FilingFetcher` (latest 10-K/10-Q via Submissions parallel arrays; CompanyFacts), `FilingStore` (ticker-keyed corpus, byte-exact docs + metadata with sha256), `FilingRecord`; CLI |
| `parsing/extract.py` | The offset-preserving core: `_Extractor` (streaming events, suppression stack, block assembly, raw-coordinate offset bookkeeping), `Block`, `extract_blocks`, `verify_span` (the round-trip check), `decode_filing`, `normalize_text` |
| `parsing/sections.py` | Canonical 10-K/10-Q item sequences, heading candidates, weighted-LIS `detect_headings`, `assign_sections` |
| `parsing/chunking.py` | `pack_chunks` — greedy block packing within sections; `Chunk` |
| `parsing/pipeline.py` | `parse_filing` (integrity gate → extract → sections → chunks → verify 100% → deterministic JSONL + manifest), `VerificationError`; CLI over the store |
| `retrieval/bm25.py` | Domain `tokenize` (keeps `10-k`/`bb6`/`4.2`), `BM25` (Okapi, postings, evidence-gated `top`) |
| `retrieval/encoders.py` | `Encoder` protocol; `FastEmbedEncoder` (pinned bge models, query instruction prefix, L2-normalize, batch cap — the 5× memory-thrash fix); model registry |
| `retrieval/fusion.py` | `rrf_fuse` — reciprocal-rank fusion, deterministic ties; `DEFAULT_RRF_K = 5` (sweep-chosen) |
| `retrieval/index.py` | `build_index` (enrichment header, window splitting, embeddings, fingerprint manifest), `load_index` (staleness + integrity checks, `StaleIndexError`), `Index`; CLI |
| `retrieval/retriever.py` | `Retriever` — filters-first eligibility mask, BM25 + windowed-max-pool embedding rankings, fusion, `RetrievedChunk` with per-method diagnostics; search CLI |
| `retrieval/evaluate.py` | Gold-set runner: recall@5/@10/MRR per method, misses, rescue counts, decoy score report, crowding, determinism check, hard-bar exit code |
| `answering/schema.py` | Draft (LLM-facing) models `DraftAnswer`/`DraftClaim`; response contract `AnswerResponse`/`VerifiedClaim`/`RetrievedChunkInfo`; `RefusalReason` |
| `answering/verify.py` | `canonical` (whitespace + unicode-punctuation folding), `verify_quote` — the deterministic gate |
| `answering/scoping.py` | `scope_ticker` — alias map from entity names, single-match filter |
| `answering/answerer.py` | `Answerer` (scope → retrieve → draft → verify → respond), `Drafter` protocol, `OpusDrafter` (structured outputs, refusal-safe), `build_default_answerer`; CLI |
| `api/app.py` | `create_app` factory, `AppState`, request models with validation, endpoints `/ask` `/search` `/chunks/{id}` `/health` |
| `api/serve.py` | uvicorn entrypoint |

### Everything else

| Path | Purpose |
|---|---|
| `eval/gold_set.jsonl` | 24 labeled retrieval questions + 5 off-corpus decoys; the permanent quality regression. Notes document label provenance including post-review corrections |
| `tests/` | 106 offline tests — inventory in §6 |
| `data/` (gitignored) | `cache/` transport dedupe · `store/` corpus (raw docs, facts, chunks) · `index/` search index · `models/` embedding model |
| `.env.example` | Documents the two required secrets |
| `pyproject.toml`, `uv.lock` | Dependencies: httpx, pydantic-settings, numpy, fastembed, anthropic, fastapi, uvicorn; dev: pytest |

## 5. Code flows

### 5.1 Fetch (`edgar.filings` CLI)

1. `Settings()` validates the User-Agent (raises with a copy-pasteable fix if bad).
2. `CikResolver.resolve` — fetches `company_tickers.json` **through** `SECClient`
   (cache-first → throttle slot → GET with UA → retry/backoff → cache write), maps
   ticker → padded CIK.
3. `FilingFetcher.latest_filings` — Submissions JSON; parallel arrays are
   newest-first, so first exact `form` match = latest 10-K/10-Q; builds the Archives
   URL (unpadded CIK + dashless accession + SEC's own `primaryDocument`).
4. Document bytes stored verbatim via atomic write; metadata sidecar (ticker, CIK,
   form, dates, accession, source URL, **sha256**, size).
5. CompanyFacts JSON stored under the separate `facts/` subtree.

### 5.2 Parse (`parsing.pipeline`)

1. Integrity gate: recompute sha256 of the stored doc vs. its fetch-time metadata.
2. `decode_filing` (UTF-8 strict, latin-1 fallback — deterministic either way).
3. `extract_blocks`: one streaming pass; every handler first closes the pending text
   fragment at the current raw position; suppression stack drops invisible content;
   block tags flush; edge whitespace trimmed only via literal fragments (exact
   arithmetic). Result: `Block(text, char_start, char_end, in_table)`.
4. `detect_headings` (weighted LIS over candidates) → `assign_sections`.
5. `pack_chunks` within sections.
6. **Verification of 100% of blocks and chunks** (`verify_span` re-extracts each raw
   span and compares exactly); any failure aborts before anything is written.
7. Deterministic JSONL + manifest (counts, headings, verification tallies).

### 5.3 Index build (`retrieval.index`)

1. Chunk files gathered in sorted order; records copied line-for-line; per-file
   sha256 recorded in the manifest.
2. Entity names read from already-fetched CompanyFacts (offline).
3. Each chunk → enrichment header + 1–5 overlapping 800-char windows → embedded
   (batched) → `embeddings.npy` + `window_map.npy`.
4. Manifest: model id, params, entity names, source fingerprints, counts.

### 5.4 Search (`Retriever.search`)

1. Validate query; build eligibility mask from ticker/form/section filters
   (filters **before** ranking so ranks are computed among eligible chunks only).
2. BM25: tokenize query → score → evidence-gated top-50.
3. Embeddings: encode query (with bge instruction prefix) → cosine against all
   windows → **max-pool per chunk** → top-50.
4. `rrf_fuse` (K=5) → top-k `RetrievedChunk`s, each carrying the verbatim record and
   per-method ranks/scores. Ties by chunk id; fully deterministic.

### 5.5 Ask (`Answerer.answer`)

1. `scope_ticker` → optional filter.
2. Retrieve top-k. Empty → `refused/retrieval_empty` (LLM never called).
3. `OpusDrafter.draft`: system prompt (hard grounding rules) + question + labeled
   chunks → `messages.parse` with `DraftAnswer` schema (adaptive thinking). A
   safety-refusal/unparseable response maps to a model refusal, never a crash.
4. Model refusal → `refused/model_refusal` with its wording.
5. **The gate**: each claim through `verify_quote` (retrieved chunk? long enough?
   canonical substring?). Failures dropped and counted; survivors get the full
   provenance block copied from the chunk record.
6. Nothing survived → `refused/all_claims_unverified`; else the answer with verified
   claims + retrieval diagnostics.

### 5.6 HTTP request lifecycle

Startup: lifespan builds production state once — encoder, `load_index` (stale check
= boot failure), retriever, answerer, chunk lookup table. Per request: FastAPI
validates the body (blank/oversized → 422), the endpoint delegates to the layer
below, and the response model is the layer's own schema. `/chunks/{id}` is a dict
lookup; unknown → 404.

## 6. Test inventory — what each file guards and why

**106 tests, all offline** (fakes for LLM + embeddings; `httpx.MockTransport` for
HTTP; virtual clocks for time). One repo-data integration test skips on fresh clones.

| File (count) | What it guards, case by case |
|---|---|
| `test_config.py` (7) | The fail-loud User-Agent contract: empty/no-email/placeholder/no-dot-domain all rejected; valid accepted; URL builders produce the SEC's exact padded-CIK formats. *Why: a bad UA is silent 403s mid-crawl — the #1 operational failure.* |
| `test_throttle.py` (3) | Spacing ≥ 1/rate between acquisitions; no 1-second window ever holds > rate (asserted over 40 virtual acquisitions); no sleeping when already behind schedule. *Why: the SEC blocks IPs; bursts must be structurally impossible.* |
| `test_cache.py` (4) | Byte-exact round-trip on adversarial content (BOM, CRLF, NUL, multibyte); miss returns None; metadata survives; distinct URLs/methods → distinct keys. *Why: this cache is the provenance anchor — a single mutated byte breaks all downstream offsets.* |
| `test_client.py` (8) | UA header reaches the wire; second identical GET served from cache (transport hit exactly once); 503→200 retries with expected backoff; `Retry-After` honored; 404 raises without retry; persistent 5xx exhausts retries; connection errors retried; JSON helper decodes a copy. *Why: each is a distinct SEC-etiquette failure mode with a distinct correct behavior.* |
| `test_filings.py` (5) | Newest **exact** form match wins (10-K/A amendment and 8-K decoys skipped, older 10-K not chosen); Archives URL uses unpadded CIK + dashless accession; narrative stored byte-for-byte with sha256 metadata; facts stored under `facts/` never mingled; missing form reported, not fatal. *Why: wrong-document selection or URL-format errors would silently corrupt the corpus.* |
| `test_extract.py` (10) | The round-trip invariant under adversarial markup: entity length-shift (`&#8220;`), sentences split across inline tags, block-tag splitting, `ix:header`/`ix:hidden`/script/style/`display:none` suppression, table marking, `<br>` as whitespace, span trimming to visible text, NBSP/zero-width normalization, decode fallback, normalization idempotence. *Why: every case is a real way offsets silently drift — the failure this project cannot tolerate.* |
| `test_sections.py` (6) | TOC rows (tables) never headings; an early false "Item 1A" loses to the longer true chain (the real ASTS decoy, reproduced); mid-prose cross-references ignored; long paragraphs starting with "Item..." rejected; 10-Q duplicate item numbers resolved by PART markers; every block gets a label with tables inheriting. *Why: section detection is heuristic — these pin the known adversarial cases so they can't regress.* |
| `test_chunking.py` (5) | Packing respects target size; never crosses sections; every prose block in exactly one chunk (coverage, no duplicates, tables excluded); oversized single block flagged not split; chunk envelope spans first→last block with canonical join. *Why: chunk boundaries are provenance envelopes; coverage gaps would lose citable text.* |
| `test_bm25.py` (9) | Tokenizer keeps `10-k`/`bb6`/`4.2`/`1a` intact, lowercases, strips curly quotes, survives empty input; scores match hand-computed Okapi values (independent of the implementation); rare>common terms; length normalization; tf saturation; zero-score docs never ranked (evidence gating); deterministic ties; eligibility mask excludes before ranking. *Why: hand-rolled math needs external ground truth; the tokenizer is where domain correctness lives.* |
| `test_fusion.py` (4) | The worked corroboration example with exact expected scores; K=5-era flat-curve behavior pinned both directions (deep burial loses, moderate corroboration wins); ties by id; empty lists degrade gracefully. *Why: fusion is 15 lines that decide every ranking — its arithmetic is pinned exactly.* |
| `test_retrieval.py` (14) | End-to-end with a keyword fake encoder: result text byte-identical to source records; enrichment searchable ("Alpha Aerospace" findable) but never leaked into results; ticker/form/section filters absolute; unknown filters → empty not error; **modified chunk file → StaleIndexError**; encoder/model mismatch → error; same query → identical results; degenerate inputs (blank raises, huge query works, k>corpus, zero-overlap → embeddings-only); window coverage; **tail-only content retrievable** (the truncation fix); corroborated chunk outranks single-signal. *Why: these are the retrieval acceptance criteria, mechanized.* |
| `test_answering.py` (15) | Verification: verbatim passes; unicode/whitespace drift forgiven but changed words rejected; fabricated/short/unknown-chunk quotes rejected; canonical idempotent. Pipeline: verified claims carry full provenance; response JSON round-trips; fabricated claim dropped while good kept; all-fail → refusal with claims/answer emptied; citing unretrieved chunk dropped; empty retrieval refuses **without calling the LLM**; model refusal honored with wording. Scoping: names/tickers matched, ambiguous/absent → None, filter reaches retrieval. *Why: this is the "never hallucinates" gate — every bypass path must be provably closed.* |
| `test_api.py` (9) | Health; `/ask` payload validates against the answering layer's own schema (drift-proof); refusals travel over HTTP; validation rejects blank/oversized/bad-k (422); `/search` returns verbatim records with diagnostics; `/chunks/{id}` + 404. *Why: the API is the external contract; the harness depends on its exact shape.* |
| `test_pipeline_integration.py` (1, skips without data) | Full parse of the real ASTS 10-K: 100% verification, all 23 items detected in order, the known satellite sentence lands in an Item 1 chunk, output files well-formed. *Why: synthetic fixtures can't imitate real filer markup.* |

## 7. Current scorecards

| Stage | Measured result |
|---|---|
| Parsing | 19,544/19,544 blocks + 679/679 chunks pass round-trip verification across 6 documents; byte-identical re-runs |
| Retrieval | fused recall@5 = 0.958, recall@10 = 0.958, MRR = 0.715 (BM25 alone 0.750, embeddings alone 0.875); 7 cross-method rescues; deterministic |
| Answering (live) | 8/8 gold questions answered; 25/25 claims independently re-verified against raw source bytes; 3/3 off-corpus decoys refused; gate observed catching live fabrication (Tesla decoy) and over-reach (1 dropped claim) |
| API (live) | End-to-end over HTTP incl. independent quote re-verification via `/chunks/{id}` |
| Tests | 106/106 offline |

## 8. Current gaps and known limitations

Honest list, in rough priority order:

1. **Corpus: 3 of 8 target tickers.** LUNR, NBIS, GEV, JOBY, PWR unfetched. Pipeline
   is ticker-generic; new filers may surface parser edge cases (the invariant will
   catch them loudly). Gold set has no questions for the new companies yet.
2. **No XBRL facts track.** Numeric questions answer from narrative text (citable,
   but not the numbers-of-record design). Needs concept mapping (`Revenues` vs
   filer-specific tags), fiscal-period selection, and routing. The response schema
   already reserves `track` for this.
3. **One documented retrieval miss (g22).** "How dependent is Rocket Lab on its
   biggest customers?" — the small embedding model can't bridge the paraphrase gap
   against 300 competing same-company windows. Candidate fixes: bge-base (one-time
   re-embed; test was interrupted, batching bug since fixed) or a cross-encoder
   reranker. Accepted exception; the retrieval evaluate hard-bars intentionally
   still report it as a failure.
4. **Refusal boundary uncalibrated.** Obvious off-corpus refusals work; adversarial
   near-corpus questions ("What does NVIDIA say about its dividend policy *for
   2030*?") haven't been systematically probed — that's precisely the evaluation
   harness's job.
5. **Amendments (10-K/A) skipped by design** — an amended filing's corrections are
   invisible to the corpus.
6. **Table text excluded from chunks** — narrative-embedded tabular facts aren't
   retrievable until the facts track exists.
7. **Answer non-determinism** — inherent to the LLM; the deterministic gate is the
   compensating control (see §2).
8. **Embedding determinism is per-machine** — ONNX CPU results are stable on one
   machine; bit-identical vectors across architectures aren't guaranteed. Ranking
   determinism holds per-index.
9. **No CI** — the suite is CI-ready (offline, keyless); wiring GitHub Actions is
   pending the repo push.
10. **API is deliberately unhardened** — no auth/rate limiting; it's a local
    evaluation target, not a public service.

**Deliberately not built** (right-sizing, not omission): vector DB, LangChain-style
frameworks, rerankers (pending evidence), Docker, streaming responses, query-routing
classifiers.

## 9. Evaluation harness (built — separate repository)

The harness lives in its own repository (`sec-filings-eval-harness`) and drives this
system **black-box over HTTP, importing none of its code** (machine-checked: zero
`filings_analyst` references). Key design points:

- **Cases are data** — plain YAML files, one per case, dispatched by `type`; the
  Python runner is a reference implementation, and a runner in any language could
  execute the same case files against the same contract.
- **Independent wire schema** — the harness re-declares the expected response shapes
  rather than importing this repo's models, so API drift is a *finding*, not a shared
  assumption (an offline test proves schema violations fail cases).
- **Citation integrity is enforced on every non-refused answer** regardless of case
  type: each claim's chunk is independently fetched via `GET /chunks/{id}`, the quote
  re-verified verbatim under canonical comparison, chunk membership checked against
  the response's own `retrieved` list, shas compared, offsets validated. The harness
  trusts nothing the target says about itself.
- **Four probe axes**: correctness (8 cases, regex-pinned expected facts), refusal
  (5 — off-corpus, near-corpus-unanswerable, adjacent-company confusion), filter
  scope (3), paraphrase robustness (2 groups, both directions: stable answers *and*
  stable refusals).

**First full live run (2026-07-20): 18/18 cases passed; 23/23 independent citation
checks ok; refusals held under paraphrase and under adjacent-company confusion
(a Lockheed satellite question against a satellite-heavy corpus); median /ask
latency 4.6s (max 50.8s — a decoy the model drafted at length before the gate
stripped it).** The harness's own logic is covered by 23 offline tests (fake
client, no network, no key).
