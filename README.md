# Grounded Filings Analyst

A RAG system over SEC filings whose defining property is that it **never hallucinates**:
every claim it makes is mechanically verified against a source span in a real filing —
traceable down to exact character offsets and a file hash — and it **refuses** to answer
what it cannot ground. Refusal is a first-class, machine-readable outcome, not an error.

```
 SEC EDGAR ──fetch──▶ raw filings ──parse──▶ verified chunks ──index──▶ hybrid search
 (free APIs)         (byte-exact)          (exact offsets)            (BM25 + vectors)
                                                                            │
        HTTP API ◀──serve── verified answer ◀──verify── LLM draft ◀──retrieve┘
        (harness            (claims + quotes         (claims must quote
         contract)           + provenance)            chunks verbatim)
```

The pipeline is built so that **trust flows backward**: an answer cites a claim, the
claim carries a verbatim quote, the quote is machine-checked against a chunk, the chunk
carries exact character offsets into a raw filing stored byte-for-byte as the SEC served
it, and the file's sha256 seals the chain. Independent verification of any answer needs
nothing but the HTTP API.

For design decisions, the complete file map, test rationale, and code flows, see
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Requirements

| Requirement | Notes |
|---|---|
| Python ≥ 3.10 | developed on 3.10.5 |
| [uv](https://docs.astral.sh/uv/) | dependency manager (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| ~500 MB disk | embedding model (~80 MB), filings + index for 3 tickers (~30 MB), room to grow |
| SEC User-Agent string | **required by the SEC** for all EDGAR requests — see setup |
| Anthropic API key | **only for the answering layer**; fetch/parse/search run free and offline |

## Setup

```bash
git clone <this repo> && cd "SEC financials"
uv sync                                   # creates .venv, installs everything
cp .env.example .env                      # then edit .env:
```

`.env` (gitignored — never commit it):

```
SEC_USER_AGENT="Your Name your.email@example.com"   # SEC rejects requests without it
ANTHROPIC_API_KEY="sk-ant-..."                      # console.anthropic.com
```

Load it in each terminal session: `set -a; source .env; set +a`

The app **fails loudly** if `SEC_USER_AGENT` is missing or looks like a placeholder —
a malformed User-Agent is the #1 cause of silent SEC 403 blocks, so it is validated at
startup, not discovered mid-crawl.

## Quickstart (fresh machine → first grounded answer)

```bash
set -a; source .env; set +a

uv run python -m filings_analyst.edgar.resolver      # 1. ticker → CIK (1 SEC call)
uv run python -m filings_analyst.edgar.filings       # 2. fetch pilot filings (NVDA ASTS RKLB)
uv run python -m filings_analyst.parsing.pipeline    # 3. parse → verified chunks (offline)
uv run python -m filings_analyst.retrieval.index     # 4. build search index (offline;
                                                     #    downloads embed model once)
uv run python -m filings_analyst.answering.answerer "How many people work at Rocket Lab?"
```

Expected output of the last command: a direct answer, then each claim with its verbatim
quote and citation (ticker, form, filing date, section, source file, character offsets).

---

## Components

Each package is one pipeline stage with its own machine-checked guarantee:

| Package | What it does | Its guarantee |
|---|---|---|
| `filings_analyst.edgar` | Fetches from SEC EDGAR: ticker→CIK resolution, latest 10-K/10-Q narrative documents, CompanyFacts XBRL | All traffic flows through one client enforcing SEC etiquette (User-Agent, ≤8 req/s throttle, disk cache, retry/backoff). Filings stored **byte-for-byte** with sha256 recorded |
| `filings_analyst.parsing` | Turns raw inline-XBRL HTML into readable, section-labeled chunks | **Round-trip invariant, checked for 100% of chunks**: re-extracting `raw[char_start:char_end]` with the canonical extractor reproduces the chunk text exactly. Any failure aborts the build |
| `filings_analyst.retrieval` | Hybrid search: hand-rolled BM25 + local embeddings, rank-fused | Deterministic ranking; verbatim chunk text on every result; the index fingerprints its source files and **refuses to load if they changed** |
| `filings_analyst.answering` | LLM drafts claims-with-quotes; **our code verifies every quote** against the cited chunk | No unverified claim can reach a response — structurally. Everything-stripped → refusal with a reason code |
| `filings_analyst.api` | FastAPI wrapper — the external evaluation contract | Serves the answering layer's schema *unmodified*; fails fast at startup on a stale index |

## Models used

| Model | Where | Why this one |
|---|---|---|
| **BM25 (Okapi)** — implemented in-repo, no library | `retrieval/bm25.py` | Exact-term matching (tickers, "Neutron", "10-K", numbers). The domain tokenizer keeps identifiers like `10-K`/`BB6` intact — the actual correctness risk, which is why it's custom |
| **bge-small-en-v1.5** (384-dim, ONNX via fastembed) | `retrieval/encoders.py` | Semantic/paraphrase matching. Local, free, pinned, offline after a one-time ~80 MB download to `data/models/`. No API keys in the retrieval path |
| **claude-opus-4-8** (Anthropic API) | `answering/answerer.py` | Drafts answers under structured outputs (schema-valid JSON by construction). Pinned in `config.py`; swappable. ~$0.05/question observed |

The two retrieval signals are merged with reciprocal-rank fusion (K=5, chosen by a
transparent parameter sweep against the committed gold set — see the architecture doc).

---

## Using the product

### Ask questions (CLI)

```bash
uv run python -m filings_analyst.answering.answerer "Who manufactures NVIDIA's chips?"
```

Anatomy of a response:
- **the answer** — one short paragraph, built only from verified claims;
- **claims** — each with the statement, the *verbatim quote* backing it, and the full
  citation: `RKLB 10-K (2026-02-26), Business, rklb-20251231.htm chars 347071–351318`;
- **refusals** — `REFUSED (retrieval_empty | model_refusal | all_claims_unverified)`.
  Ask it "What is Apple's revenue?" and it should refuse: Apple isn't in the corpus,
  and declining to guess is the product working.

### Search without the LLM (free, instant)

```bash
uv run python -m filings_analyst.retrieval.retriever "Neutron first launch" --k 5
uv run python -m filings_analyst.retrieval.retriever "risk factors" --ticker ASTS --form 10-K
```

Each hit shows its rank under each signal (`bm25 #1, embed #3`) — the "why was this
retrieved" diagnostics.

### Run the HTTP API

```bash
uv run python -m filings_analyst.api.serve          # http://127.0.0.1:8000
```

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{"question": ..., "k"?}` → answer, verified claims with provenance, retrieved-chunk diagnostics, refusal flags |
| `POST /search` | `{"query": ..., "k"?, "ticker"?, "form"?, "section"?, "method"?}` → raw hybrid retrieval with per-method diagnostics |
| `GET /chunks/{id}` | One chunk record, verbatim — lets any external client independently re-verify a cited quote |
| `GET /health` | Corpus size, tickers, pinned model ids |

Interactive OpenAPI docs at **`/docs`**. Example:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "content-type: application/json" \
  -d '{"question": "When did ASTS launch BlueWalker 3?"}'
```

---

## Getting more SEC filings

The pipeline is ticker-generic. To add companies:

1. **Fetch** (any ticker in the SEC's directory works; CIKs are resolved, never hardcoded):
   ```bash
   uv run python -m filings_analyst.edgar.filings LUNR GEV JOBY
   ```
   This pulls each company's **most recent 10-K and 10-Q** (exact form match — amendments
   like 10-K/A are deliberately skipped) plus its CompanyFacts XBRL JSON, into
   `data/store/<TICKER>/`. Re-running is free: every response is disk-cached, so nothing
   is ever fetched twice. The default ticker set lives in `filings_analyst/config.py`
   (`DEFAULT_TARGET_TICKERS`).

2. **Parse** the new filings into verified chunks:
   ```bash
   uv run python -m filings_analyst.parsing.pipeline
   ```

3. **Rebuild the index** (mandatory after any parse — see below):
   ```bash
   uv run python -m filings_analyst.retrieval.index
   ```

SEC etiquette is enforced automatically and cannot be bypassed: descriptive User-Agent
on every request, throttled to ≤8 req/s (SEC's ceiling is 10), disk caching, exponential
backoff on transient errors. Getting these wrong normally causes silent IP blocks — here
they're structural.

## Parsing requirements & guarantees

- **Input**: the primary 10-K/10-Q document as filed — modern **inline-XBRL XHTML** (the
  format all three pilot filers use; the parser was validated against three structurally
  diverse filers before generalizing).
- **Never modify stored raw files.** Chunk offsets index into the decoded bytes of the
  file exactly as fetched; the sha256 in each filing's metadata seals it. If a stored
  file changes, parsing refuses to run on it (integrity gate) and a rebuilt index will
  refuse to serve stale offsets.
- **The round-trip invariant** is checked for every block and chunk at parse time:
  re-extracting the chunk's raw span reproduces its text exactly. A parse that can't
  guarantee its offsets fails loudly instead of writing silently-wrong citations.
- Section labels (Item 1, 1A, 7...) are detected for 10-K and 10-Q structures; unknown
  sections degrade to `"preamble"`/part labels — labels are metadata, never provenance.
- Table cell text is extracted and preserved at the block level but **excluded from
  chunks** — financial numbers-of-record travel on the XBRL CompanyFacts track instead.
- Deterministic: re-running the pipeline produces byte-identical output.

## Testing & quality gates

```bash
uv run pytest                                       # 106 tests, fully offline, ~1s
uv run python -m filings_analyst.retrieval.evaluate # gold-set retrieval scorecard
```

- The offline suite needs **no network, no API key, no model download** — LLM and
  embedding backends are injected fakes. It covers SEC-client throttling/caching,
  the parser's round-trip invariant, section detection against decoys, BM25 against
  hand-computed fixtures, fusion arithmetic, verification/refusal paths, and the API
  contract.
- Retrieval quality regresses against a committed gold set
  ([eval/gold_set.jsonl](eval/gold_set.jsonl)) labeled by text-grep, never by running
  the retriever. Current: fused recall@5 = 0.958, beating both single methods.
- The answering layer's live check (8 gold questions, 3 off-corpus decoys): 8/8
  answered, 25/25 claims independently re-verified against raw source bytes, 3/3
  decoys refused.

## Data layout

```
data/
├── cache/    # every SEC response, keyed by URL hash — transport dedupe, safe to wipe
├── store/    # the corpus: <TICKER>/narrative/<filing>/ (raw doc + metadata),
│             #             <TICKER>/facts/companyfacts.json, <TICKER>/chunks/*.jsonl
├── index/    # search index: records + embeddings + fingerprint manifest
└── models/   # downloaded embedding model (one-time)
```

All of `data/` is regenerable and gitignored.

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `SEC_USER_AGENT is not set...` | `.env` not loaded → `set -a; source .env; set +a` |
| `Chunk file ... changed since the index was built` | You fetched/re-parsed → rebuild: `uv run python -m filings_analyst.retrieval.index` |
| Authentication error on ask | `ANTHROPIC_API_KEY` not loaded or rotated |
| Index build pegs CPU ~2 min | Normal — one-time embedding of all chunk windows |
| HTTP 403s from SEC | User-Agent invalid, or IP temporarily blocked — wait, verify `.env` |

## Roadmap

Remaining after the core (all core stages are complete and verified): scale from the
3-ticker pilot to all 8 target tickers; an XBRL facts track for precise numeric
answers; the external evaluation harness (separate project, driven over this HTTP
API); retrieval polish for one documented gold-set miss; CI. Details and current gaps:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
