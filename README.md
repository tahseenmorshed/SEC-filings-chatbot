# Grounded Filings Analyst

A question-answering system over SEC filings that never makes things up. Ask it about
a company and it answers with a direct quote from the actual filing, plus a citation
that points to the exact characters in the exact source file. If it cannot back an
answer with evidence from the filings, it refuses. Refusing is the point, not a bug.

The trick is that we do not trust the language model to stay grounded. It drafts an
answer, and then our own code checks every quote against the source. Anything it cannot
verify gets thrown away. If nothing survives, the answer becomes a refusal.

For a full study of how the system works, read [docs/REPORT.md](docs/REPORT.md). For
the engineering decision log and file map, read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## How it fits together

```
 SEC EDGAR  ->  raw filings  ->  verified chunks  ->  hybrid search
 (free)         (stored           (exact character     (keywords +
                byte for byte)     offsets)             meaning)
                                                            |
   HTTP API  <-  verified answer  <-  quote check  <-  LLM drafts an
   (for the      (quotes + full       (our code,        answer from the
   evaluator)     provenance)          not the LLM)       retrieved chunks
```

Trust runs backwards along that chain. An answer cites a claim, the claim carries a
quote, the quote is checked against a chunk, the chunk knows its exact position in a
raw file that is stored unchanged, and the file's fingerprint (a sha256 hash) seals the
whole thing. To verify any answer yourself, all you need is the HTTP API.

## What you need

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management
- About 500 MB of disk for the embedding model, filings, and search index
- A descriptive SEC User-Agent string (the SEC requires one on every request)
- An Anthropic API key, but only for the answering step. Fetching, parsing, and search
  all run free and offline.

## Setup

```bash
git clone https://github.com/tahseenmorshed/SEC-filings-chatbot.git
cd SEC-filings-chatbot
uv sync
cp .env.example .env
```

Then edit `.env`:

```
SEC_USER_AGENT="Your Name your.email@example.com"
ANTHROPIC_API_KEY="sk-ant-..."
```

Load it in each terminal session with `set -a; source .env; set +a`.

If the User-Agent is missing or looks like a placeholder, the app stops at startup and
tells you what to fix. A bad User-Agent is the most common reason the SEC silently
blocks you, so it is checked up front rather than discovered halfway through a download.

## Quick start

From a fresh clone to a first grounded answer:

```bash
set -a; source .env; set +a

uv run python -m filings_analyst.edgar.resolver      # 1. look up company IDs
uv run python -m filings_analyst.edgar.filings       # 2. fetch the pilot filings
uv run python -m filings_analyst.parsing.pipeline    # 3. parse into chunks
uv run python -m filings_analyst.retrieval.index     # 4. build the search index
uv run python -m filings_analyst.answering.answerer "How many people work at Rocket Lab?"
```

The last command prints the answer, then each claim with its quote and citation.

## Using it

### Ask a question

```bash
uv run python -m filings_analyst.answering.answerer "Who manufactures NVIDIA's chips?"
```

You get a short answer, then each claim with the quote that backs it and a full
citation (company, form, filing date, section, source file, character range). If the
filings cannot answer, you get `REFUSED` with a reason. Try "What is Apple's revenue?"
to see a refusal: Apple is not in the corpus, so declining to guess is correct
behavior. Each question costs roughly 5 cents and takes a few seconds.

### Search without the model (free and instant)

```bash
uv run python -m filings_analyst.retrieval.retriever "Neutron first launch" --k 5
uv run python -m filings_analyst.retrieval.retriever "risk factors" --ticker ASTS
```

Each result shows why it was found, with its rank under each search method.

### Run the API

```bash
uv run python -m filings_analyst.api.serve          # serves on http://127.0.0.1:8000
```

Four endpoints:

- `POST /ask` takes `{"question": ...}` and returns the answer, verified claims with
  provenance, the retrieved chunks, and refusal flags.
- `POST /search` returns raw search results with per-method diagnostics.
- `GET /chunks/{id}` returns one chunk exactly as stored, so any client can re-check a
  quote itself.
- `GET /health` reports the corpus size and the models in use.

Interactive documentation is at `/docs` in your browser. Example:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "content-type: application/json" \
  -d '{"question": "When did ASTS launch BlueWalker 3?"}'
```

## Adding more filings

The pipeline works for any US-listed ticker. To add companies, run three steps in
order:

```bash
uv run python -m filings_analyst.edgar.filings LUNR GEV JOBY   # 1. fetch
uv run python -m filings_analyst.parsing.pipeline              # 2. parse to chunks
uv run python -m filings_analyst.retrieval.index               # 3. rebuild the index
```

Step 1 pulls each company's most recent annual report (10-K) and quarterly report
(10-Q), plus its financial data. Re-running it is free because every response is
cached. Step 3 is required after any parse. If you skip it, the server refuses to start
with a stale-index error, which is the safety check doing its job.

The default company list lives in `src/filings_analyst/config.py`.

## The models

- **BM25** is a keyword-ranking algorithm, written from scratch in the repo (about 100
  lines). It handles exact terms like tickers, product names, and numbers. The custom
  part is the tokenizer, which keeps identifiers like `10-K` and `BB6` in one piece.
- **bge-small-en-v1.5** is a small embedding model that runs locally on the CPU. It
  handles meaning-based matching, so "how many spacecraft" can find a chunk about
  "satellites". It downloads once (about 80 MB) and then runs offline with no key.
- **claude-opus-4-8** is the language model that drafts answers, through the Anthropic
  API. It is pinned in the config and easy to swap.

The two search models are free and local. Only the final answering step calls a paid
API.

## How the filings are parsed

Filings arrive as inline XBRL, which is readable HTML with thousands of invisible
accounting tags mixed in. The parser has one hard requirement: it must produce clean
text while remembering exactly where every piece came from. Standard HTML libraries
give you clean text but lose the positions, so we use Python's streaming parser, which
reports each element's position, and do the assembly ourselves.

The rules the parser follows:

- Stored raw files are never modified. Chunk positions point into the file exactly as
  the SEC served it, and the sha256 fingerprint seals it.
- Every chunk is verified at build time. We slice the raw file at the chunk's recorded
  positions, re-extract, and confirm we get the chunk's text back exactly. If any chunk
  fails, the whole build stops. A citation system with wrong positions would be worse
  than none.
- Section labels (Item 1 Business, Item 1A Risk Factors, and so on) are detected but
  treated as metadata, never as provenance. A wrong label cannot corrupt a citation.
- Table cells are kept out of chunks. The real financial numbers come from the
  structured data instead.
- Output is deterministic. Parsing twice gives byte-identical files.

## The evaluation harness

The `eval-harness/` directory holds a separate project that tests this system from the
outside. It talks only HTTP and imports none of the analyst's code, so it cannot inherit
its blind spots. It runs 18 test cases covering correct answers, refusals, search
filters, and stability across rephrasings, and it independently re-verifies every quote
in every answer by fetching the cited chunk itself.

It has its own dependencies and its own virtual environment. To run it, start the API in
one terminal and then:

```bash
cd eval-harness
uv sync
uv run pytest                                                              # its own tests, offline
uv run python -m harness.runner --base-url http://127.0.0.1:8000           # full live run
```

See [eval-harness/README.md](eval-harness/README.md) for details.

## Running the tests

```bash
uv run pytest                                        # 106 tests, all offline, about 1 second
uv run python -m filings_analyst.retrieval.evaluate  # retrieval quality scorecard
```

The whole test suite runs with no network and no API key. The language model, the
embedder, and the network are all replaced by fakes, so the guarantees are checked on
every run for free. Each test group targets a specific real failure: a bad User-Agent
reaching the SEC, request bursts, storage corrupting a byte, offsets drifting during
parsing, the tokenizer shredding an identifier, an unverified claim slipping into an
answer, and so on. The full list with explanations is in
[docs/REPORT.md](docs/REPORT.md).

## Where files live on disk

```
data/
  cache/   every SEC response, keyed by URL. Safe to delete; it just costs re-fetches.
  store/   the corpus: raw documents, financial data, and chunks, organized by ticker.
  index/   the search index: records, embeddings, and a fingerprint manifest.
  models/  the downloaded embedding model.
```

Everything under `data/` is regenerable and is not committed to git.

## Common problems

- `SEC_USER_AGENT is not set`: you did not load `.env`. Run `set -a; source .env; set +a`.
- `Chunk file ... changed since the index was built`: you fetched or re-parsed. Rebuild
  the index with `uv run python -m filings_analyst.retrieval.index`.
- Authentication error when asking: the Anthropic key is not loaded, or was rotated.
- The index build uses the CPU heavily for a couple of minutes. That is normal. It is
  the one-time cost of embedding every chunk.

## Current status and roadmap

The core system is complete and verified end to end: fetch, parse, retrieve, answer,
and serve, each with machine-checked guarantees. An external evaluation harness (a
separate project) tests the running system over HTTP and passed 18 of 18 cases with all
23 independent citation checks.

Still to do: cover all 8 target tickers instead of the current 3, add a track for
precise financial numbers from the structured XBRL data, close one documented search
miss, and add continuous integration. Details and known gaps are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
