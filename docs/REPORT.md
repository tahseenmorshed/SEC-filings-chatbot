# Grounded Filings Analyst: Study Report

This report explains the whole system, layer by layer. It is written to be studied.
Every piece of jargon is explained the first time it appears. After reading it, you
should be able to explain the system to someone else.

---

## 1. What this project is

A question-answering system over SEC filings that never presents an unverified
statement. You ask "How many people work at Rocket Lab?" and it answers with the
number, a verbatim quote from the actual filing, and a citation that points to the
exact characters in the exact file. If it cannot back an answer with evidence, it
refuses. Refusing is a feature, not an error.

The technique is called **RAG** (Retrieval-Augmented Generation): instead of letting
a language model answer from memory, you first *retrieve* relevant documents, then
make the model answer *only* from those documents. Our twist is that we do not trust
the model to follow that rule. We verify its output mechanically.

A second, separate project (the **evaluation harness**) tests the running system
from the outside over HTTP and re-verifies every citation independently.

## 2. The core guarantee

Everything hangs on one chain of custody:

```
answer -> claim -> verbatim quote -> chunk -> character offsets -> raw file -> sha256
```

Read it right to left. We store each SEC filing byte-for-byte as the SEC served it,
and record its **sha256** (a fingerprint of the file's exact bytes; if one byte
changes, the fingerprint changes). We cut the filing into **chunks** (paragraph-sized
pieces of text), and each chunk records the exact character positions (**offsets**)
it came from in that file. When the model answers, every claim must cite a chunk and
include a quote. Our code checks the quote really appears in that chunk. So every
sentence in an answer can be traced, mechanically, to real bytes on disk.

One honest caveat: the language model itself is not deterministic. Ask twice, and the
wording may differ. The guarantee is not "same answer every time". It is "never an
unverified statement". The verification step is deterministic, and that is the part
that matters.

## 3. Layer 1: Ingestion (`src/filings_analyst/edgar/`)

**Job:** download filings from the SEC and store them safely.

**Background.** The SEC's filing system is called **EDGAR**. It is free and needs no
API key, but has strict etiquette: every request must carry a User-Agent header in
the form "Name email", and you must stay under 10 requests per second. Breaking these
rules gets your IP silently blocked. Companies are identified by a **CIK** (Central
Index Key, a number like 0001045810 for NVIDIA), not by ticker. Each filing has an
**accession number** (its unique ID) and is actually a folder of many files: the main
document, exhibits, images.

**How it works.**

1. `resolver.py` turns a ticker like RKLB into its CIK using the SEC's public
   company directory. CIKs are never hardcoded.
2. `filings.py` reads the company's filing history, picks the most recent 10-K
   (annual report) and 10-Q (quarterly report), and downloads the main document.
   The SEC itself marks which file is the main one, so we never guess.
3. It also downloads the company's **XBRL** data (machine-readable financial
   numbers: revenue, net income, and so on) as a separate track.
4. Everything is stored byte-for-byte in `data/store/`, with a metadata file
   recording the sha256, dates, and source URL.

**Key design decisions.**

- All SEC traffic goes through one client class (`client.py`). It enforces the
  User-Agent, throttles to 8 requests per second, caches every response to disk
  (nothing is ever fetched twice), and retries transient errors. Because there is
  only one door, the etiquette rules cannot be bypassed by accident.
- The throttle spaces requests evenly instead of allowing bursts. A burst is exactly
  what gets you blocked.
- Amended filings (10-K/A) are skipped on purpose. Handling amendments correctly is
  a later concern; for now we want the canonical original.
- The User-Agent is validated at startup. If it is missing or looks like a
  placeholder, the program stops with instructions, instead of failing silently
  mid-download.

## 4. Layer 2: Parsing (`src/filings_analyst/parsing/`)

**Job:** turn a 2 to 4 MB blob of filing HTML into clean, labeled, citable chunks.

**Background.** Modern filings are **inline XBRL**: normal readable sentences
tangled up with thousands of invisible accounting tags. About 4 percent of a filing
is pure machine data that a human never sees. The visible text is split across
thousands of tags, and characters like curly quotes are written as codes
(`&#8220;` is six characters on disk but one character on screen).

**The central problem.** We need clean text, but we also need to know exactly where
each piece came from. Standard HTML libraries (like BeautifulSoup) give you clean
text but throw away the positions. So we use Python's built-in streaming parser,
which reports the position of every element as it scans, and we do the small amount
of assembly ourselves.

**How it works.**

1. `extract.py` walks the raw file start to finish. It skips invisible content,
   stitches text fragments into blocks (roughly paragraphs), and records each
   block's exact start and end position in the original file. All position math is
   done in raw file coordinates, never on decoded text, because decoding changes
   lengths (that six-character quote code becomes one character).
2. `sections.py` labels blocks with the filing section they belong to (Item 1
   Business, Item 1A Risk Factors, Item 7 Management's Discussion, and so on).
   This is tricky because "Item 1A" appears in the table of contents and in
   cross-references, not just at the real heading. The fix: real headings appear in
   a known order, so we find the longest chain of candidates that follows that
   order. False candidates break the chain and lose automatically.
3. `chunking.py` groups blocks into chunks of about 3000 characters, never crossing
   a section boundary, never cutting inside a block.
4. `pipeline.py` runs it all, then checks the **round-trip invariant** for every
   single chunk: take the chunk's recorded positions, slice the raw file, re-run the
   extractor on that slice, and confirm you get the chunk's text back exactly. If
   even one chunk fails, the whole build aborts. A citation system with silently
   wrong positions would be worse than none.

**Key design decisions.**

- Section labels are metadata, not provenance. If a label is ever wrong, the
  citation is still exact. The two systems are kept independent on purpose.
- Tables are excluded from chunks. Table cells ripped out of their rows are noise,
  and the real numbers come from the XBRL track anyway.
- Output is deterministic: run the parser twice, get byte-identical files.

## 5. Layer 3: Retrieval (`src/filings_analyst/retrieval/`)

**Job:** given a question, find the handful of chunks most likely to contain the
answer, out of all chunks in the corpus.

**Two search methods, always both.**

- **BM25** is the classic keyword algorithm behind traditional search engines. It
  scores a chunk higher when it contains the question's words, weighting rare words
  (like "Neutron") far above common ones (like "company"). We wrote it ourselves in
  about 100 lines because the risky part is not the formula, it is the
  **tokenizer** (the rule for splitting text into words). A naive tokenizer
  destroys "10-K" into "10" and "K". Ours keeps domain identifiers intact.
- **Embeddings** turn a piece of text into a list of numbers (a **vector**) that
  acts like coordinates on a map of meaning. Texts about similar things land near
  each other even when they share no words. This is how "how many spacecraft" finds
  a chunk about "satellites". We use a small open model (bge-small-en-v1.5) that
  runs locally on the CPU. No API, no key, no cost, and results are reproducible.

Each method covers the other's blind spot. Keywords fail on paraphrase; embeddings
blur exact identifiers (they can confuse Electron with Neutron, two different
rockets). So both run on every query, and **RRF** (Reciprocal Rank Fusion) merges
the two ranked lists using only positions, not scores. A chunk ranked highly by
either method surfaces; a chunk ranked highly by both wins.

**Three problems we hit and solved.**

- *The "the Company" problem.* Inside its own filing, Rocket Lab says "we", never
  "Rocket Lab". So the searchable copy of each chunk is prefixed with the company
  name and section. The stored chunk, the one that gets cited, stays untouched. A
  test proves the prefix never leaks into results.
- *The invisible-tail problem.* The embedding model reads only about 2000
  characters, but the median chunk is longer. So each chunk is embedded as several
  overlapping windows, and the chunk's score is its best window. No text is
  invisible to search.
- *Vote swamping.* With the textbook fusion constant, a chunk ranked #1 by one
  method got outvoted by mediocre chunks that appeared mid-list in both. We swept
  the parameter against a test set and lowered it, so a confident #1 survives.

**Measurement.** Retrieval quality is scored against a **gold set**: 24 questions
whose correct chunks were labeled by searching the text directly, never by running
the retriever (otherwise the labels inherit the system's own blind spots). The key
metric is **recall@5**: how often a correct chunk appears in the top 5 results.
Current: 0.958 for the hybrid, beating keywords alone (0.750) and embeddings alone
(0.875), which proves the hybrid earns its complexity. One adversarial paraphrase
question misses; it is documented, not hidden.

**What we deliberately did not build:** a vector database. All embeddings together
are a few megabytes. Comparing the question against every chunk directly takes
under a millisecond and is exact. A vector database would add complexity and
approximation for nothing at this scale.

## 6. Layer 4: Answering (`src/filings_analyst/answering/`)

**Job:** produce the final answer, with the guarantee enforced.

**The flow.**

1. **Scope.** Simple code (no model) checks whether the question names one of our
   companies and, if so, filters retrieval to that ticker.
2. **Retrieve.** Top 8 chunks via the hybrid search.
3. **Draft.** The model (claude-opus-4-8) receives the question and the chunks with
   strict rules: answer only from the chunks, attach a verbatim quote to every
   claim, refuse if the chunks cannot answer. **Structured outputs** (an API feature
   that forces the reply to match a fixed schema) guarantee the response is valid
   JSON with the exact fields we expect. Malformed output is impossible, so the
   next step only judges truth, not format.
4. **Verify. This is the heart.** Our code checks every claim: the quote must
   actually appear in the cited chunk (allowing trivial differences like curly
   versus straight quotes, but never a changed word or number), the quote must be
   at least 15 characters (so a fragment like "the" cannot "verify" anything), and
   the cited chunk must be one the model was actually shown. Failed claims are
   dropped. If nothing survives, the whole answer becomes a refusal.
5. **Respond.** Answer, surviving claims with full provenance, retrieval
   diagnostics, and refusal flags.

**Refusals are first-class.** Three machine-readable reasons: `retrieval_empty`
(nothing found; the model is never even called), `model_refusal` (the model judged
the chunks insufficient), `all_claims_unverified` (the model answered but the gate
stripped everything). The last one is the anti-hallucination mechanism visibly
working. In live testing, the model tried to answer a Tesla question from memory,
and the gate converted the whole response to a refusal.

**Why the middle ground on quotes.** Forcing the model to speak only in quotes
reads robotically. Letting it paraphrase freely and having a second model judge the
paraphrase puts a fuzzy judge at the gate. So: claims may be phrased naturally, but
each must carry a verbatim quote, and only the quote is machine-verified.

## 7. Layer 5: HTTP API (`src/filings_analyst/api/`)

**Job:** expose the system as a web service, mainly for the evaluation harness.

Four endpoints: `POST /ask` (the full answering pipeline), `POST /search`
(retrieval only, with per-method diagnostics), `GET /chunks/{id}` (fetch one chunk
verbatim, so an outside client can re-verify any quote itself), and `GET /health`.
Interactive documentation is auto-generated at `/docs`.

The layer is deliberately thin. `/ask` returns the answering layer's response
object unmodified, and a test validates the wire payload against that exact schema,
so the API and the pipeline cannot drift apart. At startup the server checks that
the search index still matches the chunk files and refuses to boot if not. Serving
stale offsets would silently break the provenance chain.

## 8. The evaluation harness (separate repository)

**Job:** prove the system's claims from the outside, trusting nothing.

It is a black-box tester. It imports none of the main project's code (verified by
a search for the package name: zero references) and talks only HTTP. It even
re-declares the expected response shapes itself instead of importing them, so if
the API drifts, the drift is a caught failure rather than a shared assumption.

Test cases are plain YAML data files, so a runner in any language could execute
them. Four kinds:

- **correctness** (8 cases): known-answerable questions; must not refuse, and
  expected facts (as regular expressions) must appear in the answer.
- **refusal** (5 cases): must refuse. Includes off-corpus questions (Apple's
  revenue), a near-miss (NVIDIA's stock price on a specific date, which filings do
  not contain), and an adjacent-company trap (a Lockheed satellite question, where
  retrieval surfaces plausible satellite chunks from ASTS and answering from them
  would be a serious failure).
- **filter** (3 cases): scoped searches must return zero out-of-scope hits.
- **robustness** (2 groups): several phrasings of one question must agree. Answers
  stay answers with the same facts; refusals stay refusals.

On top of the per-case checks, **every** non-refused answer gets citation
integrity checks: each claim's chunk is fetched fresh through `GET /chunks/{id}`,
the quote is re-verified, fingerprints are compared, offsets validated.

**First full live run: 18/18 cases passed, 23/23 citation checks passed.** Median
answer latency 4.6 seconds.

## 9. Models used

| Model | Type | Where it runs | Cost | Role |
|---|---|---|---|---|
| BM25 (Okapi) | keyword algorithm, ~100 lines, written in-repo | in-process | free | exact-term search |
| bge-small-en-v1.5 | embedding model, 384 dimensions, ONNX format | local CPU, downloaded once (~80 MB) | free | meaning-based search |
| claude-opus-4-8 | large language model, via the Anthropic API | Anthropic's servers | ~$0.05 per question | drafting answers under structured outputs |

The two search models are free and local, so the entire pipeline except the final
answer step runs offline at zero cost. The LLM is pinned in `config.py` and easy to
swap.

## 10. File map

**Main repository, `src/filings_analyst/`:**

| File | Purpose |
|---|---|
| `config.py` | settings, User-Agent validation, SEC URL builders, pinned model, target tickers |
| `edgar/throttle.py` | rate limiter (even spacing, testable with a fake clock) |
| `edgar/cache.py` | byte-exact disk cache of every SEC response |
| `edgar/client.py` | the single SEC HTTP client (etiquette enforced here) |
| `edgar/resolver.py` | ticker to CIK |
| `edgar/filings.py` | pick and download latest 10-K/10-Q plus XBRL facts |
| `parsing/extract.py` | offset-preserving text extraction; the round-trip check |
| `parsing/sections.py` | section heading detection |
| `parsing/chunking.py` | blocks into chunks |
| `parsing/pipeline.py` | orchestration, 100 percent verification, deterministic output |
| `retrieval/bm25.py` | tokenizer and BM25 |
| `retrieval/encoders.py` | embedding backend (swappable, fakeable) |
| `retrieval/fusion.py` | rank fusion |
| `retrieval/index.py` | index build/load, staleness protection |
| `retrieval/retriever.py` | filters, both searches, fusion, diagnostics |
| `retrieval/evaluate.py` | gold-set scorecard |
| `answering/schema.py` | the draft schema (for the model) and response schema (for the world) |
| `answering/verify.py` | the quote verification gate |
| `answering/scoping.py` | company detection in questions |
| `answering/answerer.py` | the pipeline; the model behind a swappable interface |
| `api/app.py`, `api/serve.py` | the web service |

Plus `eval/gold_set.jsonl` (retrieval gold set), `tests/` (106 tests), and
`data/` (gitignored: cache, store, index, models).

**Harness repository:** `harness/wire_schema.py` (independent response shapes),
`harness/client.py` (plain HTTP), `harness/cases.py` (YAML loader),
`harness/checks.py` (all check logic as pure functions), `harness/runner.py`
(execution and reporting), `cases/` (18 YAML cases), `tests/` (23 tests).

## 11. Tests: what each one checks and why

All 106 main-repo tests run offline in about one second. No network, no API key.
The model and the embedder are replaced by fakes, HTTP by a mock transport, time by
a virtual clock. That design matters: it means the guarantees are checked on every
run, anywhere, for free.

A principle used throughout: every test pins a *specific real failure mode*, not
coverage for its own sake. Below, each file, its tests, and the failure it guards.

### `test_config.py` (7 tests)

Five invalid User-Agent forms are rejected (empty, no email, no name, the shipped
placeholder, a domain without a dot), one valid form is accepted, and the SEC URL
builders are checked for the exact formats the endpoints demand.
**Why:** a bad User-Agent does not error, it gets you silently blocked by the SEC.
And the SEC's URL formats are inconsistent (padded CIK in one place, unpadded in
another; accession numbers with and without dashes), which is easy to get subtly
wrong.

### `test_throttle.py` (3 tests)

Requests are spaced at least 1/8 second apart; no sliding one-second window ever
contains more than 8 requests (checked over 40 simulated requests); a caller that
is already slow is not additionally delayed.
**Why:** the SEC's limit is per-instant, not average. A burst of 20 requests
followed by silence averages fine and still gets you blocked. The fake clock makes
these tests instant and exact.

### `test_cache.py` (4 tests)

A stored response with hostile content (byte-order marks, Windows line endings, a
null byte, multibyte characters) comes back byte-identical. Misses return None.
Metadata survives. Different URLs and methods get different keys.
**Why:** this cache is the provenance anchor. If storage altered even one byte,
every character offset computed downstream would point at the wrong place, with no
error anywhere.

### `test_client.py` (8 tests)

The User-Agent actually reaches the wire. A repeated request hits the network once
(the cache serves the second). A 503 retries with the right backoff delay and then
succeeds. A Retry-After header from the server overrides our own delay. A 404
fails immediately with no retry. A persistent 500 gives up after the configured
attempts. A dropped connection is retried. JSON parsing decodes a copy, never the
cached original.
**Why:** each HTTP failure type has a different correct response. Retrying a 404
wastes the rate budget; not retrying a 503 makes the pipeline flaky; ignoring
Retry-After is rude to the SEC and risks blocking.

### `test_filings.py` (5 tests)

Given a filing history containing decoys (an 8-K, an amendment 10-K/A, an older
10-K), the fetcher picks exactly the newest true 10-K and 10-Q. The archive URL is
built with the unpadded CIK and dashless accession number. The stored document is
byte-identical with correct metadata. XBRL facts land in their own folder, never
mixed with narrative. A company that has never filed a 10-K is reported, not a
crash.
**Why:** silently ingesting an amendment or a years-old filing would poison every
answer built on it, and nothing downstream could detect it.

### `test_extract.py` (10 tests)

The heart of the parser. Entity codes like `&#8220;` (six bytes on disk, one
character on screen) do not shift offsets. A sentence split across inline tags
becomes one block. Paragraph tags split blocks. Invisible content (XBRL headers,
hidden elements, scripts, styles, display:none) never appears in output. Table
cells are marked as table content. Line-break tags become spaces, not block
breaks. Recorded spans are trimmed to the visible sentence. Special whitespace
(non-breaking spaces, zero-width characters) normalizes consistently. Decoding
falls back safely on non-UTF-8 bytes. Normalization applied twice equals
normalization applied once.
**Why:** every one of these is a real way character positions can silently drift.
A drifted offset produces a citation that points at the wrong text while looking
perfectly healthy. This is the single failure the project cannot tolerate, so it
gets the densest test coverage. Each test also re-runs the extractor on the
recorded span and demands the exact same text back (the round-trip invariant).

### `test_sections.py` (6 tests)

Table-of-contents rows are never mistaken for headings (they live in tables). A
fake "Item 1A" appearing before the real "Item 1" loses to the longer valid chain;
this reproduces an actual decoy found in the ASTS filing. Cross-references
mid-sentence ("as discussed in Item 1A") are ignored. A long paragraph that merely
starts with "Item" is not a heading. 10-Q filings, where item numbers repeat
across Part I and Part II, are disambiguated by part markers. Every block gets a
label, tables inheriting from the surrounding section.
**Why:** "Item 1A" appears about five times per filing and only one is the real
heading. Guessing wrong would mislabel entire sections. These tests pin the known
adversarial cases so heuristic changes cannot quietly regress them.

### `test_chunking.py` (5 tests)

Chunks respect the size target. A chunk never crosses a section boundary. Every
prose block lands in exactly one chunk: none lost, none duplicated, tables
excluded. An oversized single block is flagged, not split. The chunk's recorded
span runs from its first block's start to its last block's end, and the joined
text matches.
**Why:** chunks are the units of citation. A lost block is unretrievable evidence.
A chunk spanning two sections would carry a wrong section label into answers.

### `test_bm25.py` (9 tests)

The tokenizer preserves "10-K", "BB6", "4.2", and "1a" as single search terms,
lowercases, strips curly quotes, and survives empty input. Scores are compared to
values computed by hand with a calculator, independent of the code. Rare terms
outrank common ones. Shorter documents outrank longer ones at equal term counts.
Repeating a word ten times does not score ten times higher (saturation). Documents
sharing no words with the query are never ranked at all. Ties break
deterministically. Filtered-out documents are excluded before ranking, not after.
**Why:** we wrote BM25 ourselves, so the math needs external ground truth, not a
test that just re-runs the same code. And the tokenizer is where domain
correctness lives: shred "10-K" into "10" and "K" and half the corpus's most
important queries quietly degrade.

### `test_fusion.py` (4 tests)

The merge formula is pinned to exact expected numbers on a worked example. A
document buried deep in both lists loses to a confident single #1; a document
moderately ranked in both lists beats a single #1 (both directions of the
trade-off asserted). Ties break by chunk ID. An empty list from one method leaves
the other's order intact.
**Why:** fusion is 15 lines that decide every final ranking. During development a
hand-derived expectation here was wrong and the test caught the arithmetic;
pinning exact numbers keeps the behavior frozen.

### `test_retrieval.py` (14 tests)

End to end with a controllable fake embedder. Result text is byte-identical to the
stored chunk files. The search-only company prefix ("the Company" fix) is findable
but never appears in returned text. Ticker, form, and section filters are
absolute; unknown filter values return empty, not an error. Modifying a chunk file
after the index was built makes loading fail loudly. A mismatched embedding model
is rejected. The same query returns identical results every time. Blank queries
raise; huge queries work; asking for more results than exist returns what exists;
a query sharing no vocabulary with the corpus falls back to embeddings-only.
Window splitting covers all text, and a query matching only a chunk's tail still
retrieves it. A chunk ranked #1 by both methods beats single-method chunks.
**Why:** these are the retrieval acceptance criteria, mechanized. The stale-index
test in particular guards the provenance chain: serving offsets computed against
old files is the silent-corruption scenario.

### `test_answering.py` (15 tests)

Verification: a verbatim quote passes; curly-versus-straight quote differences are
forgiven; a changed word ("grew" to "shrank") is rejected; a fabricated quote, a
five-character quote, and a citation to an unknown chunk are all rejected;
normalization is idempotent. Pipeline: verified claims carry full provenance
(offsets, file, fingerprint); the response survives a JSON round trip; one bad
claim among good ones is dropped while the good ones survive; all claims failing
converts the answer to a refusal with the answer text removed; citing a chunk the
model was never shown fails; empty retrieval refuses without spending a model
call; a model refusal passes through with its reason. Scoping: company names and
tickers are detected, ambiguous questions get no filter, and the filter actually
reaches retrieval.
**Why:** this is the "never hallucinates" gate. Every conceivable path an
unverified claim could take into a response gets a test proving the path is
closed. The fakes make model misbehavior reproducible: we can simulate a lying
model at will, which live testing cannot do on demand.

### `test_api.py` (9 tests)

Health reports the corpus. The `/ask` wire payload validates against the answering
layer's own schema, proving the API adds no reshaping. Refusals travel over HTTP
intact. Blank questions, oversized questions, and out-of-range k are rejected with
422. Search hits carry verbatim records. Chunk lookup returns the record;
unknown IDs return 404.
**Why:** the API is the contract the external harness depends on. The
schema-fidelity test means the two layers cannot drift apart without a test
failing.

### `test_pipeline_integration.py` (1 test, skips on fresh clones)

Parses the real ASTS 10-K end to end: 100 percent verification, all 23 section
headings found in order, a known sentence lands in the right section's chunk.
**Why:** synthetic fixtures cannot imitate real filer markup. This is the one test
that touches real data, and it skips cleanly where the data is absent.

### Harness tests (23, separate repository)

`test_checks.py` (15): the canonical text comparison folds unicode and whitespace;
a verbatim claim passes with the chunk fetched independently; fabricated quotes,
fingerprint mismatches, unfetchable chunks, short quotes, and malformed offsets
each fail with the right check name; citing an unretrieved chunk fails without
even fetching; expected-facts regexes match against answer plus claims; refusal
and reason-pinning work; filter-scope checking flags wrong-ticker hits.
`test_cases_and_runner.py` (8): all 18 committed case files load with unique IDs;
unknown case types, bad fields, and duplicate IDs are rejected; a grounded answer
passes a correctness case; a refusal fails it; a schema violation fails it; a
refusal passes a refusal case and an answer fails it; and a tampered quote fails a
case even when the expected facts all match.
**Why the last one matters most:** it proves citation integrity is enforced
independently of correctness. An answer that "looks right" but cannot back its
quotes still fails the audit. That is the harness's reason to exist.

## 12. How to run everything

**Setup (once).**

```bash
cd "SEC financials"
uv sync
cp .env.example .env    # edit: SEC_USER_AGENT and ANTHROPIC_API_KEY
```

In each new terminal: `set -a; source .env; set +a`

**Add more filings.** Three steps, in order:

```bash
uv run python -m filings_analyst.edgar.filings LUNR GEV     # 1. fetch
uv run python -m filings_analyst.parsing.pipeline           # 2. parse to chunks
uv run python -m filings_analyst.retrieval.index            # 3. rebuild the index
```

Any US-listed ticker works. Step 3 is mandatory after any parse; the server checks
and will refuse to start on a stale index. Re-running a fetch is free (cached).

**Use it.**

```bash
uv run python -m filings_analyst.answering.answerer "How many people work at Rocket Lab?"
uv run python -m filings_analyst.retrieval.retriever "Neutron" --k 5     # search only, free
uv run python -m filings_analyst.api.serve                               # server on :8000
```

**Test it.**

```bash
uv run pytest                                        # 106 tests, offline
uv run python -m filings_analyst.retrieval.evaluate  # retrieval scorecard
```

**Evaluate it end to end** (from the harness repo, with the server running):

```bash
uv run python -m harness.runner --base-url http://127.0.0.1:8000 --json-report report.json
```

## 13. Results summary

| What | Result |
|---|---|
| Parsing verification | 19,544 blocks and 679 chunks, 100 percent round-trip verified |
| Retrieval gold set | recall@5 0.958 hybrid, vs 0.750 keywords, 0.875 embeddings |
| Answering live check | 8/8 answered, 25/25 claims traced to raw bytes, 3/3 decoys refused |
| External harness | 18/18 cases, 23/23 independent citation checks |
| Test suites | 106/106 main, 23/23 harness, all offline |

## 14. Known gaps and next steps

1. Corpus is 3 of the 8 target tickers. The pipeline is generic; new filers may
   surface parser edge cases, which the invariant will catch loudly.
2. No XBRL numeric track yet. Precise financial figures should come from the
   structured XBRL data, not narrative text. The response schema already reserves a
   field for it.
3. One documented retrieval miss (an adversarial paraphrase). Candidate fixes: a
   larger embedding model or a reranker.
4. Refusal behavior at the boundary (plausible questions about our companies that
   filings do not answer) has only 5 probes so far. Growing the harness's case set
   is the cheapest way to keep pressure on this.
5. No CI yet. The suites are CI-ready (offline, keyless).
6. Amendments (10-K/A) are skipped; the API has no authentication (it is a local
   evaluation target, not a public service).

## 15. Glossary

| Term | Meaning |
|---|---|
| RAG | retrieve relevant documents first, then make the model answer only from them |
| EDGAR | the SEC's public filing system |
| CIK | the SEC's numeric company ID |
| 10-K / 10-Q | annual / quarterly report |
| accession number | unique ID of one filing |
| XBRL, inline XBRL | machine-readable financial tagging; inline means embedded in the readable document |
| chunk | a paragraph-sized piece of filing text with its exact source positions |
| offsets | character positions into the raw file |
| sha256 | a fingerprint of a file's exact bytes |
| round-trip invariant | slicing the raw file at a chunk's offsets and re-extracting reproduces the chunk text exactly |
| tokenizer | the rule for splitting text into searchable words |
| BM25 | classic keyword-ranking algorithm |
| embedding / vector | a text mapped to coordinates in a space where similar meanings are close |
| RRF | merging two ranked lists by position rather than score |
| recall@5 | fraction of test questions whose correct chunk appears in the top 5 results |
| gold set | hand-labeled question/answer test data |
| structured outputs | API feature forcing the model's reply to match a fixed schema |
| refusal | a first-class "cannot ground this" response with a machine-readable reason |
| black-box testing | testing a system only through its public interface, importing none of its code |
