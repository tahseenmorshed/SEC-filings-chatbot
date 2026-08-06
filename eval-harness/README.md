# Evaluation Harness

This is a black-box tester for the Grounded Filings Analyst in the parent directory. It
drives the running system over its HTTP API and imports none of its code. The analyst
claims that it never presents an unverified statement. This harness checks that claim
from the outside, one response at a time, trusting nothing the system says about itself.

It lives in the same repository for convenience, but it is a separate project with its
own dependencies and its own virtual environment. That separation is deliberate and
worth preserving.

## Why it stays independent

If a tester imports the code it is testing, it inherits that code's blind spots. So this
harness talks only HTTP. It even re-declares the response shapes it expects, in
`harness/wire_schema.py`, rather than importing them from the analyst. If the API's
shape ever drifts from what a consumer would reasonably expect, that drift shows up as a
failed test instead of being silently shared.

You can verify the independence at any time:

```bash
grep -r "filings_analyst" harness/ tests/     # should return nothing
```

## How it works

Test cases are plain YAML files, one case per file, in `cases/`. Because they are data
rather than code, a runner written in any language could execute the same cases against
the same API. The Python runner here is just a reference implementation.

There are four kinds of case:

- **correctness**: ask a question that the filings can answer. The system must not
  refuse, and the facts we expect must appear in the answer.
- **refusal**: ask something the filings cannot ground. The system must refuse. This
  includes questions about companies not in the corpus, plausible-sounding questions the
  filings do not actually answer (like a stock price on a specific date), and confusion
  traps (a question about a company whose competitors are in the corpus).
- **filter**: run a scoped search. Every result must match the scope, with no leaks.
- **robustness**: ask the same question several different ways. All the phrasings must
  agree. Answers stay answers with the same facts, and refusals stay refusals.

On top of whatever a case checks, every answer that is not a refusal gets a citation
audit. For each claim, the harness fetches the cited chunk fresh through
`GET /chunks/{id}`, re-checks that the quote really appears in it, compares the file
fingerprints, and validates the character offsets. The system's own word is never taken
for granted.

## Running it

First install the harness dependencies, which are separate from the analyst's:

```bash
cd eval-harness
uv sync
```

In one terminal, start the analyst from the repository root with its `.env` loaded:

```bash
cd ..
set -a; source .env; set +a
uv run python -m filings_analyst.api.serve
```

In another terminal, run the harness:

```bash
cd eval-harness
uv run python -m harness.runner --base-url http://127.0.0.1:8000 --json-report report.json
```

The exit code is 0 only if every case passes. Use `--only case_id ...` to run a subset.
A full run makes about 19 answering calls, which costs roughly a dollar at the analyst's
observed rate of about 5 cents per question.

The harness has its own test suite that runs offline against a fake client, with no
network and no key:

```bash
uv run pytest
```

## Reading the results

For each case you get PASS or FAIL, the latency, and every failed check with detail. At
the end you get pass counts by type, a total for the citation checks, and latency
figures. The `--json-report` flag writes the full result of every check to a file for
machines to read.

## First live run

18 of 18 cases passed, with all 23 independent citation checks passing. The refusal
cases held under rephrasing and under the confusion trap. Median answer latency was 4.6
seconds.
