"""The runner: load cases → execute against the live API → verdict + report.

Usage::

    python -m harness.runner --base-url http://127.0.0.1:8000 [--cases cases/] \
        [--json-report report.json] [--only case_id ...]

Exit code 0 iff every hard bar passes. Schema validation and citation integrity are
enforced on every response regardless of case type; soft metrics (latency, dropped
claims) are reported but never gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from .cases import (
    Case,
    CorrectnessCase,
    FilterCase,
    RefusalCase,
    RobustnessCase,
    load_cases,
)
from .checks import (
    CaseOutcome,
    CheckResult,
    check_citation_integrity,
    check_expected_facts,
    check_filter_scope,
    check_refusal,
)
from .client import HarnessClient
from .wire_schema import AnswerResponse, HealthResponse, SearchResponse


def _validate(schema, payload: dict, results: list[CheckResult]) -> bool:
    try:
        schema.model_validate(payload)
        results.append(CheckResult(True, "wire_schema"))
        return True
    except ValidationError as exc:
        results.append(CheckResult(False, "wire_schema", str(exc)[:300]))
        return False


def _ask_and_check(client: HarnessClient, question: str, k, results: list[CheckResult]) -> dict | None:
    resp = client.ask(question, k=k)
    if resp.status_code != 200:
        results.append(CheckResult(False, "http_status", f"POST /ask -> {resp.status_code}"))
        return None
    results.append(CheckResult(True, "http_status"))
    payload = resp.json()
    if not _validate(AnswerResponse, payload, results):
        return None
    if not payload["refused"]:
        results.extend(check_citation_integrity(client, payload))
    return payload


def run_case(client: HarnessClient, case: Case) -> CaseOutcome:
    results: list[CheckResult] = []
    t0 = time.time()
    payload: dict | None = None

    if isinstance(case, CorrectnessCase):
        payload = _ask_and_check(client, case.question, case.k, results)
        if payload is not None:
            results.append(CheckResult(not payload["refused"], "answered",
                                       payload.get("refusal_reason") or ""))
            if not payload["refused"]:
                results.extend(check_expected_facts(payload, case.expected_facts))

    elif isinstance(case, RefusalCase):
        payload = _ask_and_check(client, case.question, case.k, results)
        if payload is not None:
            results.extend(check_refusal(payload, case.expected_reason))

    elif isinstance(case, FilterCase):
        resp = client.search(
            case.query, k=case.k, ticker=case.expected_ticker,
            form=case.expected_form, section=case.expected_section,
        )
        ok = resp.status_code == 200
        results.append(CheckResult(ok, "http_status", f"POST /search -> {resp.status_code}"))
        if ok:
            payload = resp.json()
            if _validate(SearchResponse, payload, results):
                results.extend(check_filter_scope(
                    payload,
                    expected_ticker=case.expected_ticker,
                    expected_form=case.expected_form,
                    expected_section=case.expected_section,
                ))

    elif isinstance(case, RobustnessCase):
        refusal_states: list[bool] = []
        for q in case.questions:
            p = _ask_and_check(client, q, None, results)
            if p is None:
                continue
            refusal_states.append(p["refused"])
            if case.expected_refused:
                results.extend(check_refusal(p, None))
            elif not p["refused"]:
                results.extend(check_expected_facts(p, case.expected_facts))
            payload = p
        if len(refusal_states) == len(case.questions) and len(set(refusal_states)) > 1:
            results.append(CheckResult(
                False, "paraphrase_agreement",
                f"paraphrases disagreed on refusal: {refusal_states}",
            ))

    return CaseOutcome(
        case_id=case.id,
        case_type=type(case).__name__.removesuffix("Case").lower(),
        hard_pass=all(r.ok for r in results),
        checks=results,
        latency_s=time.time() - t0,
        raw=payload,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grounded Filings Analyst eval harness")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cases", default="cases", type=Path)
    parser.add_argument("--json-report", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="run only these case ids")
    args = parser.parse_args(argv)

    client = HarnessClient(args.base_url)

    # Preflight: the target must be up and structurally sane before any case runs.
    try:
        health = client.health()
    except Exception as exc:  # connection refused etc.
        print(f"FATAL: cannot reach {args.base_url}: {exc}", file=sys.stderr)
        return 2
    if health.status_code != 200:
        print(f"FATAL: /health -> {health.status_code}", file=sys.stderr)
        return 2
    try:
        h = HealthResponse.model_validate(health.json())
    except ValidationError as exc:
        print(f"FATAL: /health shape drift:\n{exc}", file=sys.stderr)
        return 2
    print(f"target ok: {h.chunks} chunks, tickers {','.join(h.tickers)}, "
          f"answer model {h.answer_model}\n")

    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]

    outcomes = [run_case(client, c) for c in cases]

    # ---- report -----------------------------------------------------------------
    by_type: dict[str, list[CaseOutcome]] = {}
    for o in outcomes:
        by_type.setdefault(o.case_type, []).append(o)

    total_citation_checks = 0
    failed_citation_checks = 0
    for o in outcomes:
        for c in o.checks:
            if c.check in ("citation_integrity", "quote_verbatim_in_chunk", "sha_matches",
                           "cites_retrieved_chunk", "chunk_fetchable", "offsets_present",
                           "offsets_well_formed", "quote_min_length"):
                total_citation_checks += 1
                if not c.ok:
                    failed_citation_checks += 1

    print(f"{'case':32} {'type':12} {'result':8} {'latency':>8}")
    for o in outcomes:
        print(f"{o.case_id:32} {o.case_type:12} "
              f"{'PASS' if o.hard_pass else 'FAIL':8} {o.latency_s:7.1f}s")
        if not o.hard_pass:
            for c in o.checks:
                if not c.ok:
                    print(f"    ✗ {c.check}: {c.detail}")

    n_pass = sum(1 for o in outcomes if o.hard_pass)
    print(f"\ncases: {n_pass}/{len(outcomes)} passed")
    for t, group in sorted(by_type.items()):
        p = sum(1 for o in group if o.hard_pass)
        print(f"  {t:12} {p}/{len(group)}")
    print(f"citation checks: {total_citation_checks - failed_citation_checks}"
          f"/{total_citation_checks} ok")
    lat = sorted(o.latency_s for o in outcomes)
    if lat:
        print(f"latency: median {lat[len(lat) // 2]:.1f}s, max {lat[-1]:.1f}s")

    if args.json_report:
        args.json_report.write_text(json.dumps({
            "base_url": args.base_url,
            "cases": [
                {
                    "id": o.case_id, "type": o.case_type, "pass": o.hard_pass,
                    "latency_s": round(o.latency_s, 2),
                    "checks": [
                        {"ok": c.ok, "check": c.check, "detail": c.detail}
                        for c in o.checks
                    ],
                }
                for o in outcomes
            ],
            "summary": {
                "passed": n_pass, "total": len(outcomes),
                "citation_checks_ok": total_citation_checks - failed_citation_checks,
                "citation_checks_total": total_citation_checks,
            },
        }, indent=2), encoding="utf-8")
        print(f"json report → {args.json_report}")

    return 0 if n_pass == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
