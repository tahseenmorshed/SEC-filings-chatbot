"""FilingFetcher/FilingStore: selection logic, URL construction, byte-exact storage.

Served entirely by httpx.MockTransport — no real SEC calls.
"""

from __future__ import annotations

import hashlib
import json

import httpx

from filings_analyst.edgar.client import SECClient
from filings_analyst.edgar.filings import FilingFetcher, FilingStore
from filings_analyst.edgar.throttle import RateLimiter

CIK = "0001045810"

# Newest-first, like the real endpoint. Includes decoys that exact-match must skip:
# an 8-K (wrong form), a 10-K/A (amendment), and an older 10-K that must NOT win.
SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["8-K", "10-Q", "10-K/A", "10-K", "10-Q", "10-K"],
            "filingDate": [
                "2026-06-01",
                "2026-05-28",
                "2026-04-02",
                "2026-02-26",
                "2025-11-19",
                "2025-02-26",
            ],
            "reportDate": [
                "2026-06-01",
                "2026-04-27",
                "2026-01-25",
                "2026-01-25",
                "2025-10-26",
                "2025-01-26",
            ],
            "accessionNumber": [
                "0001045810-26-000090",
                "0001045810-26-000080",
                "0001045810-26-000050",
                "0001045810-26-000023",
                "0001045810-25-000200",
                "0001045810-25-000023",
            ],
            "primaryDocument": [
                "nvda-8k.htm",
                "nvda-20260427.htm",
                "nvda-10ka.htm",
                "nvda-20260125.htm",
                "nvda-20251026.htm",
                "nvda-20250126.htm",
            ],
        }
    }
}

# Adversarial narrative body: BOM, multibyte, CRLF, trailing spaces, no final newline.
DOC_BODY = b"\xef\xbb\xbf<html>Revenue \xe2\x80\x94 grew\r\n<b>12%</b>   \n</html>"
FACTS_BODY = json.dumps({"cik": 1045810, "facts": {"us-gaap": {}}}).encode()


def _handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url.endswith(f"/submissions/CIK{CIK}.json"):
        return httpx.Response(200, content=json.dumps(SUBMISSIONS).encode())
    if "/Archives/edgar/data/" in url:
        return httpx.Response(200, content=DOC_BODY)
    if url.endswith(f"/api/xbrl/companyfacts/CIK{CIK}.json"):
        return httpx.Response(200, content=FACTS_BODY)
    return httpx.Response(404)


def _fetcher(settings, tmp_path) -> tuple[FilingFetcher, FilingStore, SECClient]:
    client = SECClient(
        settings,
        limiter=RateLimiter(8.0, monotonic=lambda: 0.0, sleep=lambda _s: None),
        transport=httpx.MockTransport(_handler),
    )
    store = FilingStore(tmp_path / "store")
    return FilingFetcher(client, store), store, client


def test_selects_newest_exact_form_matches_only(settings, tmp_path):
    fetcher, _, client = _fetcher(settings, tmp_path)
    with client:
        records = fetcher.latest_filings("NVDA", CIK)

    by_form = {r.form: r for r in records}
    assert set(by_form) == {"10-K", "10-Q"}
    # The 10-K/A amendment (2026-04-02) must be skipped; the newest exact 10-K wins.
    assert by_form["10-K"].filing_date == "2026-02-26"
    assert by_form["10-K"].accession_number == "0001045810-26-000023"
    # The newest 10-Q, not the older one.
    assert by_form["10-Q"].filing_date == "2026-05-28"


def test_archive_url_uses_unpadded_cik_and_dashless_accession(settings, tmp_path):
    fetcher, _, client = _fetcher(settings, tmp_path)
    with client:
        records = fetcher.latest_filings("NVDA", CIK)

    ten_k = next(r for r in records if r.form == "10-K")
    assert ten_k.source_url == (
        "https://www.sec.gov/Archives/edgar/data/1045810"
        "/000104581026000023/nvda-20260125.htm"
    )


def test_fetch_company_stores_narrative_byte_for_byte(settings, tmp_path):
    fetcher, store, client = _fetcher(settings, tmp_path)
    with client:
        summary = fetcher.fetch_company("NVDA", CIK)

    assert summary["missing_forms"] == []
    for item in summary["narrative"]:
        stored = item["path"].read_bytes()
        assert stored == DOC_BODY  # exact bytes, unchanged

        meta = json.loads((item["path"].parent / "filing.meta.json").read_text())
        assert meta["ticker"] == "NVDA"
        assert meta["cik"] == CIK
        assert meta["form"] in {"10-K", "10-Q"}
        assert meta["accession_number"].count("-") == 2  # dashed form preserved in meta
        assert meta["sha256"] == hashlib.sha256(DOC_BODY).hexdigest()
        assert meta["source_url"].startswith("https://www.sec.gov/Archives/")
        assert meta["report_date"]  # period of report present


def test_facts_track_is_stored_separately(settings, tmp_path):
    fetcher, store, client = _fetcher(settings, tmp_path)
    with client:
        summary = fetcher.fetch_company("NVDA", CIK)

    facts_path = summary["facts"]["path"]
    assert facts_path.read_bytes() == FACTS_BODY
    # Facts live under facts/, never mingled with narrative/.
    assert facts_path.parent.name == "facts"
    assert "narrative" not in facts_path.parts

    meta = json.loads((facts_path.parent / "companyfacts.meta.json").read_text())
    assert meta["cik"] == CIK
    assert meta["source_url"].endswith(f"/api/xbrl/companyfacts/CIK{CIK}.json")


def test_missing_form_is_reported_not_fatal(settings, tmp_path):
    """A company that has never filed a 10-K (e.g. too new) still fetches cleanly."""
    no_10k = {
        "filings": {
            "recent": {
                "form": ["10-Q"],
                "filingDate": ["2026-05-01"],
                "reportDate": ["2026-03-31"],
                "accessionNumber": ["0001234567-26-000001"],
                "primaryDocument": ["newco-10q.htm"],
            }
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/submissions/" in url:
            return httpx.Response(200, content=json.dumps(no_10k).encode())
        if "/Archives/" in url:
            return httpx.Response(200, content=b"<html>10-Q</html>")
        return httpx.Response(200, content=FACTS_BODY)

    client = SECClient(
        settings,
        limiter=RateLimiter(8.0, monotonic=lambda: 0.0, sleep=lambda _s: None),
        transport=httpx.MockTransport(handler),
    )
    with client:
        fetcher = FilingFetcher(client, FilingStore(tmp_path / "store"))
        summary = fetcher.fetch_company("NEWCO", "0001234567")

    assert summary["missing_forms"] == ["10-K"]
    assert [i["form"] for i in summary["narrative"]] == ["10-Q"]
    assert summary["facts"] is not None
