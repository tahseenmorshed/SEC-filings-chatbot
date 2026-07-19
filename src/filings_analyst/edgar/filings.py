"""Filing retrieval: narrative documents (10-K/10-Q) and structured XBRL facts.

Two deliberately separate tracks, both flowing through :class:`SECClient` (throttled,
cached, User-Agent-enforced — no raw calls):

* **Narrative** — the Submissions endpoint lists a company's filing history as parallel
  arrays sorted newest-first. We select the most recent 10-K and 10-Q, resolve the
  primary document's URL under the Archives path, and store the document **byte-for-byte
  as served** — no stripping, entity decoding, or reformatting. Downstream provenance
  (exact character offsets into this text) depends on the stored file being an exact
  copy. A small metadata record sits alongside each document.

* **Facts** — the CompanyFacts endpoint is the numbers source of truth (revenue, net
  income, share counts). Its raw JSON is stored under a separate ``facts/`` subtree,
  never mingled with narrative files.

Non-obvious decisions, spelled out:

* **Primary document selection.** A filing (one accession number) is a *folder* of many
  sub-documents: the main document, dozens of exhibits, images, and XBRL companions.
  We do not guess among them — the SEC designates the main document itself in the
  Submissions metadata's ``primaryDocument`` field, and we take their word for it.

* **Exact form match.** ``form == "10-K"`` exactly, so amendments (``10-K/A``) and
  related types (``10-K405``, ``NT 10-K``) are excluded for now. An amendment supersedes
  parts of the original, but handling that correctly is a parsing-stage concern; at the
  fetch stage we want the canonical original.

* **Recency window.** ``filings.recent`` covers a company's ~1000 most recent filings —
  years of history, far more than enough to contain the latest 10-K and 10-Q — so we
  never need the paged archive files.

* **Store vs. cache.** Bytes land in two places on purpose. ``data/cache/`` is
  transport-level dedupe keyed by URL hash (opaque, safe to wipe). ``data/store/`` is
  the ingestion *product*: a browsable, ticker-keyed corpus that later stages read.

Run as a script (defaults to the three-company pilot set)::

    uv run python -m filings_analyst.edgar.filings            # NVDA ASTS RKLB
    uv run python -m filings_analyst.edgar.filings NVDA       # one company
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import (
    Settings,
    archive_document_url,
    company_facts_url,
    submissions_url,
)
from .cache import atomic_write_bytes
from .client import SECClient
from .resolver import CikResolver

# The narrative forms we care about for now. Exact matches only (see module docstring).
TARGET_FORMS = ("10-K", "10-Q")

# Deliberately three structurally diverse companies (large/clean, small/new, middle) so
# real markup variation is inspected before any parser is written. Not all eight yet.
PILOT_TICKERS = ["NVDA", "ASTS", "RKLB"]


@dataclass(frozen=True)
class FilingRecord:
    """Provenance metadata for one fetched narrative filing."""

    ticker: str
    cik: str  # 10-digit zero-padded
    form: str
    filing_date: str
    report_date: str  # period of report (may be empty for some filings)
    accession_number: str
    primary_document: str  # SEC's designated main document filename
    source_url: str


class FilingStore:
    """Ticker-keyed on-disk corpus of raw filings.

    Layout::

        <root>/<TICKER>/narrative/<FORM>_<filing-date>_<accession>/<original-filename>
        <root>/<TICKER>/narrative/<FORM>_<filing-date>_<accession>/filing.meta.json
        <root>/<TICKER>/facts/companyfacts.json
        <root>/<TICKER>/facts/companyfacts.meta.json
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def narrative_dir(self, record: FilingRecord) -> Path:
        # Accession without dashes keeps the folder name filesystem-friendly; the
        # dashed form is preserved in the metadata record.
        folder = (
            f"{record.form}_{record.filing_date}_"
            f"{record.accession_number.replace('-', '')}"
        )
        return self.root / record.ticker.upper() / "narrative" / folder

    def save_narrative(self, record: FilingRecord, body: bytes) -> Path:
        """Store a narrative document byte-for-byte plus its metadata sidecar."""
        directory = self.narrative_dir(record)
        directory.mkdir(parents=True, exist_ok=True)

        # The original SEC filename is preserved — it is itself provenance.
        doc_path = directory / record.primary_document
        atomic_write_bytes(doc_path, body)

        meta = {
            **asdict(record),
            # Integrity anchor: lets any later stage verify it is reading the exact
            # bytes these offsets were computed against.
            "sha256": hashlib.sha256(body).hexdigest(),
            "size_bytes": len(body),
            "stored_filename": record.primary_document,
        }
        atomic_write_bytes(
            directory / "filing.meta.json",
            json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        return doc_path

    def save_company_facts(
        self, ticker: str, cik: str, body: bytes, source_url: str
    ) -> Path:
        """Store the raw CompanyFacts JSON, byte-for-byte, in the facts subtree."""
        directory = self.root / ticker.upper() / "facts"
        directory.mkdir(parents=True, exist_ok=True)

        facts_path = directory / "companyfacts.json"
        atomic_write_bytes(facts_path, body)
        meta = {
            "ticker": ticker.upper(),
            "cik": cik,
            "source_url": source_url,
            "sha256": hashlib.sha256(body).hexdigest(),
            "size_bytes": len(body),
        }
        atomic_write_bytes(
            directory / "companyfacts.meta.json",
            json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        return facts_path


class FilingFetcher:
    """Fetches both retrieval tracks for a company through the shared SECClient."""

    def __init__(self, client: SECClient, store: FilingStore) -> None:
        self._client = client
        self._store = store

    def latest_filings(self, ticker: str, cik: str) -> list[FilingRecord]:
        """The most recent filing of each TARGET_FORM, from the Submissions endpoint."""
        submissions = self._client.get_json(submissions_url(cik))
        recent = submissions["filings"]["recent"]
        forms: list[str] = recent["form"]

        records = []
        for target in TARGET_FORMS:
            try:
                # Parallel arrays are sorted newest-first, so the first exact match is
                # the most recent filing of that form.
                i = forms.index(target)
            except ValueError:
                continue  # company has never filed this form (reported by caller)
            accession = recent["accessionNumber"][i]
            primary_doc = recent["primaryDocument"][i]
            records.append(
                FilingRecord(
                    ticker=ticker.upper(),
                    cik=cik,
                    form=target,
                    filing_date=recent["filingDate"][i],
                    report_date=recent["reportDate"][i],
                    accession_number=accession,
                    primary_document=primary_doc,
                    source_url=archive_document_url(cik, accession, primary_doc),
                )
            )
        return records

    def fetch_company(self, ticker: str, cik: str) -> dict:
        """Fetch both tracks for one company. Returns a summary for display."""
        summary: dict = {"ticker": ticker.upper(), "cik": cik, "narrative": [], "facts": None}

        # Track 1: narrative filings (10-K, 10-Q).
        records = self.latest_filings(ticker, cik)
        for record in records:
            body = self._client.get_bytes(record.source_url)
            path = self._store.save_narrative(record, body)
            summary["narrative"].append(
                {
                    "form": record.form,
                    "filing_date": record.filing_date,
                    "report_date": record.report_date,
                    "path": path,
                    "size_bytes": len(body),
                }
            )
        found_forms = {r.form for r in records}
        summary["missing_forms"] = [f for f in TARGET_FORMS if f not in found_forms]

        # Track 2: structured XBRL facts — kept entirely separate.
        facts_url = company_facts_url(cik)
        facts_body = self._client.get_bytes(facts_url)
        facts_path = self._store.save_company_facts(ticker, cik, facts_body, facts_url)
        summary["facts"] = {"path": facts_path, "size_bytes": len(facts_body)}

        return summary


def _human_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def _main(argv: list[str]) -> None:
    tickers = [t.upper() for t in argv] or list(PILOT_TICKERS)
    settings = Settings()  # raises with guidance if SEC_USER_AGENT is unset/invalid

    with SECClient(settings) as client:
        resolver = CikResolver(client)
        fetcher = FilingFetcher(client, FilingStore(settings.store_dir))

        print(f"Fetching filings for: {', '.join(tickers)}\n")
        for ticker in tickers:
            cik = resolver.resolve(ticker)
            summary = fetcher.fetch_company(ticker, cik)

            print(f"{ticker} (CIK {cik})")
            for item in summary["narrative"]:
                period = f", period {item['report_date']}" if item["report_date"] else ""
                print(
                    f"  {item['form']:<5} filed {item['filing_date']}{period}"
                    f"  →  {item['path']}  ({_human_size(item['size_bytes'])})"
                )
            for missing in summary["missing_forms"]:
                print(f"  {missing:<5} — none found in recent filing history")
            facts = summary["facts"]
            print(
                f"  facts →  {facts['path']}  ({_human_size(facts['size_bytes'])})"
            )
            print()


if __name__ == "__main__":
    _main(sys.argv[1:])
