"""Parse pipeline: stored raw filings → verified, section-labeled chunk files.

For each narrative filing in the store this: decodes the raw document, extracts
offset-anchored blocks, detects section headings, packs prose blocks into chunks, then
**verifies the round-trip invariant for every block and every chunk** before writing
anything. Any verification failure aborts with a non-zero exit — a silently-wrong
citation must never reach disk.

Outputs, per filing (deterministic bytes — no timestamps — so re-runs are diffable)::

    <store>/<TICKER>/chunks/<FORM>_<accession>.chunks.jsonl    # one chunk per line
    <store>/<TICKER>/chunks/<FORM>_<accession>.manifest.json   # stats + verification

Run::

    uv run python -m filings_analyst.parsing.pipeline           # everything in store
    uv run python -m filings_analyst.parsing.pipeline NVDA      # one ticker
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DEFAULT_STORE_DIR
from ..edgar.cache import atomic_write_bytes
from .chunking import DEFAULT_MAX_CHARS, DEFAULT_TARGET_CHARS, Chunk, pack_chunks
from .extract import Block, decode_filing, extract_blocks, verify_span
from .sections import SectionHeading, assign_sections, detect_headings


class VerificationError(RuntimeError):
    """A block or chunk failed the round-trip invariant."""


@dataclass
class ParseResult:
    filing_meta: dict
    encoding: str
    blocks: list[Block]
    headings: list[SectionHeading]
    chunks: list[Chunk]
    chunks_path: Path
    manifest_path: Path
    stats: dict = field(default_factory=dict)


def parse_filing(
    meta_path: Path,
    *,
    store_root: Path,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> ParseResult:
    """Parse one stored filing (identified by its filing.meta.json) into chunks."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    doc_path = meta_path.parent / meta["stored_filename"]
    raw = doc_path.read_bytes()

    # Integrity gate: the bytes we parse must be the bytes ingestion recorded.
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != meta["sha256"]:
        raise VerificationError(
            f"{doc_path}: stored bytes do not match ingestion sha256 "
            f"(expected {meta['sha256']}, got {actual_sha})"
        )

    doc_text, encoding = decode_filing(raw)
    blocks = extract_blocks(doc_text)
    headings = detect_headings(blocks, meta["form"])
    section_of = assign_sections(blocks, headings)
    chunks = pack_chunks(
        blocks, section_of, target_chars=target_chars, max_chars=max_chars
    )

    # The round-trip invariant, enforced for every block and every chunk.
    for b in blocks:
        if not verify_span(
            doc_text, b.char_start, b.char_end, b.text, include_tables=True
        ):
            raise VerificationError(
                f"{doc_path}: block {b.order} failed round-trip at "
                f"({b.char_start}, {b.char_end})"
            )
    for c in chunks:
        if not verify_span(
            doc_text, c.char_start, c.char_end, c.text, include_tables=False
        ):
            raise VerificationError(
                f"{doc_path}: chunk {c.order} failed round-trip at "
                f"({c.char_start}, {c.char_end})"
            )

    # --- write outputs (deterministic bytes) -------------------------------------
    ticker = meta["ticker"]
    acc_nodash = meta["accession_number"].replace("-", "")
    out_dir = store_root / ticker / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['form']}_{acc_nodash}"
    chunks_path = out_dir / f"{stem}.chunks.jsonl"
    manifest_path = out_dir / f"{stem}.manifest.json"

    source_path = str(doc_path.relative_to(store_root))
    lines = []
    for c in chunks:
        record = {
            "chunk_id": f"{acc_nodash}-{c.order:05d}",
            "ticker": ticker,
            "cik": meta["cik"],
            "form": meta["form"],
            "filing_date": meta["filing_date"],
            "report_date": meta["report_date"],
            "accession_number": meta["accession_number"],
            "source_path": source_path,
            "source_sha256": meta["sha256"],
            "encoding": encoding,
            "section_key": c.section_key,
            "section_title": c.section_title,
            "order": c.order,
            "n_blocks": c.n_blocks,
            "oversized": c.oversized,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "text": c.text,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    atomic_write_bytes(chunks_path, ("\n".join(lines) + "\n").encode("utf-8"))

    prose = [b for b in blocks if not b.in_table]
    stats = {
        "blocks": len(blocks),
        "prose_blocks": len(prose),
        "table_blocks": len(blocks) - len(prose),
        "prose_chars": sum(len(b.text) for b in prose),
        "chunks": len(chunks),
        "oversized_chunks": sum(1 for c in chunks if c.oversized),
        "sections_detected": sum(1 for h in headings if not h.is_part_marker),
    }
    manifest = {
        "ticker": ticker,
        "cik": meta["cik"],
        "form": meta["form"],
        "accession_number": meta["accession_number"],
        "source_path": source_path,
        "source_sha256": meta["sha256"],
        "encoding": encoding,
        "params": {"target_chars": target_chars, "max_chars": max_chars},
        "counts": stats,
        "headings": [
            {
                "key": h.key,
                "title": h.title,
                "char_start": h.char_start,
                "block_order": h.block_order,
            }
            for h in headings
        ],
        "verification": {
            "blocks_checked": len(blocks),
            "blocks_passed": len(blocks),
            "chunks_checked": len(chunks),
            "chunks_passed": len(chunks),
        },
    }
    atomic_write_bytes(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )

    return ParseResult(
        filing_meta=meta,
        encoding=encoding,
        blocks=blocks,
        headings=headings,
        chunks=chunks,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        stats=stats,
    )


def _main(argv: list[str]) -> int:
    store_root = DEFAULT_STORE_DIR
    tickers = {t.upper() for t in argv} or None

    meta_paths = sorted(store_root.glob("*/narrative/*/filing.meta.json"))
    if tickers is not None:
        meta_paths = [p for p in meta_paths if p.parts[-4] in tickers]
    if not meta_paths:
        print(f"No stored filings found under {store_root}", file=sys.stderr)
        return 1

    failures = 0
    for meta_path in meta_paths:
        try:
            result = parse_filing(meta_path, store_root=store_root)
        except VerificationError as exc:
            failures += 1
            print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
            continue
        meta, s = result.filing_meta, result.stats
        heads = [h.key for h in result.headings if not h.is_part_marker]
        print(
            f"{meta['ticker']:5} {meta['form']:5} {meta['accession_number']}  "
            f"blocks={s['blocks']:5,} chunks={s['chunks']:4,} "
            f"sections={s['sections_detected']:2} "
            f"verified: blocks {s['blocks']}/{s['blocks']} ✓, "
            f"chunks {s['chunks']}/{s['chunks']} ✓"
        )
        print(f"      sections: {', '.join(heads)}")
        print(f"      → {result.chunks_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
