"""Retrieval index: build from chunk files, load with staleness verification.

What's persisted (data/index/ by default):

* ``records.jsonl``  — every chunk record, copied line-for-line from the source
  chunk files in deterministic order. This is the retrieval corpus; the ``text``
  field stays byte-identical to what the parser wrote.
* ``embeddings.npy`` — one unit-length float32 vector per *window* (see below).
* ``window_map.npy`` — window row → record index.
* ``manifest.json``  — model id/dim, parameters, entity names, and the sha256 of
  every source chunk file. ``load_index`` recomputes those hashes and refuses to
  load on any mismatch: a re-parsed store must never be served through offsets
  captured at an earlier build.

Two effectiveness decisions live here:

* **Enrichment** ("the Company" problem): inside ASTS's filings the text says "we",
  never "ASTS", so both search signals index an enriched representation prefixed
  with ticker/company/form/section. Only the *index* sees this; cited text is
  always the verbatim record.
* **Windows** (the truncation problem): the embedding model reads ~512 tokens but
  the median chunk is ~2.6k chars, so each chunk is embedded as 1..N overlapping
  windows (each carrying the enrichment header) and a chunk's semantic score is
  its best window's score. No part of any chunk is invisible to semantic search.
  BM25 needs none of this — it reads full text natively.

Build (downloads the model on first run, then fully offline)::

    uv run python -m filings_analyst.retrieval.index
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DEFAULT_STORE_DIR
from ..edgar.cache import atomic_write_bytes
from .bm25 import DEFAULT_B, DEFAULT_K1
from .encoders import Encoder

DEFAULT_INDEX_DIR = Path("data/index")

FORMAT_VERSION = 1

# Window geometry, in characters. Chosen by a transparent grid sweep on the gold set
# (1600/800/600 × RRF K): 800 chars balances *focus* (a pinpoint fact isn't diluted by
# surrounding text — fixes "how many employees"-style questions) against *context*
# (a sentence like "we utilize TSMC" still carries enough neighborhood to embed
# meaningfully). ~4 chars/token → ~200 tokens/window, safely inside the model's 512.
WINDOW_CHARS = 800
WINDOW_STRIDE = 600  # 200-char overlap so sentences straddling a boundary appear whole


class StaleIndexError(RuntimeError):
    """The index no longer matches the chunk files it was built from."""


def enrich_header(record: dict, entity_name: str) -> str:
    """Search-context header (company/form/section identity for both signals)."""
    return (
        f"{record['ticker']} {entity_name} | {record['form']} "
        f"filed {record['filing_date']} period {record['report_date']} | "
        f"{record['section_title']}"
    )


def enriched_text(record: dict, entity_name: str) -> str:
    """The full representation BM25 indexes (header + verbatim text)."""
    return f"{enrich_header(record, entity_name)} | {record['text']}"


def split_windows(text: str, *, size: int = WINDOW_CHARS, stride: int = WINDOW_STRIDE) -> list[str]:
    """Deterministic overlapping windows covering the whole text."""
    if len(text) <= size:
        return [text]
    windows = []
    start = 0
    while start < len(text):
        windows.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += stride
    return windows


def _entity_name(store_root: Path, ticker: str) -> str:
    """Company name from the already-fetched CompanyFacts (offline); ticker fallback."""
    facts = store_root / ticker / "facts" / "companyfacts.json"
    if facts.exists():
        try:
            name = json.loads(facts.read_text(encoding="utf-8")).get("entityName")
            if name:
                return str(name)
        except (json.JSONDecodeError, OSError):
            pass
    return ticker


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Index:
    """A loaded, verified index ready for the Retriever."""

    records: list[dict]
    enriched: list[str]  # enriched text per record (BM25 input)
    embeddings: np.ndarray  # (n_windows, dim) float32 unit-norm
    window_map: np.ndarray  # (n_windows,) int64 → record index
    entity_names: dict[str, str]
    manifest: dict


def build_index(
    encoder: Encoder,
    *,
    store_root: Path = DEFAULT_STORE_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    window_chars: int = WINDOW_CHARS,
    window_stride: int = WINDOW_STRIDE,
) -> dict:
    """Build the index from every chunks.jsonl under the store. Returns the manifest."""
    chunk_files = sorted(store_root.glob("*/chunks/*.chunks.jsonl"))
    if not chunk_files:
        raise FileNotFoundError(
            f"No chunk files under {store_root} — run the parse pipeline first."
        )

    record_lines: list[str] = []
    records: list[dict] = []
    sources: dict[str, str] = {}
    for path in chunk_files:  # sorted() → deterministic corpus order
        sources[str(path.relative_to(store_root))] = _sha256_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            record_lines.append(line)  # verbatim copy — text field untouched
            records.append(json.loads(line))

    entity_names = {
        t: _entity_name(store_root, t) for t in sorted({r["ticker"] for r in records})
    }

    # Window texts: header + slice of the verbatim text (header on every window so
    # each carries company/section identity).
    window_texts: list[str] = []
    window_map: list[int] = []
    for i, record in enumerate(records):
        header = enrich_header(record, entity_names[record["ticker"]])
        for w in split_windows(record["text"], size=window_chars, stride=window_stride):
            window_texts.append(f"{header} | {w}")
            window_map.append(i)

    embeddings = encoder.encode_passages(window_texts)
    if embeddings.shape != (len(window_texts), encoder.dim):
        raise RuntimeError(
            f"encoder returned {embeddings.shape}, "
            f"expected {(len(window_texts), encoder.dim)}"
        )

    index_dir.mkdir(parents=True, exist_ok=True)
    records_blob = ("\n".join(record_lines) + "\n").encode("utf-8")
    atomic_write_bytes(index_dir / "records.jsonl", records_blob)
    np.save(index_dir / "embeddings.npy", embeddings)
    np.save(index_dir / "window_map.npy", np.array(window_map, dtype=np.int64))

    manifest = {
        "format_version": FORMAT_VERSION,
        "model_id": encoder.model_id,
        "dim": encoder.dim,
        "params": {
            "k1": DEFAULT_K1,
            "b": DEFAULT_B,
            "window_chars": window_chars,
            "window_stride": window_stride,
        },
        "entity_names": entity_names,
        "sources": sources,
        "records_sha256": hashlib.sha256(records_blob).hexdigest(),
        "counts": {"records": len(records), "windows": len(window_texts)},
    }
    atomic_write_bytes(
        index_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return manifest


def load_index(
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
    store_root: Path = DEFAULT_STORE_DIR,
    expected_model_id: str | None = None,
) -> Index:
    """Load and verify the index; raises StaleIndexError on any drift."""
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No index at {index_dir} — build one: "
            f"uv run python -m filings_analyst.retrieval.index"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest["format_version"] != FORMAT_VERSION:
        raise StaleIndexError(
            f"Index format v{manifest['format_version']} != v{FORMAT_VERSION}; rebuild."
        )
    if expected_model_id is not None and manifest["model_id"] != expected_model_id:
        raise StaleIndexError(
            f"Index built with {manifest['model_id']!r} but encoder is "
            f"{expected_model_id!r}; rebuild."
        )

    # Staleness: every source chunk file must still exist with identical bytes.
    for rel_path, expected_sha in manifest["sources"].items():
        src = store_root / rel_path
        if not src.exists() or _sha256_file(src) != expected_sha:
            raise StaleIndexError(
                f"Chunk file {rel_path} changed (or vanished) since the index was "
                f"built — offsets may no longer match. Rebuild: "
                f"uv run python -m filings_analyst.retrieval.index"
            )

    records_blob = (index_dir / "records.jsonl").read_bytes()
    if hashlib.sha256(records_blob).hexdigest() != manifest["records_sha256"]:
        raise StaleIndexError("records.jsonl corrupted (sha mismatch); rebuild.")

    records = [json.loads(line) for line in records_blob.decode("utf-8").splitlines()]
    embeddings = np.load(index_dir / "embeddings.npy")
    window_map = np.load(index_dir / "window_map.npy")

    if len(records) != manifest["counts"]["records"]:
        raise StaleIndexError("record count mismatch vs manifest; rebuild.")
    if embeddings.shape[0] != window_map.shape[0] != manifest["counts"]["windows"]:
        raise StaleIndexError("embeddings/window_map size mismatch; rebuild.")
    if window_map.size and int(window_map.max()) >= len(records):
        raise StaleIndexError("window_map points past the record list; rebuild.")

    entity_names = manifest["entity_names"]
    enriched = [enriched_text(r, entity_names[r["ticker"]]) for r in records]
    return Index(
        records=records,
        enriched=enriched,
        embeddings=embeddings.astype(np.float32),
        window_map=window_map.astype(np.int64),
        entity_names=entity_names,
        manifest=manifest,
    )


def _main() -> int:
    from .encoders import FastEmbedEncoder

    print("Loading embedding model (downloads ~80MB on first run)...")
    encoder = FastEmbedEncoder()
    print("Building index...")
    manifest = build_index(encoder)
    c = manifest["counts"]
    print(
        f"Indexed {c['records']} chunks as {c['windows']} embedding windows "
        f"({manifest['model_id']}) → {DEFAULT_INDEX_DIR}/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
