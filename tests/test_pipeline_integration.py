"""Integration: the full parse pipeline against a real stored filing (no network).

Skips cleanly when the local store is empty (e.g. fresh clone / CI): these tests
exercise real SEC markup that unit fixtures can't fully imitate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from filings_analyst.parsing.pipeline import parse_filing

STORE = Path("data/store")
ASTS_META = sorted(STORE.glob("ASTS/narrative/10-K_*/filing.meta.json"))

pytestmark = pytest.mark.skipif(
    not ASTS_META, reason="local filing store not populated (run the fetch first)"
)


def test_asts_10k_parses_verified_with_grounded_probe(tmp_path):
    result = parse_filing(ASTS_META[0], store_root=STORE)

    # parse_filing raises VerificationError on any round-trip failure, so arriving
    # here means 100% of blocks and chunks verified.
    assert result.stats["chunks"] > 50
    assert result.stats["oversized_chunks"] == 0

    # Ground truth: the satellite launch plan sentence must be a citable chunk in
    # Item 1 (Business).
    hits = [
        c for c in result.chunks
        if "45 to 60 Block 2 BB satellites" in c.text and c.section_key == "item_1"
    ]
    assert hits, "expected the launch-plan sentence in an Item 1 chunk"

    # All 23 canonical 10-K items detected, strictly increasing.
    keys = [h.key for h in result.headings if not h.is_part_marker]
    assert len(keys) == 23
    starts = [h.char_start for h in result.headings]
    assert starts == sorted(starts)

    # Output files exist and are valid JSONL with the provenance fields.
    lines = result.chunks_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == result.stats["chunks"]
    first = json.loads(lines[0])
    for field in ("chunk_id", "cik", "accession_number", "source_sha256",
                  "char_start", "char_end", "section_key", "text"):
        assert field in first
