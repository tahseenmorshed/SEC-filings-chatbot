"""Chunking: pack consecutive prose blocks into retrieval-sized, citable units.

Strategy (structure-aware greedy packing): concatenate consecutive prose blocks up to a
target size, breaking **only at block boundaries** and **never across a section
boundary**. Because packed blocks are contiguous in document order, each chunk's
provenance stays a single clean envelope ``(first block's char_start, last block's
char_end)`` and the round-trip invariant carries over: re-extracting that raw span with
the canonical extractor (excluding table blocks, exactly as the chunker excludes them)
reproduces ``chunk.text``.

Table blocks are excluded from chunks in this milestone: the numbers-of-record travel
on the XBRL CompanyFacts track, and linearized cell fragments would pollute retrieval.
Their text is preserved at the block level, so revisiting this costs nothing.

Oversized blocks are not split (splitting inside a block would require offset math on
normalized text — deliberately out of scope while the pilots' largest block is ~2.2k
chars); a lone over-limit block becomes its own chunk flagged ``oversized``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .extract import Block, join_blocks

DEFAULT_TARGET_CHARS = 3000
DEFAULT_MAX_CHARS = 4500

_JOINER_LEN = 2  # "\n\n"


@dataclass(frozen=True)
class Chunk:
    """A citable retrieval unit with exact raw-source provenance."""

    text: str
    char_start: int
    char_end: int
    section_key: str
    section_title: str
    order: int  # position among the document's chunks
    n_blocks: int
    oversized: bool


def pack_chunks(
    blocks: list[Block],
    section_of: dict[int, tuple[str, str]],
    *,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[Chunk]:
    """Pack prose blocks into chunks; every prose block lands in exactly one chunk."""
    chunks: list[Chunk] = []
    run: list[Block] = []
    run_len = 0

    def flush() -> None:
        nonlocal run, run_len
        if not run:
            return
        key, title = section_of[run[0].order]
        text = join_blocks(run, include_tables=True)  # run is prose-only already
        chunks.append(
            Chunk(
                text=text,
                char_start=run[0].char_start,
                char_end=run[-1].char_end,
                section_key=key,
                section_title=title,
                order=len(chunks),
                n_blocks=len(run),
                oversized=len(run) == 1 and len(run[0].text) > max_chars,
            )
        )
        run, run_len = [], 0

    prev_key: str | None = None
    for b in blocks:
        if b.in_table:
            continue
        key, _ = section_of[b.order]
        if run and (
            key != prev_key or run_len + _JOINER_LEN + len(b.text) > target_chars
        ):
            flush()
        run.append(b)
        run_len += len(b.text) + (_JOINER_LEN if len(run) > 1 else 0)
        prev_key = key
    flush()
    return chunks
