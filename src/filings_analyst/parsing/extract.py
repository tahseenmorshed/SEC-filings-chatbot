"""Offset-preserving text extraction from raw filing HTML (inline XBRL).

Why not BeautifulSoup/lxml: they extract clean text but never expose *where* in the raw
file each piece came from, and our citations depend on exact source offsets. Instead we
drive the stdlib's streaming ``html.parser.HTMLParser``: it fires an event per token in
strict left-to-right order, and ``getpos()`` reports each event's position in the input.

Offset strategy ("next event ends the previous one"): entity references decode to fewer
characters than their raw spelling (``&#8220;`` → one curly quote), so doing arithmetic
on *decoded* lengths silently drifts. We never do. Each text fragment's raw end is
taken from the *start position of the following event* — measured purely in raw
coordinates. The events are contiguous, so this is exact by construction.

The round-trip invariant this module guarantees (and :func:`verify_span` checks):

    extract on doc_text[b.char_start:b.char_end]  ==  b.text

for every emitted block. If our bookkeeping were ever wrong anywhere, this check fails
loudly at build time — the invariant is self-validating.

Definitions:

* Offsets are **character offsets into the decoded document text** (see
  :func:`decode_filing`; the decode is deterministic, and the raw bytes' sha256 is
  recorded at ingestion, so decoded coordinates are stable provenance).
* Canonical text normalization: entities decoded, zero-width characters removed,
  whitespace runs (including non-breaking spaces) collapsed to single spaces, ends
  stripped. Deterministic, applied identically at build and verify time.
"""

from __future__ import annotations

import html as html_mod
import re
from dataclasses import dataclass
from html.parser import HTMLParser

# Content invisible to a human reader of the filing. ix:header holds the XBRL context
# machinery (hundreds of KB of machine data in real filings) and must never leak into
# chunks. Tag names arrive lowercased from html.parser.
SUPPRESS_TAGS = {"script", "style", "title", "ix:header", "ix:hidden"}

# Tags that terminate the current text block when they open or close. A <span> is
# deliberately absent: filers split single sentences across many spans.
BLOCK_TAGS = {
    "p", "div", "li", "ul", "ol", "table", "thead", "tbody", "tfoot",
    "tr", "td", "th", "caption", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "hr", "section", "article",
}

# Void elements never receive a closing tag and must not join the open-element stack.
VOID_TAGS = {
    "br", "img", "hr", "meta", "link", "input", "area", "base", "col",
    "embed", "source", "track", "wbr",
}

_DISPLAY_NONE = re.compile(r"display\s*:\s*none", re.IGNORECASE)


def decode_filing(raw: bytes) -> tuple[str, str]:
    """Deterministically decode raw filing bytes; returns (text, encoding_used).

    Strict UTF-8 first (a superset of the ASCII most filings declare); latin-1 as the
    total fallback (it maps every byte, so it cannot fail and stays deterministic).
    """
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def normalize_text(s: str) -> str:
    """Canonical whitespace normalization (part of the round-trip definition).

    Zero-width characters are removed; any whitespace run (including NBSP, which
    ``str.split`` treats as whitespace) collapses to a single space; ends stripped.
    """
    s = s.replace("\u200b", "").replace("\ufeff", "")  # zero-width space, BOM/ZWNBSP
    return " ".join(s.split())


@dataclass(frozen=True)
class Block:
    """One readable run of text with its exact raw-source span."""

    text: str  # normalized, human-readable
    char_start: int  # offsets into the decoded document text
    char_end: int
    in_table: bool
    order: int


class _Extractor(HTMLParser):
    """Streaming extractor. Feed the whole document once, then close()."""

    def __init__(self, doc_text: str) -> None:
        # convert_charrefs=False so entity references arrive as their own events with
        # their own positions, keeping all offset math in raw coordinates.
        super().__init__(convert_charrefs=False)
        self._doc_len = len(doc_text)

        # Absolute offset of each line start: getpos() speaks (line, col).
        self._line_starts = [0]
        idx = doc_text.find("\n")
        while idx != -1:
            self._line_starts.append(idx + 1)
            idx = doc_text.find("\n", idx + 1)

        self._stack: list[tuple[str, bool]] = []  # (tag, suppresses)
        self._suppress_count = 0
        self._table_depth = 0

        # Fragment awaiting its end position: (abs_start, decoded_text, is_literal).
        # is_literal means decoded chars == raw chars (plain data, no entity).
        self._pending: tuple[int, str, bool] | None = None
        self._frags: list[tuple[int, int, str, bool]] = []  # (start, end, text, literal)
        self._blocks: list[Block] = []

    # -- position bookkeeping ----------------------------------------------------
    def _abs(self) -> int:
        line, col = self.getpos()
        return self._line_starts[line - 1] + col

    def _event(self) -> int:
        """Called first in every handler: the current event ends any pending fragment."""
        abs_ = self._abs()
        if self._pending is not None:
            start, text, literal = self._pending
            self._pending = None
            self._frags.append((start, abs_, text, literal))
        return abs_

    @property
    def _suppressed(self) -> bool:
        return self._suppress_count > 0

    # -- block assembly ------------------------------------------------------------
    def _flush_block(self) -> None:
        frags = self._frags
        self._frags = []
        # Drop whitespace-only literal fragments at the edges, and trim edge whitespace
        # inside literal fragments (safe: literal fragments have raw chars == decoded
        # chars, so trimming is exact offset arithmetic; entity fragments are never
        # trimmed).
        while frags and frags[0][3] and not frags[0][2].strip():
            frags.pop(0)
        while frags and frags[-1][3] and not frags[-1][2].strip():
            frags.pop()
        if not frags:
            return
        start, end = frags[0][0], frags[-1][1]
        if frags[0][3]:
            start += len(frags[0][2]) - len(frags[0][2].lstrip())
        if frags[-1][3]:
            end -= len(frags[-1][2]) - len(frags[-1][2].rstrip())
        text = normalize_text("".join(f[2] for f in frags))
        if text:
            self._blocks.append(
                Block(
                    text=text,
                    char_start=start,
                    char_end=end,
                    in_table=self._table_depth > 0,
                    order=len(self._blocks),
                )
            )

    # -- tag handlers ---------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._event()
        if tag == "br":  # line break within a paragraph: whitespace, not a block break
            if not self._suppressed:
                self._pending = (self._abs(), " ", False)
            return
        if tag in BLOCK_TAGS:
            self._flush_block()
            if tag == "table":
                self._table_depth += 1
        if tag in VOID_TAGS:
            return
        suppresses = tag in SUPPRESS_TAGS or any(
            name == "style" and value and _DISPLAY_NONE.search(value)
            for name, value in attrs
        )
        self._stack.append((tag, suppresses))
        if suppresses:
            self._suppress_count += 1

    def handle_endtag(self, tag: str) -> None:
        self._event()
        if tag in BLOCK_TAGS:
            self._flush_block()
            if tag == "table" and self._table_depth:
                self._table_depth -= 1
        # Pop to the matching open tag (tolerant of stray end tags).
        if any(t == tag for t, _ in self._stack):
            while self._stack:
                t, suppresses = self._stack.pop()
                if suppresses:
                    self._suppress_count -= 1
                if t == tag:
                    break

    def handle_startendtag(self, tag: str, attrs: list) -> None:
        self._event()
        if tag == "br":
            if not self._suppressed:
                self._pending = (self._abs(), " ", False)
        elif tag in BLOCK_TAGS:
            self._flush_block()

    # -- text handlers ---------------------------------------------------------------
    def handle_data(self, data: str) -> None:
        abs_ = self._event()
        if data and not self._suppressed:
            self._pending = (abs_, data, True)

    def handle_entityref(self, name: str) -> None:
        abs_ = self._event()
        if not self._suppressed:
            self._pending = (abs_, html_mod.unescape(f"&{name};"), False)

    def handle_charref(self, name: str) -> None:
        abs_ = self._event()
        if not self._suppressed:
            self._pending = (abs_, html_mod.unescape(f"&#{name};"), False)

    # -- non-content events still delimit fragments ----------------------------------
    def handle_comment(self, data: str) -> None:
        self._event()

    def handle_decl(self, decl: str) -> None:
        self._event()

    def handle_pi(self, data: str) -> None:
        self._event()

    def unknown_decl(self, data: str) -> None:
        self._event()

    # -- lifecycle -------------------------------------------------------------------
    def close(self) -> None:
        super().close()
        if self._pending is not None:  # EOF ends the last fragment
            start, text, literal = self._pending
            self._pending = None
            self._frags.append((start, self._doc_len, text, literal))
        self._flush_block()

    def blocks(self) -> list[Block]:
        return list(self._blocks)


def extract_blocks(doc_text: str) -> list[Block]:
    """Extract readable blocks (with exact raw spans) from decoded filing text."""
    parser = _Extractor(doc_text)
    parser.feed(doc_text)
    parser.close()
    return parser.blocks()


def join_blocks(blocks: list[Block], *, include_tables: bool) -> str:
    """Canonical joining of blocks into one text (shared by chunker and verifier)."""
    return "\n\n".join(
        b.text for b in blocks if include_tables or not b.in_table
    )


def verify_span(
    doc_text: str,
    char_start: int,
    char_end: int,
    expected_text: str,
    *,
    include_tables: bool,
) -> bool:
    """The round-trip check: re-extract the raw span and compare exactly.

    Runs the same extractor on ``doc_text[char_start:char_end]``; block boundaries and
    suppression inside the slice reproduce because the same rules apply. Table blocks
    are filtered the same way the caller filtered them at build time.
    """
    blocks = extract_blocks(doc_text[char_start:char_end])
    return join_blocks(blocks, include_tables=include_tables) == expected_text
