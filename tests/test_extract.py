"""Offset-preserving extractor: the round-trip invariant under adversarial markup."""

from __future__ import annotations

from filings_analyst.parsing.extract import (
    decode_filing,
    extract_blocks,
    normalize_text,
    verify_span,
)


def _assert_all_blocks_round_trip(doc: str) -> list:
    blocks = extract_blocks(doc)
    for b in blocks:
        assert verify_span(
            doc, b.char_start, b.char_end, b.text, include_tables=True
        ), f"round-trip failed for block {b.order}: {b.text!r}"
    return blocks


def test_entities_shift_lengths_but_offsets_stay_true():
    # &#8220; (7 raw chars) decodes to one character — the classic drift trap.
    doc = '<p>We said &#8220;grow&#8221; &amp; deliver 12%.</p>'
    blocks = _assert_all_blocks_round_trip(doc)
    assert len(blocks) == 1
    assert blocks[0].text == "We said “grow” & deliver 12%."


def test_sentence_split_across_inline_tags_is_one_block():
    doc = (
        "<p><span>We launched </span><ix:nonNumeric>five satellites"
        "</ix:nonNumeric><span> in 2024.</span></p>"
    )
    blocks = _assert_all_blocks_round_trip(doc)
    assert [b.text for b in blocks] == ["We launched five satellites in 2024."]


def test_block_tags_split_blocks():
    doc = "<div><p>First paragraph.</p><p>Second paragraph.</p></div>"
    blocks = _assert_all_blocks_round_trip(doc)
    assert [b.text for b in blocks] == ["First paragraph.", "Second paragraph."]


def test_ix_header_and_hidden_are_suppressed():
    doc = (
        "<ix:header><xbrli:context>machine noise</xbrli:context></ix:header>"
        "<p>Visible text.</p>"
        "<p>Before <ix:hidden>secret</ix:hidden>after.</p>"
        "<script>alert(1)</script><style>.x{}</style>"
    )
    blocks = _assert_all_blocks_round_trip(doc)
    texts = [b.text for b in blocks]
    assert texts == ["Visible text.", "Before after."]
    assert not any("noise" in t or "secret" in t or "alert" in t for t in texts)


def test_display_none_is_suppressed():
    doc = '<p>Shown.</p><div style="color:red;DISPLAY: NONE">hidden stuff</div><p>Also shown.</p>'
    blocks = _assert_all_blocks_round_trip(doc)
    assert [b.text for b in blocks] == ["Shown.", "Also shown."]


def test_table_blocks_are_marked():
    doc = "<p>Prose.</p><table><tr><td>Cell A</td><td>Cell B</td></tr></table><p>More prose.</p>"
    blocks = _assert_all_blocks_round_trip(doc)
    flags = [(b.text, b.in_table) for b in blocks]
    assert flags == [
        ("Prose.", False),
        ("Cell A", True),
        ("Cell B", True),
        ("More prose.", False),
    ]


def test_br_is_whitespace_not_a_block_break():
    doc = "<p>Line one<br/>line two<br>line three</p>"
    blocks = _assert_all_blocks_round_trip(doc)
    assert [b.text for b in blocks] == ["Line one line two line three"]


def test_offsets_point_at_the_real_source_span():
    doc = "junk<p>  The actual sentence.  </p>junk"
    blocks = _assert_all_blocks_round_trip(doc)
    # Text outside the <p> forms its own blocks; the paragraph's span is trimmed to
    # the visible sentence itself.
    block = next(b for b in blocks if "actual" in b.text)
    assert doc[block.char_start : block.char_end] == "The actual sentence."


def test_nbsp_and_zero_width_normalization():
    doc = "<p>a&#160;b&#8203;c   d</p>"  # nbsp, zero-width space, run of spaces
    (block,) = _assert_all_blocks_round_trip(doc)
    assert block.text == "a b​c d".replace("​", "")  # "a bc d"


def test_decode_filing_utf8_and_latin1_fallback():
    assert decode_filing("héllo".encode("utf-8")) == ("héllo", "utf-8")
    text, enc = decode_filing(b"caf\xe9")  # invalid UTF-8, valid latin-1
    assert enc == "latin-1" and text == "café"


def test_normalize_text_is_idempotent():
    s = "  a  b\n\nc\t d ​"
    once = normalize_text(s)
    assert normalize_text(once) == once
