"""Section detection: TOC decoys, cross-references, and 10-Q part disambiguation."""

from __future__ import annotations

from filings_analyst.parsing.extract import Block
from filings_analyst.parsing.sections import assign_sections, detect_headings


def _blocks(specs: list[tuple[str, bool]]) -> list[Block]:
    """Build a block list from (text, in_table) specs with synthetic offsets."""
    out = []
    pos = 0
    for i, (text, in_table) in enumerate(specs):
        out.append(
            Block(text=text, char_start=pos, char_end=pos + len(text), in_table=in_table, order=i)
        )
        pos += len(text) + 10
    return out


def test_toc_rows_in_tables_are_never_headings():
    blocks = _blocks([
        ("Item 1.", True),  # TOC
        ("Item 1A.", True),  # TOC
        ("Item 1. Business", False),  # real
        ("We make rockets.", False),
        ("Item 1A. Risk Factors", False),  # real
        ("Rockets sometimes explode.", False),
    ])
    headings = detect_headings(blocks, "10-K")
    assert [(h.key, h.block_order) for h in headings] == [
        ("item_1", 2),
        ("item_1a", 4),
    ]


def test_early_false_heading_loses_to_longer_true_chain():
    # A false "Item 1A" line appearing BEFORE the real Item 1 (as in the real ASTS
    # 10-K summary box). Accepting it would block Item 1 — the LIS must prefer the
    # longer, all-true chain.
    blocks = _blocks([
        ("Item 1A. Risk Factors.", False),  # false positive, early
        ("Item 1. Business", False),  # real
        ("Prose about the business.", False),
        ("Item 1A. Risk Factors", False),  # real
        ("Prose about risks.", False),
        ("Item 2. Properties", False),  # real
    ])
    headings = detect_headings(blocks, "10-K")
    assert [(h.key, h.block_order) for h in headings] == [
        ("item_1", 1),
        ("item_1a", 3),
        ("item_2", 5),
    ]


def test_cross_references_mid_prose_are_ignored():
    blocks = _blocks([
        ("Item 1. Business", False),
        ("See the discussion in Item 1A of this Annual Report for details.", False),
        ("Item 1A. Risk Factors", False),
    ])
    headings = detect_headings(blocks, "10-K")
    assert [h.block_order for h in headings] == [0, 2]


def test_long_paragraph_starting_with_item_is_not_a_heading():
    long_tail = "Risk Factors and many other words " * 10
    blocks = _blocks([
        ("Item 1. Business", False),
        (f"Item 1A. {long_tail}", False),  # prose, too long to be a heading
        ("Item 1A. Risk Factors", False),
    ])
    headings = detect_headings(blocks, "10-K")
    assert [h.block_order for h in headings] == [0, 2]


def test_10q_duplicate_item_numbers_resolved_by_parts():
    blocks = _blocks([
        ("PART I — FINANCIAL INFORMATION", False),
        ("Item 1. Financial Statements", False),
        ("Balance sheet prose.", False),
        ("Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations", False),
        ("MD&A prose.", False),
        ("PART II — OTHER INFORMATION", False),
        ("Item 1. Legal Proceedings", False),
        ("Litigation prose.", False),
        ("Item 1A. Risk Factors", False),
    ])
    headings = detect_headings(blocks, "10-Q")
    keys = [h.key for h in headings]
    assert keys == [
        "part1", "part1_item1", "part1_item2",
        "part2", "part2_item1", "part2_item1a",
    ]


def test_assign_sections_labels_every_block():
    blocks = _blocks([
        ("Cover page text.", False),
        ("Item 1. Business", False),
        ("Business prose.", False),
        ("A table cell", True),
        ("Item 1A. Risk Factors", False),
        ("Risk prose.", False),
    ])
    headings = detect_headings(blocks, "10-K")
    mapping = assign_sections(blocks, headings)
    assert mapping[0] == ("preamble", "Preamble")
    assert mapping[2][0] == "item_1"
    assert mapping[3][0] == "item_1"  # table blocks inherit the current section
    assert mapping[5][0] == "item_1a"
