"""Section identification: label blocks with the 10-K/10-Q item they belong to.

The hard part is that "Item 1A" appears many times in a filing: the table of contents,
the real heading, and prose cross-references ("as discussed in Item 1A..."). Three
defenses, layered:

1. **Table exclusion** — TOC rows live inside ``<table>`` in every pilot filing, and
   table blocks are never heading candidates.
2. **Block-start anchoring + shortness** — a candidate must *begin* the block and have
   little trailing text; mid-sentence cross-references never match.
3. **Weighted longest-increasing-subsequence (LIS)** — real headings appear in the
   form's canonical item order. Among all candidates we select the maximum-weight chain
   that is strictly increasing in that order. A stray early false positive (e.g. an
   "Item 1A. Risk Factors" line in a summary box) would *block* later true headings
   from forming a chain, so the longer all-true chain wins automatically. Title
   similarity to the canonical item name adds weight.

10-Qs repeat item numbers across parts (Part I Item 1 = Financial Statements; Part II
Item 1 = Legal Proceedings), so PART markers participate in the same chain and item
candidates may claim a slot in either part — the LIS resolves the ambiguity globally.

Design stance: section labels are **metadata, not provenance**. A block that ends up
labeled "preamble"/"unknown" is still perfectly citable; offsets never depend on any
of the heuristics in this module.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .extract import Block

# Canonical item sequences. Titles are used for weighting and display only.
FORM_10K_ITEMS: list[tuple[str, str]] = [
    ("1", "Business"),
    ("1A", "Risk Factors"),
    ("1B", "Unresolved Staff Comments"),
    ("1C", "Cybersecurity"),
    ("2", "Properties"),
    ("3", "Legal Proceedings"),
    ("4", "Mine Safety Disclosures"),
    ("5", "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities"),
    ("6", "Reserved"),
    ("7", "Management's Discussion and Analysis of Financial Condition and Results of Operations"),
    ("7A", "Quantitative and Qualitative Disclosures About Market Risk"),
    ("8", "Financial Statements and Supplementary Data"),
    ("9", "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure"),
    ("9A", "Controls and Procedures"),
    ("9B", "Other Information"),
    ("9C", "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections"),
    ("10", "Directors, Executive Officers and Corporate Governance"),
    ("11", "Executive Compensation"),
    ("12", "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters"),
    ("13", "Certain Relationships and Related Transactions, and Director Independence"),
    ("14", "Principal Accountant Fees and Services"),
    ("15", "Exhibits and Financial Statement Schedules"),
    ("16", "Form 10-K Summary"),
]

FORM_10Q_PART1_ITEMS: list[tuple[str, str]] = [
    ("1", "Financial Statements"),
    ("2", "Management's Discussion and Analysis of Financial Condition and Results of Operations"),
    ("3", "Quantitative and Qualitative Disclosures About Market Risk"),
    ("4", "Controls and Procedures"),
]

FORM_10Q_PART2_ITEMS: list[tuple[str, str]] = [
    ("1", "Legal Proceedings"),
    ("1A", "Risk Factors"),
    ("2", "Unregistered Sales of Equity Securities and Use of Proceeds"),
    ("3", "Defaults Upon Senior Securities"),
    ("4", "Mine Safety Disclosures"),
    ("5", "Other Information"),
    ("6", "Exhibits"),
]

_ITEM_RE = re.compile(r"^item\s+(\d{1,2})\s*([a-c])?\s*[.:]?\s*(.*)$", re.IGNORECASE)
_PART_RE = re.compile(r"^part\s+(i{1,3}|iv)\b[\s.:—–-]*(.{0,80})$", re.IGNORECASE)

# A heading block is short: the item designation plus at most a title.
_MAX_HEADING_TRAILING = 120
_TITLE_SIMILARITY_BONUS_THRESHOLD = 0.55


@dataclass(frozen=True)
class SectionHeading:
    """A detected section boundary."""

    block_order: int  # Block.order of the heading block
    char_start: int  # where the heading starts in the document
    key: str  # "item_1a", "part2_item1", "part1", ...
    title: str
    is_part_marker: bool


@dataclass(frozen=True)
class _Slot:
    """One position in the canonical sequence a candidate may occupy."""

    seq_index: int
    key: str
    title: str
    is_part_marker: bool


def _build_slots(form: str) -> dict[str, list[_Slot]]:
    """Map a match token (item code or PART numeral) to its candidate slot(s)."""
    slots: dict[str, list[_Slot]] = {}
    seq = 0

    def add(token: str, key: str, title: str, marker: bool) -> None:
        nonlocal seq
        slots.setdefault(token, []).append(_Slot(seq, key, title, marker))
        seq += 1

    if form == "10-K":
        for code, title in FORM_10K_ITEMS:
            add(f"ITEM{code}", f"item_{code.lower()}", title, False)
    elif form == "10-Q":
        add("PARTI", "part1", "Part I — Financial Information", True)
        for code, title in FORM_10Q_PART1_ITEMS:
            add(f"ITEM{code}", f"part1_item{code.lower()}", title, False)
        add("PARTII", "part2", "Part II — Other Information", True)
        for code, title in FORM_10Q_PART2_ITEMS:
            add(f"ITEM{code}", f"part2_item{code.lower()}", title, False)
    else:
        raise ValueError(f"Unsupported form for section labeling: {form!r}")
    return slots


def _candidates(blocks: list[Block], slots: dict[str, list[_Slot]]):
    """Yield (block, slot, weight) for every plausible heading occurrence."""
    for b in blocks:
        if b.in_table:
            continue
        m = _ITEM_RE.match(b.text)
        if m:
            code = m.group(1) + (m.group(2) or "").upper()
            trailing = m.group(3)
            if len(trailing) > _MAX_HEADING_TRAILING:
                continue
            for slot in slots.get(f"ITEM{code.upper()}", []):
                weight = 1.0
                if trailing:
                    ratio = difflib.SequenceMatcher(
                        None, trailing.casefold(), slot.title.casefold()
                    ).ratio()
                    if ratio >= _TITLE_SIMILARITY_BONUS_THRESHOLD:
                        weight += 0.5
                yield b, slot, weight
            continue
        m = _PART_RE.match(b.text)
        if m:
            numeral = m.group(1).upper()
            for slot in slots.get(f"PART{numeral}", []):
                yield b, slot, 1.5  # part markers are unambiguous anchors

def _max_weight_increasing_chain(cands: list[tuple[Block, _Slot, float]]):
    """Classic O(n²) weighted-LIS over (block order, sequence index)."""
    n = len(cands)
    best_w = [0.0] * n
    prev = [-1] * n
    for i in range(n):
        b_i, s_i, w_i = cands[i]
        best_w[i] = w_i
        for j in range(i):
            b_j, s_j, w_j = cands[j]
            if b_j.order < b_i.order and s_j.seq_index < s_i.seq_index:
                if best_w[j] + w_i > best_w[i]:
                    best_w[i] = best_w[j] + w_i
                    prev[i] = j
    if not cands:
        return []
    end = max(range(n), key=lambda i: best_w[i])
    chain = []
    while end != -1:
        chain.append(cands[end])
        end = prev[end]
    return list(reversed(chain))


def detect_headings(blocks: list[Block], form: str) -> list[SectionHeading]:
    """Detect section headings for a 10-K or 10-Q."""
    slots = _build_slots(form)
    cands = list(_candidates(blocks, slots))
    cands.sort(key=lambda c: (c[0].order, c[1].seq_index))
    chain = _max_weight_increasing_chain(cands)
    return [
        SectionHeading(
            block_order=b.order,
            char_start=b.char_start,
            key=slot.key,
            title=slot.title,
            is_part_marker=slot.is_part_marker,
        )
        for b, slot, _ in chain
    ]


def assign_sections(
    blocks: list[Block], headings: list[SectionHeading]
) -> dict[int, tuple[str, str]]:
    """Map every Block.order to its (section_key, section_title).

    Blocks before the first heading get ("preamble", "Preamble"). A part marker closes
    the previous item's section; blocks between it and the next item heading carry the
    part's own key (e.g. "part2").
    """
    by_order = {h.block_order: h for h in headings}
    current = ("preamble", "Preamble")
    mapping: dict[int, tuple[str, str]] = {}
    for b in blocks:
        h = by_order.get(b.order)
        if h is not None:
            current = (h.key, h.title)
        mapping[b.order] = current
    return mapping
