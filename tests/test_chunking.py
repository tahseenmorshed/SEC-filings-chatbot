"""Chunker: packing bounds, section boundaries, coverage, and provenance envelopes."""

from __future__ import annotations

from filings_analyst.parsing.chunking import pack_chunks
from filings_analyst.parsing.extract import Block


def _blocks(texts: list[str], *, in_table: set[int] = frozenset()) -> list[Block]:
    out = []
    pos = 0
    for i, t in enumerate(texts):
        out.append(Block(text=t, char_start=pos, char_end=pos + len(t), in_table=i in in_table, order=i))
        pos += len(t) + 5
    return out


def _uniform_sections(blocks, key="item_1", title="Business"):
    return {b.order: (key, title) for b in blocks}


def test_greedy_packing_respects_target():
    blocks = _blocks(["a" * 400 for _ in range(10)])
    chunks = pack_chunks(blocks, _uniform_sections(blocks), target_chars=1000, max_chars=2000)
    assert all(len(c.text) <= 1000 for c in chunks)
    # 400+2+400 = 802 fits; adding a third (1204) exceeds 1000 → 2 blocks per chunk.
    assert [c.n_blocks for c in chunks] == [2, 2, 2, 2, 2]


def test_chunks_never_cross_section_boundaries():
    blocks = _blocks(["short one", "short two", "short three", "short four"])
    section_of = {
        0: ("item_1", "Business"),
        1: ("item_1", "Business"),
        2: ("item_1a", "Risk Factors"),
        3: ("item_1a", "Risk Factors"),
    }
    chunks = pack_chunks(blocks, section_of, target_chars=10_000, max_chars=20_000)
    assert [(c.section_key, c.n_blocks) for c in chunks] == [
        ("item_1", 2),
        ("item_1a", 2),
    ]


def test_every_prose_block_in_exactly_one_chunk():
    blocks = _blocks([f"block {i} " + "x" * (i * 37 % 300) for i in range(50)],
                     in_table={7, 8, 21})
    chunks = pack_chunks(blocks, _uniform_sections(blocks), target_chars=500, max_chars=1000)
    packed = sum(c.n_blocks for c in chunks)
    prose = [b for b in blocks if not b.in_table]
    assert packed == len(prose)  # full coverage, no duplicates
    # table blocks excluded
    assert not any("block 7 " in c.text or "block 21 " in c.text for c in chunks)


def test_oversized_single_block_is_flagged_not_split():
    blocks = _blocks(["small text", "z" * 5000, "more small text"])
    chunks = pack_chunks(blocks, _uniform_sections(blocks), target_chars=3000, max_chars=4500)
    over = [c for c in chunks if c.oversized]
    assert len(over) == 1
    assert over[0].n_blocks == 1 and len(over[0].text) == 5000


def test_chunk_envelope_spans_first_to_last_block():
    blocks = _blocks(["alpha", "beta", "gamma"])
    chunks = pack_chunks(blocks, _uniform_sections(blocks), target_chars=10_000, max_chars=20_000)
    (c,) = chunks
    assert c.char_start == blocks[0].char_start
    assert c.char_end == blocks[-1].char_end
    assert c.text == "alpha\n\nbeta\n\ngamma"
