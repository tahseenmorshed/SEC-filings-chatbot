"""Parsing track: raw filing HTML → offset-anchored blocks → sections → chunks.

Everything in this package upholds one invariant (the round-trip guarantee): for any
emitted block or chunk, re-extracting ``doc_text[char_start:char_end]`` with the same
deterministic function reproduces its ``text`` exactly. Offsets always point into the
decoded original file as fetched — never into a derived/cleaned copy.
"""
