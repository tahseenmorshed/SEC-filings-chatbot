"""Deterministic question scoping: which company is the question about?

A cheap, no-LLM pass that matches known tickers and company names against the
question. Exactly one company matched → apply a ticker filter to retrieval (sharper
ranks, no cross-company bleed). Zero or several matched → no filter, let hybrid
retrieval's enrichment carry company identity. Deterministic and fully testable.
"""

from __future__ import annotations

import re

# Corporate suffix tokens that don't identify a company ("Inc", "Corp", ...).
_SUFFIX_TOKENS = {"inc", "corp", "corporation", "ltd", "llc", "co", "usa", "plc", "sa"}


def build_alias_map(entity_names: dict[str, str]) -> dict[str, str]:
    """Map lowercase alias → ticker, from the index's ticker → entity-name table."""
    aliases: dict[str, str] = {}
    for ticker, name in entity_names.items():
        aliases[ticker.lower()] = ticker
        # "Rocket Lab USA, Inc." → tokens ["rocket", "lab"] → "rocket lab"
        tokens = [
            t for t in re.findall(r"[a-z0-9]+", name.lower())
            if t not in _SUFFIX_TOKENS
        ]
        if tokens:
            aliases[" ".join(tokens)] = ticker
            # Also the full name minus punctuation (keeps "ast spacemobile" and
            # "rocket lab usa" both matchable).
            full = " ".join(re.findall(r"[a-z0-9]+", name.lower()))
            aliases.setdefault(full, ticker)
    return aliases


def scope_ticker(question: str, entity_names: dict[str, str]) -> str | None:
    """Return the single ticker the question names, or None."""
    aliases = build_alias_map(entity_names)
    q = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
    matched = {
        ticker
        for alias, ticker in aliases.items()
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", q)
    }
    return matched.pop() if len(matched) == 1 else None
