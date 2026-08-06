"""Case loading: plain YAML data files, one case per file, dispatched by `type`.

Cases are data, not code, so a reference runner in any language could execute them.
Four types:

* ``correctness``  — /ask a question; must NOT refuse; every `expected_facts` regex
  must match somewhere in the answer text or claim statements/quotes.
* ``refusal``      — /ask a question; must refuse. `category` (off_corpus /
  near_corpus_unanswerable / adjacent_company_confusion) is metadata for reporting.
* ``filter``       — /search with a scoping filter; every hit's record must match.
* ``robustness``   — a group of paraphrases of the same underlying question; all must
  agree on refused-state and (if answered) all satisfy the same `expected_facts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CorrectnessCase:
    id: str
    question: str
    expected_facts: list[str]
    k: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class RefusalCase:
    id: str
    question: str
    category: str
    expected_reason: str | None = None
    k: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class FilterCase:
    id: str
    query: str
    expected_ticker: str | None = None
    expected_form: str | None = None
    expected_section: str | None = None
    k: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class RobustnessCase:
    id: str
    questions: list[str]
    expected_facts: list[str] = field(default_factory=list)
    expected_refused: bool = False
    notes: str = ""


Case = CorrectnessCase | RefusalCase | FilterCase | RobustnessCase

_TYPE_MAP = {
    "correctness": CorrectnessCase,
    "refusal": RefusalCase,
    "filter": FilterCase,
    "robustness": RobustnessCase,
}


def load_case(path: Path) -> Case:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    kind = data.pop("type", None)
    cls = _TYPE_MAP.get(kind)
    if cls is None:
        raise ValueError(f"{path}: unknown or missing case type {kind!r}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(f"{path}: bad fields for type {kind!r}: {exc}") from exc


def load_cases(cases_dir: Path) -> list[Case]:
    paths = sorted(cases_dir.glob("*.yaml")) + sorted(cases_dir.glob("*.yml"))
    if not paths:
        raise FileNotFoundError(f"No case files found under {cases_dir}")
    cases = [load_case(p) for p in paths]
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate case ids: {sorted(dupes)}")
    return cases
