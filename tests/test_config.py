"""Settings: the User-Agent must be valid or construction fails loudly."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from filings_analyst.config import Settings, company_facts_url, submissions_url


def _settings(ua: str) -> Settings:
    # Pass cache_dir explicitly so tests never touch a real data/ dir.
    return Settings(user_agent=ua, cache_dir="/tmp/does-not-matter")


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "NoEmailHere",  # no email, no space
        "jane@example.org",  # email but no name/space
        "Your Name your.email@example.com",  # the shipped placeholder
        "Jane Doe jane@localhost",  # no dot in domain
    ],
)
def test_invalid_user_agent_is_rejected(bad):
    with pytest.raises(ValidationError):
        _settings(bad)


def test_valid_user_agent_is_accepted():
    s = _settings("Jane Doe jane@realdomain.io")
    assert s.user_agent == "Jane Doe jane@realdomain.io"
    assert s.min_request_interval == pytest.approx(1.0 / 8.0)


def test_url_builders_use_padded_cik():
    assert submissions_url("0001045810").endswith("/submissions/CIK0001045810.json")
    assert company_facts_url("0001045810").endswith(
        "/api/xbrl/companyfacts/CIK0001045810.json"
    )
