"""Application configuration.

All tunables live here as a single ``Settings`` object loaded from the environment
(and an optional ``.env`` file). The most important value is ``user_agent``: the SEC
rejects requests without a descriptive ``"Name email"`` User-Agent, so we validate it
at construction time and fail loudly rather than discovering the problem as silent 403s
mid-crawl.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- SEC endpoints -----------------------------------------------------------------
# Structured facts and filing metadata live on data.sec.gov; the ticker directory lives
# on www.sec.gov. Kept as module constants (not env-configurable) since they are part of
# the SEC's public API surface, not deployment config.
DATA_HOST = "https://data.sec.gov"
WWW_HOST = "https://www.sec.gov"

COMPANY_TICKERS_URL = f"{WWW_HOST}/files/company_tickers.json"


def submissions_url(cik: str) -> str:
    """URL for a company's filing history (``cik`` must be 10-digit zero-padded)."""
    return f"{DATA_HOST}/submissions/CIK{cik}.json"


def company_facts_url(cik: str) -> str:
    """URL for a company's structured XBRL facts (``cik`` must be 10-digit zero-padded)."""
    return f"{DATA_HOST}/api/xbrl/companyfacts/CIK{cik}.json"


def archive_document_url(cik: int | str, accession_number: str, document: str) -> str:
    """URL for a document inside a filing's folder under the Archives path.

    Unlike the JSON APIs, Archives paths use the *unpadded* CIK and the accession
    number with its dashes removed.
    """
    return (
        f"{WWW_HOST}/Archives/edgar/data/{int(cik)}"
        f"/{accession_number.replace('-', '')}/{document}"
    )


# Default filesystem locations, importable without constructing Settings (offline
# stages like parsing must not require SEC_USER_AGENT, which guards network access).
DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_STORE_DIR = Path("data/store")

# The LLM used by the grounded answering layer. Pinned like every other model in this
# project; the ANTHROPIC_API_KEY env var (see .env.example) authenticates the SDK.
DEFAULT_ANSWER_MODEL = "claude-opus-4-8"

# The 8 companies this project tracks. Resolved to CIKs programmatically — never hardcode
# CIKs; see filings_analyst.edgar.resolver.
DEFAULT_TARGET_TICKERS = [
    "RKLB",
    "ASTS",
    "NVDA",
    "LUNR",
    "NBIS",
    "GEV",
    "JOBY",
    "PWR",
]


class Settings(BaseSettings):
    """Environment-driven settings. Prefix env vars with ``SEC_`` (e.g. ``SEC_USER_AGENT``)."""

    model_config = SettingsConfigDict(
        env_prefix="SEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required in effect: defaults to empty so the validator (not a generic "field
    # required" error) produces the actionable message when SEC_USER_AGENT is unset.
    user_agent: str = ""

    target_tickers: list[str] = Field(default_factory=lambda: list(DEFAULT_TARGET_TICKERS))

    # Throttle target. SEC's hard ceiling is 10 req/s per IP; we aim for 8 as a margin.
    max_requests_per_second: float = 8.0

    # Where raw responses are cached. Relative paths resolve against the process CWD.
    cache_dir: Path = DEFAULT_CACHE_DIR

    # Where fetched filings land as a browsable, ticker-keyed corpus (distinct from the
    # URL-hash cache above, which is transport-level dedupe, not an organized store).
    store_dir: Path = DEFAULT_STORE_DIR

    # HTTP behavior.
    request_timeout: float = 30.0
    max_retries: int = 4
    backoff_base: float = 0.5  # seconds; delay = backoff_base * 2**attempt

    @field_validator("user_agent")
    @classmethod
    def _validate_user_agent(cls, v: str) -> str:
        v = v.strip()
        placeholder = (
            'SEC_USER_AGENT is not set to a real value. The SEC requires a descriptive '
            'User-Agent in the form "Name email@example.com" on every request; requests '
            "without one are rejected (403). Set it, e.g.:\n"
            '    export SEC_USER_AGENT="Jane Doe jane@example.com"'
        )
        if not v:
            raise ValueError(placeholder)
        # Must look like "Name email": contain whitespace and an email-ish token.
        if " " not in v or "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(placeholder)
        # Reject the shipped example placeholder.
        if "example.com" in v.lower() or v.lower().startswith("your name"):
            raise ValueError(placeholder)
        return v

    @property
    def min_request_interval(self) -> float:
        """Minimum seconds between requests implied by ``max_requests_per_second``."""
        return 1.0 / self.max_requests_per_second
