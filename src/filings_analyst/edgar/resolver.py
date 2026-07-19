"""Ticker → CIK resolution.

The SEC's financial endpoints key on the **CIK** (Central Index Key), not the ticker, in
a 10-digit zero-padded form (e.g. NVDA → ``0001045810``). The authoritative ticker
directory is ``company_tickers.json``; we fetch it *through* :class:`SECClient` so it is
User-Agent-enforced, throttled, and disk-cached like every other request. CIKs are never
hardcoded — always resolved from this file.

Run as a script to resolve the configured target tickers::

    uv run python -m filings_analyst.edgar.resolver
"""

from __future__ import annotations

from ..config import COMPANY_TICKERS_URL, Settings
from .client import SECClient


class UnknownTicker(KeyError):
    """Raised when a ticker is absent from the SEC ticker directory."""


def pad_cik(cik: int | str) -> str:
    """Return ``cik`` as the 10-digit zero-padded string the SEC endpoints require."""
    return f"{int(cik):010d}"


class CikResolver:
    """Resolves tickers to zero-padded CIKs using the SEC ticker directory."""

    def __init__(self, client: SECClient) -> None:
        self._client = client
        # ticker (uppercased) -> padded CIK; lazily loaded on first resolve().
        self._map: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._map is None:
            raw = self._client.get_json(COMPANY_TICKERS_URL)
            # Shape: { "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "..."}, ... }
            self._map = {
                str(row["ticker"]).upper(): pad_cik(row["cik_str"])
                for row in raw.values()
            }
        return self._map

    def resolve(self, ticker: str) -> str:
        """Return the 10-digit zero-padded CIK for ``ticker`` (case-insensitive)."""
        mapping = self._load()
        key = ticker.strip().upper()
        try:
            return mapping[key]
        except KeyError:
            raise UnknownTicker(
                f"Ticker {ticker!r} not found in the SEC ticker directory"
            ) from None

    def resolve_all(self, tickers: list[str]) -> dict[str, str]:
        """Resolve a list of tickers to ``{ticker: padded_cik}`` (preserves input casing)."""
        return {t: self.resolve(t) for t in tickers}


def _main() -> None:
    settings = Settings()  # raises with guidance if SEC_USER_AGENT is unset/invalid
    with SECClient(settings) as client:
        resolver = CikResolver(client)
        results = resolver.resolve_all(settings.target_tickers)

    width = max(len(t) for t in results)
    print("Resolved ticker → CIK (10-digit zero-padded):\n")
    for ticker, cik in results.items():
        print(f"  {ticker.rjust(width)}  →  {cik}")


if __name__ == "__main__":
    _main()
