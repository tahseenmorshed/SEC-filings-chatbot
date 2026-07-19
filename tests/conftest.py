"""Shared test fixtures.

Everything here keeps tests deterministic and offline: a controllable fake clock (so the
throttle and backoff can be asserted without real waiting) and a valid ``Settings`` that
doesn't depend on the developer's environment.
"""

from __future__ import annotations

import pytest

from filings_analyst.config import Settings


class FakeClock:
    """A manually-advanced monotonic clock plus a ``sleep`` that advances it.

    Pass ``clock.now`` where a ``monotonic`` callable is expected and ``clock.sleep``
    where a ``sleep`` callable is expected. ``sleeps`` records every sleep duration so
    tests can assert on backoff/throttle timing.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._t += seconds

    def advance(self, seconds: float) -> None:
        self._t += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def settings(tmp_path) -> Settings:
    """A valid Settings pointing the cache at a temp dir, independent of the env."""
    return Settings(
        user_agent="Test Runner test@example.org",
        cache_dir=tmp_path / "cache",
        max_retries=3,
        backoff_base=0.5,
    )
