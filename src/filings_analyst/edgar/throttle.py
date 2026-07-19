"""Client-side rate limiting for SEC access.

The SEC caps traffic at 10 requests/second per IP and silently blocks IPs that exceed
it. We use a **minimum-interval spacing limiter** rather than a token bucket: every
acquisition is forced to be at least ``1/rate`` seconds after the previous one. Unlike a
token bucket (which permits bursts up to the bucket size), spacing guarantees the
instantaneous rate never exceeds the target even at the very start — the safest reading
of the SEC's limit.

The clock and sleep functions are injectable so the limiter can be tested
deterministically against a fake clock, with no real waiting.
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class RateLimiter:
    """Spaces acquisitions at least ``1/rate`` seconds apart. Thread-safe."""

    def __init__(
        self,
        rate: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._min_interval = 1.0 / rate
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        # Timestamp (monotonic clock) at which the next request is permitted.
        self._next_allowed: float | None = None

    def acquire(self) -> None:
        """Block until it is permissible to issue the next request."""
        with self._lock:
            now = self._monotonic()
            if self._next_allowed is None or now >= self._next_allowed:
                # We're on schedule (or first call): permit immediately, and schedule the
                # earliest time the following request may go.
                self._next_allowed = now + self._min_interval
                return
            # We're ahead of schedule: wait out the remaining gap.
            wait = self._next_allowed - now
            self._next_allowed += self._min_interval
        # Sleep outside the lock so concurrent callers can queue up fairly.
        self._sleep(wait)
