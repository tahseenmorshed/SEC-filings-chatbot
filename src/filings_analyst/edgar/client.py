"""The single HTTP client all SEC traffic flows through.

Centralizing SEC access in one place means the etiquette rules can't be bypassed
anywhere else in the codebase:

* **User-Agent** — set on every request from ``Settings.user_agent``.
* **Throttle** — a :class:`RateLimiter` spaces live requests within the SEC's budget.
* **Cache** — responses are read from / written to :class:`DiskCache`; a cache hit costs
  no network call and no throttle slot.
* **Retry** — transient failures (connection errors, 429, 5xx) retry with exponential
  backoff, honoring ``Retry-After`` when the SEC sends it.

Everything is injectable (cache, transport, limiter, sleep) so the whole client can be
exercised in tests with an ``httpx.MockTransport`` and fake clocks — no real network.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable

import httpx

from ..config import Settings
from .cache import CachedResponse, DiskCache, cache_key
from .throttle import RateLimiter

# Statuses worth retrying: rate-limited, or a transient server-side error.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class SECRequestError(RuntimeError):
    """Raised for a non-retryable HTTP error or once retries are exhausted."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SECClient:
    """Throttled, cached, User-Agent-enforcing HTTP client for SEC endpoints."""

    def __init__(
        self,
        settings: Settings,
        *,
        cache: DiskCache | None = None,
        limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.cache = cache if cache is not None else DiskCache(settings.cache_dir)
        self.limiter = (
            limiter
            if limiter is not None
            else RateLimiter(settings.max_requests_per_second)
        )
        self._sleep = sleep
        self._client = httpx.Client(
            headers={
                "User-Agent": settings.user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=settings.request_timeout,
            transport=transport,
            follow_redirects=True,
        )

    # -- context management -------------------------------------------------------
    def __enter__(self) -> "SECClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- public API ---------------------------------------------------------------
    def get(self, url: str, *, use_cache: bool = True) -> CachedResponse:
        """GET ``url``, returning raw bytes + metadata (from cache when available)."""
        key = cache_key(url, "GET")
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        response = self._get_with_retry(url)
        cached = CachedResponse(body=response.content, meta=self._build_meta(response))
        self.cache.set(key, cached.body, cached.meta)
        return cached

    def get_bytes(self, url: str, *, use_cache: bool = True) -> bytes:
        """Return the raw response body for ``url``."""
        return self.get(url, use_cache=use_cache).body

    def get_json(self, url: str, *, use_cache: bool = True):
        """Return the response body for ``url`` parsed as JSON.

        Decodes a *copy* of the bytes for parsing; the cache retains the verbatim
        original so provenance is never affected.
        """
        import json

        return json.loads(self.get(url, use_cache=use_cache).body.decode("utf-8"))

    # -- internals ----------------------------------------------------------------
    def _get_with_retry(self, url: str) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            # Only live requests consume a throttle slot (cache hits already returned).
            self.limiter.acquire()
            try:
                response = self._client.get(url)
            except httpx.TransportError as exc:
                # Connection/read errors are transient; back off and retry.
                last_exc = exc
                if attempt >= self.settings.max_retries:
                    break
                self._sleep(self._backoff_delay(attempt))
                continue

            if response.status_code in _RETRYABLE_STATUS:
                if attempt >= self.settings.max_retries:
                    raise SECRequestError(
                        f"GET {url} failed after {attempt + 1} attempts: "
                        f"HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                self._sleep(self._retry_after(response) or self._backoff_delay(attempt))
                continue

            if response.status_code >= 400:
                # Non-retryable client error (e.g. 404) — fail immediately.
                raise SECRequestError(
                    f"GET {url} returned HTTP {response.status_code}",
                    status_code=response.status_code,
                )

            return response

        raise SECRequestError(
            f"GET {url} failed after {self.settings.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    def _backoff_delay(self, attempt: int) -> float:
        return self.settings.backoff_base * (2**attempt)

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        """Parse a ``Retry-After`` header (seconds form) if present and valid."""
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _build_meta(response: httpx.Response) -> dict:
        return {
            "url": str(response.request.url),
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content_type": response.headers.get("Content-Type", ""),
            "method": response.request.method,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
