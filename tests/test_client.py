"""SECClient: User-Agent enforcement, caching, retry/backoff, and error handling.

All requests are served by an ``httpx.MockTransport`` — no real network. The rate limiter
is given a no-op sleep so the only sleeps recorded are the client's own backoff waits,
which the retry tests assert on directly.
"""

from __future__ import annotations

import json

import httpx
import pytest

from filings_analyst.edgar.client import SECClient, SECRequestError
from filings_analyst.edgar.throttle import RateLimiter


def _no_wait_limiter() -> RateLimiter:
    # Fixed clock + no-op sleep: acquire() never actually delays and never touches the
    # client's sleep recorder.
    return RateLimiter(8.0, monotonic=lambda: 0.0, sleep=lambda _s: None)


def _make_client(settings, handler, *, sleeps=None):
    transport = httpx.MockTransport(handler)
    return SECClient(
        settings,
        limiter=_no_wait_limiter(),
        transport=transport,
        sleep=(sleeps.append if sleeps is not None else (lambda _s: None)),
    )


def test_user_agent_header_is_sent(settings):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["user_agent"] = request.headers.get("User-Agent")
        return httpx.Response(200, content=b"ok")

    with _make_client(settings, handler) as client:
        client.get("https://data.sec.gov/thing")

    assert seen["user_agent"] == settings.user_agent


def test_second_identical_request_is_served_from_cache(settings):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=b"payload")

    with _make_client(settings, handler) as client:
        first = client.get("https://data.sec.gov/thing")
        second = client.get("https://data.sec.gov/thing")

    assert calls["n"] == 1  # transport hit exactly once
    assert first.body == second.body == b"payload"


def test_get_json_parses_body(settings):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"cik_str": 320193}).encode())

    with _make_client(settings, handler) as client:
        data = client.get_json("https://data.sec.gov/thing.json")

    assert data["cik_str"] == 320193


def test_retries_on_503_then_succeeds_with_backoff(settings):
    sleeps: list[float] = []
    responses = iter([httpx.Response(503), httpx.Response(200, content=b"done")])

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    with _make_client(settings, handler, sleeps=sleeps) as client:
        result = client.get("https://data.sec.gov/flaky")

    assert result.body == b"done"
    # One retry → one backoff wait of backoff_base * 2**0 = 0.5s.
    assert sleeps == [settings.backoff_base]


def test_retry_after_header_overrides_backoff(settings):
    sleeps: list[float] = []
    responses = iter(
        [httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(200, content=b"ok")]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)

    with _make_client(settings, handler, sleeps=sleeps) as client:
        client.get("https://data.sec.gov/limited")

    assert sleeps == [2.0]


def test_404_raises_without_retrying(settings):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404)

    with _make_client(settings, handler) as client:
        with pytest.raises(SECRequestError) as exc:
            client.get("https://data.sec.gov/missing")

    assert exc.value.status_code == 404
    assert calls["n"] == 1  # no retry on a client error


def test_persistent_5xx_exhausts_retries_and_raises(settings):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    with _make_client(settings, handler) as client:
        with pytest.raises(SECRequestError) as exc:
            client.get("https://data.sec.gov/broken")

    assert exc.value.status_code == 500
    assert calls["n"] == settings.max_retries + 1  # initial try + max_retries


def test_connection_error_is_retried(settings):
    sleeps: list[float] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, content=b"recovered")

    with _make_client(settings, handler, sleeps=sleeps) as client:
        result = client.get("https://data.sec.gov/network")

    assert result.body == b"recovered"
    assert calls["n"] == 2
    assert sleeps == [settings.backoff_base]
