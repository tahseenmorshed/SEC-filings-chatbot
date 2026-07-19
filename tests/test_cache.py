"""DiskCache: proves the byte-exact round-trip the provenance guarantee depends on."""

from __future__ import annotations

from filings_analyst.edgar.cache import DiskCache, cache_key


def test_round_trip_is_byte_exact_on_adversarial_content(tmp_path):
    cache = DiskCache(tmp_path)
    # Content designed to break naive text handling: a UTF-8 BOM, multibyte characters,
    # mixed CRLF/LF newlines, trailing whitespace, an embedded NUL, and no final newline.
    body = (
        b"\xef\xbb\xbf"  # UTF-8 BOM
        + "Revenüe grew — 12%\r\n".encode("utf-8")
        + b"line two   \n"
        + b"tab\tsep\x00end"
    )
    key = cache_key("https://data.sec.gov/thing.json")

    cache.set(key, body, {"url": "https://data.sec.gov/thing.json", "status_code": 200})
    got = cache.get(key)

    assert got is not None
    assert got.body == body  # exact bytes, unchanged
    assert len(got.body) == len(body)
    assert got.status_code == 200


def test_miss_returns_none(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get(cache_key("https://data.sec.gov/never-fetched")) is None


def test_metadata_survives_round_trip(tmp_path):
    cache = DiskCache(tmp_path)
    key = cache_key("https://data.sec.gov/x")
    meta = {
        "url": "https://data.sec.gov/x",
        "status_code": 200,
        "headers": {"Content-Type": "application/json"},
    }
    cache.set(key, b"{}", meta)

    got = cache.get(key)
    assert got is not None
    assert got.url == "https://data.sec.gov/x"
    assert got.headers["Content-Type"] == "application/json"


def test_distinct_urls_get_distinct_keys():
    assert cache_key("https://a") != cache_key("https://b")
    assert cache_key("https://a", "GET") != cache_key("https://a", "POST")
