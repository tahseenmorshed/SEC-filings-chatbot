"""On-disk response cache.

Every SEC response is written here so we never re-fetch the same URL. Beyond saving
requests, this cache is the project's **provenance anchor**: it stores the response body
as **raw bytes, verbatim** — no decoding, no newline or whitespace normalization, no
re-encoding. Downstream, filing narrative will be chunked with exact character offsets;
those offsets are only meaningful if the bytes they index into are byte-for-byte the
same as what the SEC served. Hence the round-trip guarantee: ``set(k, b); get(k).body``
returns ``b`` unchanged.

Layout, keyed by ``sha256(method + " " + url)``::

    <cache_dir>/<first-2-hex>/<key>.body        # raw response bytes
    <cache_dir>/<first-2-hex>/<key>.meta.json   # url, status, headers, fetched_at, ...

Sharding by the first two hex characters keeps any single directory small.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + ``os.replace``).

    A crash mid-write can never leave a truncated file that later reads as valid.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        # Clean up the temp file on any failure before re-raising.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class CachedResponse:
    """A response retrieved from (or about to be stored in) the cache."""

    body: bytes
    meta: dict[str, Any]

    @property
    def status_code(self) -> int:
        return int(self.meta.get("status_code", 0))

    @property
    def url(self) -> str:
        return str(self.meta.get("url", ""))

    @property
    def headers(self) -> dict[str, str]:
        return dict(self.meta.get("headers", {}))


def cache_key(url: str, method: str = "GET") -> str:
    """Deterministic key for a (method, url) pair."""
    digest = hashlib.sha256(f"{method.upper()} {url}".encode("utf-8"))
    return digest.hexdigest()


class DiskCache:
    """Filesystem-backed byte cache with sidecar JSON metadata."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _paths(self, key: str) -> tuple[Path, Path]:
        shard = self.root / key[:2]
        return shard / f"{key}.body", shard / f"{key}.meta.json"

    def get(self, key: str) -> CachedResponse | None:
        """Return the cached response for ``key``, or ``None`` on a miss."""
        body_path, meta_path = self._paths(key)
        if not body_path.exists() or not meta_path.exists():
            return None
        body = body_path.read_bytes()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return CachedResponse(body=body, meta=meta)

    def set(self, key: str, body: bytes, meta: dict[str, Any]) -> None:
        """Store ``body`` (verbatim) and ``meta`` under ``key``."""
        body_path, meta_path = self._paths(key)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(body_path, body)
        atomic_write_bytes(
            meta_path,
            json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
        )
