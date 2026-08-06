"""Thin HTTP client for the target API. Plain requests — no target-code import."""

from __future__ import annotations

import requests

DEFAULT_TIMEOUT = 60.0


class HarnessClient:
    def __init__(self, base_url: str, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def health(self) -> requests.Response:
        return self._session.get(f"{self.base_url}/health", timeout=self.timeout)

    def ask(self, question: str, *, k: int | None = None) -> requests.Response:
        body: dict = {"question": question}
        if k is not None:
            body["k"] = k
        return self._session.post(f"{self.base_url}/ask", json=body, timeout=self.timeout)

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        ticker: str | None = None,
        form: str | None = None,
        section: str | None = None,
        method: str | None = None,
    ) -> requests.Response:
        body: dict = {"query": query}
        for key, val in (
            ("k", k), ("ticker", ticker), ("form", form),
            ("section", section), ("method", method),
        ):
            if val is not None:
                body[key] = val
        return self._session.post(f"{self.base_url}/search", json=body, timeout=self.timeout)

    def get_chunk(self, chunk_id: str) -> requests.Response:
        return self._session.get(f"{self.base_url}/chunks/{chunk_id}", timeout=self.timeout)
