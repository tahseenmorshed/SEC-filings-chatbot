"""Run the API server: ``uv run python -m filings_analyst.api.serve [port]``."""

from __future__ import annotations

import sys

import uvicorn

from .app import create_app


def main(argv: list[str]) -> int:
    port = int(argv[0]) if argv else 8000
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
