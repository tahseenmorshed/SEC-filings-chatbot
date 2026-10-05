"""Run the API server: ``uv run python -m filings_analyst.api.serve [port]``.

Bind address and port come from the ``HOST`` and ``PORT`` environment variables when
set, which is how containers and cloud platforms configure them. The defaults keep a
plain local run bound to loopback, reachable only from this machine.
"""

from __future__ import annotations

import os
import sys

import uvicorn

from .app import create_app 


def main(argv: list[str]) -> int:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(argv[0]) if argv else int(os.environ.get("PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
