# Image for the Grounded Filings Analyst API.
#
# Build:  docker build -t filings-analyst .
# Run:    docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY filings-analyst
#
# The corpus (index, chunks, embedding model) is baked in, so each image is a sealed
# release: this code plus this exact index. Secrets are never baked in; they arrive
# as environment variables at run time.

FROM python:3.12-slim

# uv, copied as a single binary from its official image (pinned to the local version).
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv

# Build-time settings, read by the `uv sync` steps below, so they must come first.
#   UV_COMPILE_BYTECODE  precompile .py to .pyc now; the non-root app user cannot
#                        write them at run time, so it would recompile on every start
#   UV_LINK_MODE=copy    copy packages out of the cache mount (a separate filesystem,
#                        so uv's default hard links cannot reach it)
#   UV_PYTHON_DOWNLOADS  never fetch a different Python; use the image's 3.12
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, without the project itself. This layer is the slow one and is
# reused from cache until pyproject.toml or uv.lock change. The cache mount keeps
# uv's download cache on the build machine, outside the image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# The sealed corpus. .dockerignore narrows data/ to exactly what serving needs.
# --chmod makes it world-readable (some index files are owner-only on the host),
# while ownership stays with root so the app user still cannot modify it.
# It sits above the code because it changes only on a re-index, not on every edit.
COPY --chmod=a+rX data/ data/

# The project code changes most often, so it goes last. Edits here rebuild only
# from this point down.
COPY README.md ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Run as an unprivileged user. Files above stay root-owned, so the app can read the
# corpus but cannot modify it.
RUN useradd --create-home --uid 10001 app
USER app

# Run-time settings, read by the app when a container starts. They sit at the end so
# changing one rebuilds only these final metadata steps, not the layers above.
#   PATH              `python` means the virtualenv's Python, which has the packages
#   PYTHONUNBUFFERED  write print() output immediately, so logs are never held back
#   HOST, PORT        listen on all interfaces, on 8000 (overridable with -e)
#   HF_HUB_OFFLINE    never contact Hugging Face: fail loudly rather than download a
#                     model other than the one the index was built with
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["python", "-m", "filings_analyst.api.serve"]
