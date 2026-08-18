# 3.11, not 3.10: numpy, pandas, scikit-learn and scipy all declare
# requires-python >= 3.11.
FROM python:3.11-slim

# Pinned to the uv version this repo was locked with (uv --version), so
# resolution behaves identically in and out of the container.
COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONPATH=/app

# Dependencies first, from the lockfile, so editing application code doesn't
# bust this layer. --no-dev excludes pytest/httpx, which the image never runs.
# --no-cache: without it, uv's download cache (~540MB for this dependency
# set) gets baked into the image layer alongside the installed venv, nearly
# doubling image size for a cache nothing at runtime ever reads.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-cache

COPY . /app

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
