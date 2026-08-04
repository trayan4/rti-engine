# One image, five running commands. Every container app in infra/modules/
# apps.bicep points at this same image and differs only in the process it
# starts — so there is exactly one build to keep correct, not five.

FROM python:3.13-slim AS builder

# Build-only dependencies. None of this reaches the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first, so a source change does not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ src/
COPY alembic.ini ./
COPY migrations/ migrations/
RUN uv sync --frozen --no-dev

# Baked in at build time. Downloading it on every cold start would add
# several hundred megabytes and a minute to a container that is supposed
# to scale from zero.
RUN uv run python -m spacy download en_core_web_lg


FROM python:3.13-slim AS runtime

RUN groupadd --system app && useradd --system --gid app app

WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY --from=builder /app/src src
COPY --from=builder /app/alembic.ini .
COPY --from=builder /app/migrations migrations
COPY data/corpus/ data/corpus/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# The corpus and code are owned by the app user; nothing here writes to
# the image at runtime, so there is nothing else to grant.
RUN chown -R app:app /app
USER app

# Overridden per app in apps.bicep. Present so the image runs something
# sensible if started without one — useful for a local smoke test.
CMD ["uvicorn", "rti_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
