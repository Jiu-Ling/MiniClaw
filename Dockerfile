FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --no-dev

COPY . .

RUN mkdir -p .miniclaw

RUN chmod +x /app/scripts/docker-entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"

VOLUME ["/app/.miniclaw"]

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]

CMD ["miniclaw", "--help"]