# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ThaiSmartAddress v7.0 — Dockerfile                                         ║
# ║                                                                              ║
# ║  Multi-stage build:                                                          ║
# ║    Stage 1 (builder) — install Python deps into a venv                      ║
# ║    Stage 2 (runtime) — copy venv + app source only, no build tools          ║
# ║                                                                              ║
# ║  Build:  docker build -t thai-smart-address-api:7.0 .                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Build tools for native extensions (orjson, rapidfuzz)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip --quiet \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="ThaiSmartAddress API" \
      org.opencontainers.image.version="7.0.0" \
      org.opencontainers.image.description="Production Thai address parser API"

WORKDIR /app

# curl needed for Docker health check
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Application source (see .dockerignore for exclusions)
COPY api.py constants.py database.py geo_engine.py models.py parser.py ./

# Non-root user — principle of least privilege
# --create-home is required: PyThaiNLP resolves its corpus cache path via
# get_pythainlp_data_path() which calls os.makedirs(~/.pythainlp) at import
# time. Without a home directory that call raises PermissionError before the
# app even starts. We also set PYTHAINLP_DATA_DIR explicitly so the cache
# location is predictable and not tied to the HOME env var.
RUN useradd --system --uid 1001 --create-home tsa \
    && mkdir -p /app/pythainlp_data \
    && chown -R tsa:tsa /app
ENV PYTHAINLP_DATA_DIR=/app/pythainlp_data
USER tsa

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
