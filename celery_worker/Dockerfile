FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-vie \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and source code (needed for editable install)
COPY pyproject.toml setup.py uv.lock ./
COPY lightrag/ ./lightrag/
COPY celery_worker/ ./celery_worker/

# Install dependencies with caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .[api,offline,docling]

# Create data directories
RUN mkdir -p /app/data/rag_storage /app/data/inputs /app/data/logs

ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
ENV LIGHTRAG_LOG_DIR=/app/data/logs

ENTRYPOINT ["celery", "-A", "celery_worker.app", "worker"]
CMD ["-l", "info", "--pool=prefork", "--concurrency=2"]
