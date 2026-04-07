# syntax=docker/dockerfile:1

# Frontend build stage
# Build frontend assets on the native build platform to avoid
# cross-architecture emulation issues during multi-platform builds.
FROM --platform=$BUILDPLATFORM oven/bun:1 AS frontend-builder

WORKDIR /app

# Copy frontend source code
COPY lightrag_webui/ ./lightrag_webui/

# Build frontend assets for inclusion in the API package
RUN --mount=type=cache,target=/root/.bun/install/cache \
    cd lightrag_webui \
    && bun install --frozen-lockfile \
    && bun run build

# Python build stage - using uv for faster package installation
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install system deps (Rust is required by some wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Ensure shared data directory exists for uv caches
RUN mkdir -p /root/.local/share/uv

# Copy project metadata and sources
COPY pyproject.toml .
COPY setup.py .
COPY uv.lock .

# Install base, API, and offline extras without the project to improve caching
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --extra offline --extra docling --no-install-project --no-editable

# Copy project sources after dependency layer
COPY lightrag/ ./lightrag/

# Include pre-built frontend assets from the previous stage
COPY --from=frontend-builder /app/lightrag/api/webui ./lightrag/api/webui

# Sync ONLY dependencies into the virtual environment. 
# We use --no-install-project to ensure the .venv remains identical 
# even if the application code inside /app/lightrag changes.
RUN --mount=type=cache,target=/root/.local/share/uv \
    uv sync --frozen --no-dev --extra api --extra offline --extra docling --no-install-project --no-editable \
    && /app/.venv/bin/python -m ensurepip --upgrade

# Prepare offline cache directory and pre-populate tiktoken data
# Use uv run to execute commands from the virtual environment
RUN mkdir -p /app/data/tiktoken \
    && uv run lightrag-download-cache --cache-dir /app/data/tiktoken || status=$?; \
    if [ -n "${status:-}" ] && [ "$status" -ne 0 ] && [ "$status" -ne 2 ]; then exit "$status"; fi

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Install uv for package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_SYSTEM_PYTHON=1

# Install system dependencies for OCR (tesseract) and PDF-to-image (poppler)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-vie \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration files
COPY pyproject.toml .
COPY setup.py .
COPY uv.lock .

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/.venv /app/.venv

# Ensure the installed scripts are on PATH
ENV PATH=/app/.venv/bin:/root/.local/bin:$PATH

# Create persistent data directories and copy offline cache
RUN mkdir -p /app/data/rag_storage /app/data/inputs /app/data/tiktoken
COPY --from=builder /app/data/tiktoken /app/data/tiktoken

# Point to the prepared cache and define working directories
ENV TIKTOKEN_CACHE_DIR=/app/data/tiktoken
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs

# Finally, copy the frequently changing application code 
# We do this last to maximize cache hits for all the dependencies setup above
COPY --from=builder /app/lightrag ./lightrag

# Expose API port
EXPOSE 9621

ENTRYPOINT ["python", "-m", "lightrag.api.lightrag_server"]
