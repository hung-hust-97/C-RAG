# Repository Guidelines

LightRAG is an advanced Retrieval-Augmented Generation (RAG) framework designed to enhance information retrieval and generation through graph-based knowledge representation.

## Project Structure & Module Organization
- `lightrag/`: Core Python package with orchestrators (`lightrag/lightrag.py`), storage adapters in `kg/`, LLM bindings in `llm/`, and helpers such as `operate.py` and `utils_*.py`.
- `lightrag-api/`: FastAPI service (`lightrag_server.py`) with routers under `routers/` and Gunicorn launcher `run_with_gunicorn.py`.
- `lightrag_webui/`: React 19 + TypeScript client driven by Bun + Vite; UI components live in `src/`.
- `scripts/setup/`: Interactive environment setup wizard. `setup.sh` orchestrates staged `--base` / `--storage` / `--server` / validation flows, `lib/` holds prompt/validation/file helpers, and `templates/*.yml` contains compose fragments for bundled services.
- Tests live in `tests/` and root-level `test_*.py`. Working datasets stay in `inputs/`, `rag_storage/`, `temp/`; deployment collateral lives in `docs/`, `k8s-deploy/`, and `docker-compose.yml`.
- `Makefile`: Canonical entry point for the setup wizard and local developer shortcuts; prefer documented targets over invoking ad hoc shell snippets.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: set up the Python runtime.
- `pip install -e .` / `pip install -e .[api]`: install the package and API extras in editable mode.
- `make env-base`: first-run interactive setup for LLM, embedding, and reranker configuration; writes `.env` and may generate `docker-compose.final.yml`.
- `make env-storage`, `make env-server`: optional follow-up wizard stages for storage backends and server/security/SSL settings; both reuse the existing `.env`.
- `make env-validate`, `make env-security-check`, `make env-backup`: validate, audit, or back up the current `.env` via the setup wizard.
- `lightrag-server` or `uvicorn lightrag.api.lightrag_server:app --reload`: start the API locally; ensure `.env` is present.
- `python -m pytest tests` (offline markers apply by default) or `python -m pytest tests --run-integration` / `python test_graph_storage.py`: run the full suite, opt into integration coverage, or target an individual script.
- `ruff check .`: lint Python sources before committing.
- `bun install`, `bun run dev`, `bun run build`, `bun test`: manage the web UI workflow (Bun is mandatory).

## Coding Style & Naming Conventions
- Backend code follow PEP 8 with four-space indentation, annotate functions, and reach for dataclasses when modelling state.
- Use `lightrag.utils.logger` instead of `print`; respect logger configuration flags.
- Extend storage or pipeline abstractions via `lightrag.base` and keep reusable helpers in the existing `utils_*.py`.
- Python modules remain lowercase with underscores; React components use `PascalCase.tsx` and hooks-first patterns.
- Front-end code should remain in TypeScript with two-space indentation, rely on functional React components with hooks, and follow Tailwind utility style.

## Testing Guidelines
- Keep pytest additions close to the code you touch (`tests/` mirrors feature folders and there are root-level `test_*.py` helpers); functions must start with `test_`.
- Follow `tests/pytest.ini`: markers include `offline`, `integration`, `requires_db`, and `requires_api`, and the suite runs with `-m "not integration"` by default—pass `--run-integration` (or set `LIGHTRAG_RUN_INTEGRATION=true`) when external services are available.
- Use the custom CLI toggles from `tests/conftest.py`: `--keep-artifacts`/`LIGHTRAG_KEEP_ARTIFACTS=true`, `--stress-test`/`LIGHTRAG_STRESS_TEST=true`, and `--test-workers N`/`LIGHTRAG_TEST_WORKERS` to dial up workloads or preserve temp files during investigations.
- Export other required `LIGHTRAG_*` environment variables before running integration or storage tests so adapters can reach configured backends.
- For UI updates, pair changes with Vitest specs and run `bun test`.

## Commit & Pull Request Guidelines
- Use concise, imperative commit subjects (e.g., `Fix lock key normalization`) and add body context only when necessary.
- PRs should include a summary, operational impact, linked issues, and screenshots or API samples for user-facing work.
- Verify `ruff check .`, `python -m pytest`, and affected Bun commands succeed before requesting review; note the runs in the PR text.
- This repo is a fork of `HKUDS/LightRAG`. Always target **`HKUDS/LightRAG:main`** (upstream) when creating PRs, not the fork's own main.

## Security & Configuration Tips
- Copy `.env.example` and `config.ini.example`; never commit secrets or real connection strings.
- Configure storage backends through `LIGHTRAG_*` variables and validate them with `docker-compose` services when needed.
- Treat `lightrag.log*` as local artefacts; purge sensitive information before sharing logs or outputs.

## PostgreSQL Connection Pool Architecture & Troubleshooting

### Architecture Overview
LightRAG uses a **shared connection pool** architecture for PostgreSQL connections:
- **One pool per process** (NOT per workspace): `ClientManager._shared_instance` is a class-level singleton
- All workspaces within a process share the same connection pool
- Connections are reused across requests (not closed after each query)
- Pool size determines maximum concurrent connections, not total requests

### Key Configuration Variables
Located in `.env`:
- `POSTGRES_MAX_CONNECTIONS`: Client-side pool size (default: 400)
  - Controls max concurrent connections from application to PostgreSQL
  - Shared across all workspaces in the same process
- `POSTGRES_MAX_INACTIVE_CONNECTION_LIFETIME`: Idle timeout in seconds (default: 300)
  - Automatically closes idle connections after this duration
  - Prevents connection leaks and reduces resource usage
- PostgreSQL server `max_connections`: Server-side limit (default: 1000 in docker-compose.yml)
  - Must be higher than total client connections across all processes
  - Formula: `num_processes × POSTGRES_MAX_CONNECTIONS ≤ max_connections`

### Connection Pool Behavior
**Sequential Requests:**
- 10 sequential requests → 1 connection reused 10 times
- Connection stays open in pool between requests

**Concurrent Requests:**
- 10 concurrent requests with `max_size=3` → 3 connections active, 7 requests wait in queue
- Requests are served as connections become available

**Idle Timeout:**
- Connections idle for `max_inactive_connection_lifetime` seconds are automatically closed
- Prevents accumulation of unused connections over time

### Common Issues & Solutions

#### Issue 1: Connection Pool Exhausted
**Symptoms:**
- `TooManyConnectionsError` or `asyncpg.exceptions.TooManyConnectionsError`
- Requests timing out or hanging
- Error: "sorry, too many clients already"

**Diagnosis:**
```bash
# Check current connections
docker exec -it lightrag-postgres psql -U postgres -d lightrag -c "
SELECT count(*), state, wait_event_type 
FROM pg_stat_activity 
WHERE datname='lightrag' 
GROUP BY state, wait_event_type;"

# Monitor in real-time
.kiro/scripts/monitor_pg_connections.sh
```

**Solutions:**
1. **Increase pool size** (if server has capacity):
   ```bash
   # In .env
   POSTGRES_MAX_CONNECTIONS=400  # Increase from default
   ```

2. **Increase server limit** (in docker-compose.yml):
   ```yaml
   postgres:
     command: postgres -c max_connections=1000 -c shared_buffers=512MB
   ```

3. **Reduce idle timeout** (close unused connections faster):
   ```bash
   # In .env
   POSTGRES_MAX_INACTIVE_CONNECTION_LIFETIME=180  # 3 minutes instead of 5
   ```

4. **Check for connection leaks**:
   - Look for long-running idle connections
   - Verify all async contexts properly close connections
   - Check for stuck transactions

#### Issue 2: Deprecated Table Schema Errors
**Symptoms:**
- `column "id" does not exist` in `LIGHTRAG_TRACK_STATUS` table
- Upload failures with PostgreSQL errors

**Solution:**
The `LIGHTRAG_TRACK_STATUS` table has been removed. Ensure it's not in the `TABLES` dictionary in `lightrag/kg/postgres_impl.py`:
```python
# WRONG - causes errors
TABLES = {
    "LIGHTRAG_TRACK_STATUS": {...},  # Remove this
    ...
}

# CORRECT
TABLES = {
    "LIGHTRAG_DOC_STATUS": {...},
    "LIGHTRAG_DOC_CHUNKS": {...},
    ...
}
```

#### Issue 3: Connection Leaks Over Time
**Symptoms:**
- Idle connections increasing over time
- Connections not being released
- Old connections (age > 30 minutes)

**Diagnosis:**
```bash
# Check connection age
docker exec -it lightrag-postgres psql -U postgres -d lightrag -c "
SELECT pid, state, 
       now() - state_change as idle_duration,
       query
FROM pg_stat_activity 
WHERE datname='lightrag' AND state='idle'
ORDER BY state_change;"
```

**Solutions:**
1. Verify `max_inactive_connection_lifetime` is set in pool config
2. Check for missing `await` statements in async code
3. Ensure RAG instance cache doesn't create multiple pools per event loop
4. Restart services to clear leaked connections

#### Issue 4: MinIO/S3 Connection Issues
**Symptoms:**
- Upload succeeds but extraction fails
- Celery worker can't access uploaded files
- Connection refused to MinIO

**Solutions:**
1. **Use internal Docker network endpoint**:
   ```bash
   # In .env - WRONG
   S3_ENDPOINT_URL=http://localhost:19000
   
   # CORRECT
   S3_ENDPOINT_URL=http://minio:9000
   ```

2. **Ensure Celery worker has S3 environment variables** (in docker-compose.yml):
   ```yaml
   celery_worker:
     environment:
       - S3_ENDPOINT_URL=${S3_ENDPOINT_URL}
       - S3_ACCESS_KEY_ID=${S3_ACCESS_KEY_ID}
       - S3_SECRET_ACCESS_KEY=${S3_SECRET_ACCESS_KEY}
       - S3_BUCKET_NAME=${S3_BUCKET_NAME}
   ```

### Monitoring & Validation Tools
Located in `.kiro/scripts/`:
- `monitor_pg_connections.sh`: Real-time connection monitoring
- `check_pool_contention.py`: Test pool behavior under load
- `check_connection_config.sh`: Validate configuration (checks `num_processes × pool_size ≤ max_connections`)

### Best Practices
1. **Capacity Planning**: Set `max_connections` to at least 2× expected concurrent load for headroom
2. **Idle Timeout**: Use 3-5 minutes for most workloads; shorter for high-churn scenarios
3. **Pool Size**: Start with 100-200 per process; scale based on actual concurrent query patterns
4. **Monitoring**: Regularly check connection usage and idle duration
5. **Shared Pool**: Remember that all workspaces share the pool—don't multiply by workspace count
6. **Process Count**: In multi-process deployments (Gunicorn), total connections = `num_workers × pool_size`

### Configuration Example
Recommended settings for production with 1000 concurrent users:
```bash
# .env
POSTGRES_MAX_CONNECTIONS=400
POSTGRES_MAX_INACTIVE_CONNECTION_LIFETIME=300

# docker-compose.yml
postgres:
  command: postgres -c max_connections=1000 -c shared_buffers=512MB -c max_prepared_transactions=100
```

This configuration supports:
- 1 API process × 400 pool = 400 client connections
- 1000 server limit provides 2.5× headroom
- 5-minute idle timeout prevents connection accumulation

## Document Identification & Migration (track_id → doc_id + content_hash)
- **CRITICAL**: The `track_id` system has been completely removed and replaced with UUID-based `doc_id` + `content_hash` for duplicate detection.
- **Document ID Format**: All documents now use `doc-{uuid4}` format (e.g., `doc-550e8400-e29b-41d4-a716-446655440000`) instead of hash-based `doc-{md5_hash}`.
- **Immediate Availability**: `doc_id` is generated and returned immediately on upload (before extraction), eliminating the need for batch tracking.
- **Duplicate Detection**: Use `content_hash` (MD5 of content) for detecting duplicate documents, not `doc_id` comparison.
- **Batch Operations**: Batch upload/scan endpoints return `doc_ids` array instead of single `track_id` for monitoring multiple documents.
- **Deprecated Endpoints**: `/track_status/{track_id}` endpoint has been removed; use `/documents/{doc_id}` for individual document queries.
- **Database Schema**: Documents table includes `content_hash` column (VARCHAR(32), nullable, indexed by workspace) for duplicate detection.
- **Error Handling**: When encountering `track_id` references in legacy code or errors:
  - Replace with `doc_id` for document identification
  - Use `content_hash` for duplicate detection logic
  - Update batch tracking to use `doc_ids` arrays
  - Query individual documents via `/documents/{doc_id}` endpoint
  - Use `/documents/paginated` for querying multiple documents with filters
- **Migration Path**: Existing hash-based `doc_id` values are supported during transition but all new documents use UUID format.
- **API Responses**: Never include `track_id` in API responses; always use `doc_id` (single) or `doc_ids` (batch) fields.
- **Status Counting**: Use `count_by="document"` for status counts; `count_by="track"` option has been removed and returns 400 error.

## Document Processing Status System
LightRAG uses a simplified 6-state status model to track document processing through the pipeline:

**Status Flow:**
```
UPLOADING → EXTRACTING → EXTRACTED → CHUNKING → PROCESSED
               ↓            ↓           ↓           ↓
            FAILED       FAILED      FAILED      FAILED
```

**Status Definitions:**
1. **UPLOADING**: File đang được upload (reserved for future use)
2. **EXTRACTING**: Đang extract full text (OCR/Docling) - handled by extraction worker
3. **EXTRACTED**: Đã extract xong, chờ chunking + KG - ready for chunking worker
4. **CHUNKING**: Đang chunking + extract entities - handled by chunking/KG worker
5. **PROCESSED**: Hoàn thành tất cả - final state
6. **FAILED**: Lỗi ở bất kỳ stage nào - can be reprocessed

**Legacy Statuses (Deprecated):**
- `PENDING` → mapped to `EXTRACTED`
- `PROCESSING` → mapped to `CHUNKING`
- `PREPROCESSED` → deprecated
- `CHUNKED` → deprecated

**Worker Responsibilities:**
- **Extraction Worker** (`celery_worker/tasks.py`): Picks up documents in `EXTRACTING` status, performs OCR/Docling extraction, updates to `EXTRACTED` or `FAILED`
- **Chunking/KG Worker** (`lightrag.py:apipeline_process_enqueue_documents`): Picks up documents in `EXTRACTED`, `CHUNKING`, `FAILED`, `PENDING`, `PROCESSING` statuses, performs chunking and entity extraction, updates to `PROCESSED` or `FAILED`

**Chunking Configuration:**
- Default chunking method is `semantic_chunking_markdown`.
  - Splits content logically by paragraphs and headers.
  - Strictly avoids splitting HTML or Markdown tables across chunks.
  - If a section is split, its parent header is prepended to subsequent chunks to preserve context.
- Token boundaries follow `CHUNK_SIZE` and `CHUNK_OVERLAP_SIZE`.

**Reprocess Behavior:**
- `/documents/reprocess_failed` endpoint picks up: `FAILED`, `EXTRACTING`, `EXTRACTED`, `PENDING`, `PROCESSING`
- Stuck `EXTRACTING` documents with content are automatically reset to `EXTRACTED`
- Stuck `EXTRACTING` documents without content trigger re-extraction if file exists
- Documents are validated for consistency before reprocessing

**Extraction Strategy & Fallback Logic:**
- **PDF & Images (.pdf, .jpg, .png, etc.)**:
  - **Priority**: Direct DeepSeek OCR (Full document). Skip Docling and Hybrid mode for optimal speed on CPU.
  - **Failure Handling**: If DeepSeek OCR fails or returns no content, the status is set to `FAILED` immediately. No fallback to local Tesseract/pypdf to avoid low-quality results and heavy CPU usage.
- **Office Formats (.docx, .pptx, .xlsx)**:
  - **Primary**: Docling (Local structural extraction).
  - **Fallback**: If Docling fails or returns no content, fallback to DeepSeek OCR.
  - **Failure Handling**: If both Docling and OCR fail, the status is set to `FAILED`.
- **General Rules**:
  - Extraction runs on **CPU** by default (GPU not required for current worker deployment).
  - All extracted content is converted to **Markdown** before being passed to the chunking pipeline.
  - Any total failure in extraction MUST result in a `FAILED` status with an appropriate error message in the database.

## Automation & Agent Workflow
- Use repo-relative `workdir` arguments for every shell command and prefer `rg`/`rg --files` for searches since they are faster under the CLI harness.
- Default edits to ASCII, rely on `apply_patch` for single-file changes, and only add concise comments that aid comprehension of complex logic.
- Honor existing local modifications; never revert or discard user changes (especially via `git reset --hard`) unless explicitly asked.
- Follow the planning tool guidance: skip it for trivial fixes, but provide multi-step plans for non-trivial work and keep the plan updated as steps progress.
- Validate changes by running the relevant `ruff`/`pytest`/`bun test` commands whenever feasible, and describe any unrun checks with follow-up guidance.
- For Codex and other fresh-shell automation, prefer `./scripts/test.sh` instead of bare `pytest`; the script falls back through `PYTHON`, the active virtualenv, `uv`, `.venv`, and `venv` before trying `python` or `python3`.
- For setup workflow changes, prefer `make env-*` targets over calling `scripts/setup/setup.sh` directly; the `Makefile` resolves a Bash 4+ interpreter for macOS/Linux compatibility.
- When editing setup logic, keep `.env` host-usable and treat `docker-compose.final.yml` as generated output assembled from `scripts/setup/templates/*.yml`; compose-only overrides belong in the wizard-managed compose layer rather than being persisted back into `.env`.
- Place all agent-generated utility scripts and test files in the `.kiro/` directory: use `.kiro/scripts/` for utility scripts, and organize test files appropriately within `.kiro/`.
- DO NOT create summary reports, analysis documents, or documentation files in `.kiro/reports/` or anywhere else. Always provide analysis, summaries, and findings directly in the chat session instead of creating markdown files.
