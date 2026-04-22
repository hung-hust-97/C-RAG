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
- **IMPORTANT**: asyncpg only marks connections for closure; actual cleanup happens on next `acquire()`
- Without active traffic or explicit pool closure, idle connections persist indefinitely
- Celery workers now implement `worker_process_shutdown` signal to explicitly close pools on exit

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

4. **Reduce Celery worker pool size** (prevent worker connection exhaustion):
   ```bash
   # In .env or docker-compose.yml
   CELERY_POSTGRES_MAX_CONNECTIONS=50  # Reduce from default 400
   ```

5. **Check for connection leaks**:
   - Look for long-running idle connections
   - Verify all async contexts properly close connections
   - Check for stuck transactions
   - Ensure Celery workers have `worker_process_shutdown` cleanup handlers

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
- `diagnose_pg_connections.sh`: Comprehensive connection diagnostic with recommendations
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

**Reprocess Behavior:**
- `/documents/reprocess_failed` endpoint picks up: `FAILED`, `EXTRACTING`, `EXTRACTED`, `PENDING`, `PROCESSING`
- Stuck `EXTRACTING` documents with content are automatically reset to `EXTRACTED`
- Stuck `EXTRACTING` documents without content trigger re-extraction if file exists
- Documents are validated for consistency before reprocessing

## Celery Worker Architecture & Troubleshooting

### Worker Overview
LightRAG uses Celery workers for asynchronous document processing. The worker implementation is in `celery_worker/tasks.py` and handles:
- Document extraction (OCR/Docling)
- Chunking and entity extraction
- Graph construction
- Async query execution

### Worker Tasks
Located in `celery_worker/tasks.py`:

1. **task_extract_and_enqueue**: OCR & Markdown extraction for a single file
   - Picks up files from disk
   - Runs DeepSeek/Docling to produce Markdown text
   - Updates document status to `EXTRACTED`
   - Automatically triggers `task_chunk_and_graph`

2. **task_chunk_and_graph**: Chunking and graph building
   - Processes ALL documents in `EXTRACTED`, `PENDING`, `PROCESSING`, `FAILED` statuses
   - Performs chunking → entity extraction → graph upsert
   - Mirrors logic of `lightrag.py:apipeline_process_enqueue_documents()`

3. **task_process_all_extracted**: Process all workspaces with EXTRACTED documents
   - Queries database for all workspaces with EXTRACTED documents
   - Triggers `task_chunk_and_graph` for each workspace automatically
   - Use this to process stuck documents across all workspaces

4. **task_execute_query**: Offline async query execution
   - Executes RAG queries outside request-response cycle
   - Stores results in Redis for later retrieval

5. **apipeline_process_enqueue_documents_task**: Multi-workspace pipeline processing
   - Processes/enqueues documents for specific workspace
   - Supports `reprocess_failed` parameter for reprocessing stuck documents

### Worker Status Handling
**CRITICAL**: Workers must use `DocStatus` enum from `lightrag.base`, NOT hardcoded strings.

```python
# CORRECT - Use enum
from lightrag.base import DocStatus

status = DocStatus.EXTRACTED
if doc_status == DocStatus.PROCESSED:
    ...

# WRONG - Hardcoded strings
status = "extracted"  # Will cause validation errors
if doc_status == "processed":  # Inconsistent with enum
    ...
```

**Status Values**: All status values are UPPERCASE:
- `DocStatus.UPLOADING` = "UPLOADING"
- `DocStatus.EXTRACTING` = "EXTRACTING"
- `DocStatus.EXTRACTED` = "EXTRACTED"
- `DocStatus.CHUNKING` = "CHUNKING"
- `DocStatus.PROCESSED` = "PROCESSED"
- `DocStatus.FAILED` = "FAILED"
- `DocStatus.DUPLICATED` = "DUPLICATED"

### Worker Initialization & Cleanup
The worker implements critical lifecycle hooks:

**Process Initialization** (`worker_process_init` signal):
- Resets asyncio locks after fork to prevent event loop binding issues
- Called when a new worker process is forked
- Clears lock caches so new locks are created in child's event loop

**Process Shutdown** (`worker_process_shutdown` signal):
- Closes all PostgreSQL connection pools to prevent connection leaks
- Critical for preventing idle connection accumulation
- Closes pools for: KV storage, doc status storage, graph storage, vector storage
- Clears RAG instance cache

### Common Worker Issues

#### Issue 1: Documents Stuck in EXTRACTED Status
**Symptoms:**
- Documents remain in `EXTRACTED` status indefinitely
- Worker is running but not processing documents
- Worker logs show "No documents to process"

**Diagnosis:**
```bash
# Check worker status
docker logs --tail 50 c-rag_celery_worker_1

# Check document statuses
docker exec c-rag_postgres_1 psql -U postgres -d crag -c \
  "SELECT status, COUNT(*) FROM lightrag_doc_status GROUP BY status;"

# Check if EXTRACTED documents have content
docker exec c-rag_postgres_1 psql -U postgres -d crag -c "
SELECT COUNT(*) as total,
       SUM(CASE WHEN f.id IS NOT NULL THEN 1 ELSE 0 END) as with_content,
       SUM(CASE WHEN f.id IS NULL THEN 1 ELSE 0 END) as without_content
FROM lightrag_doc_status d
LEFT JOIN lightrag_doc_full f ON d.id = f.id AND d.workspace = f.workspace
WHERE d.status = 'EXTRACTED';"
```

**Root Cause:**
- Documents marked as `EXTRACTED` but have no content in `lightrag_doc_full` table
- This can happen during status migration or if extraction failed silently
- Worker skips documents without content (line 1881 in lightrag.py: `if content_data:`)

**Solutions:**
1. **Fix EXTRACTED documents without content** - mark them as FAILED:
   ```bash
   docker exec c-rag_postgres_1 psql -U postgres -d crag -c "
   UPDATE lightrag_doc_status 
   SET status = 'FAILED',
       error_msg = 'No content found - extraction may have failed',
       updated_at = NOW()
   WHERE id IN (
       SELECT d.id
       FROM lightrag_doc_status d
       LEFT JOIN lightrag_doc_full f ON d.id = f.id AND d.workspace = f.workspace
       WHERE d.status = 'EXTRACTED' AND f.id IS NULL
   );"
   ```

2. **Manual trigger for all workspaces**: Use the new task to process all workspaces:
   ```bash
   # Trigger processing for ALL workspaces with EXTRACTED documents
   docker exec c-rag_celery_worker_1 python -c "
   from celery_worker.tasks import task_process_all_extracted
   result = task_process_all_extracted.delay()
   print(f'Task queued: {result.id}')
   "
   ```

3. **Manual trigger for specific workspace**:
   ```bash
   # Via Python
   docker exec c-rag_celery_worker_1 python -c "
   from celery_worker.tasks import task_chunk_and_graph
   task_chunk_and_graph.delay('workspace_id')
   "
   ```

4. **Check worker connectivity**:
   - Verify Redis connection: `docker logs c-rag_redis_1`
   - Verify worker can reach PostgreSQL: Check `POSTGRES_HOST` in worker environment
   - Inside Docker, use service names (e.g., `postgres`, `redis`), not `localhost`

5. **Restart worker** to clear any stuck state:
   ```bash
   docker-compose restart celery_worker
   ```

#### Issue 2: Pydantic Validation Errors
**Symptoms:**
- Error: `Input should be 'uploading', 'extracting', 'extracted'... [type=enum, input_value='EXTRACTED']`
- API returns 500 errors when querying documents

**Root Cause:**
- Database has uppercase status values (e.g., `EXTRACTED`)
- Code expects lowercase values (e.g., `extracted`)
- Mismatch between `DocStatus` enum definition and database values

**Solution:**
1. **Update database statuses** to uppercase:
   ```bash
   python .kiro/scripts/update_document_statuses.py
   ```

2. **Rebuild Docker images** to get latest code:
   ```bash
   docker-compose down
   docker-compose build --no-cache lightrag celery_worker
   docker-compose up -d
   ```

3. **Verify enum consistency** in `lightrag/base.py`:
   ```python
   class DocStatus(str, Enum):
       UPLOADING = "UPLOADING"      # Must be uppercase
       EXTRACTING = "EXTRACTING"
       EXTRACTED = "EXTRACTED"
       # ... etc
   ```

#### Issue 3: Worker Can't Connect to PostgreSQL
**Symptoms:**
- Worker logs show: `OSError("Multiple exceptions: [Errno 111] Connect call failed")`
- Connection refused to localhost:5432

**Root Cause:**
- Worker is trying to connect to `localhost` instead of Docker service name
- `.env` file has `POSTGRES_HOST=localhost` which works for host but not Docker

**Solution:**
1. **Update docker-compose.yml** to force correct hostname:
   ```yaml
   celery_worker:
     environment:
       POSTGRES_HOST: postgres  # Force Docker service name
   ```

2. **Or use separate .env for Docker**:
   ```bash
   # In .env for host development
   POSTGRES_HOST=localhost
   
   # In docker-compose.yml override
   environment:
     POSTGRES_HOST: ${POSTGRES_HOST:-postgres}
   ```

#### Issue 4: Event Loop Binding Errors
**Symptoms:**
- Error: `RuntimeError: Task <Task> attached to a different loop`
- Worker crashes after processing some documents

**Root Cause:**
- Asyncio locks inherited from parent process after fork
- Locks bound to parent's event loop can't be used in child

**Solution:**
- Already implemented in `celery_worker/tasks.py`
- `worker_process_init` signal resets all locks after fork
- If adding new async locks, ensure they're reset in `_reset_locks_after_fork()`

### Worker Monitoring Commands

```bash
# Check worker status
docker-compose ps | grep celery

# View worker logs
docker logs --tail 100 -f c-rag_celery_worker_1

# Check active Celery tasks
docker exec c-rag_celery_worker_1 celery -A celery_worker.app inspect active

# Check registered tasks
docker exec c-rag_celery_worker_1 celery -A celery_worker.app inspect registered

# Check worker stats
docker exec c-rag_celery_worker_1 celery -A celery_worker.app inspect stats

# Restart worker
docker-compose restart celery_worker

# View Redis queue length
docker exec c-rag_redis_1 redis-cli LLEN celery
```

### Worker Configuration
Key environment variables in `docker-compose.yml`:

```yaml
celery_worker:
  environment:
    # Storage connections - must use Docker service names
    POSTGRES_HOST: postgres
    NEO4J_URI: bolt://neo4j:7687
    MILVUS_URI: http://milvus:19530
    
    # Redis for task queue
    CELERY_BROKER_URL: redis://redis:6379/0
    CELERY_RESULT_BACKEND: redis://redis:6379/1
    
    # S3/MinIO for file access
    S3_ENDPOINT_URL: http://minio:9000
    
    # Worker pool configuration
    CELERY_POSTGRES_MAX_CONNECTIONS: 50  # Lower than API to prevent exhaustion
```

### Best Practices
1. **Status Consistency**: Always use `DocStatus` enum, never hardcoded strings
2. **Connection Management**: Ensure `worker_process_shutdown` closes all pools
3. **Docker Networking**: Use service names (`postgres`, `redis`) not `localhost`
4. **Manual Triggering**: Workers don't auto-pickup; trigger via API or Celery task
5. **Monitoring**: Regularly check worker logs and task queue length
6. **Graceful Shutdown**: Use `docker-compose stop` (not `kill`) to allow cleanup
7. **Reprocess Failed Documents**: Use `/documents/reprocess_failed` endpoint (no `/api` prefix)
   - Without `workspace_id`: processes ALL workspaces with FAILED documents
   - With `workspace_id`: processes only specified workspace
   - Example: `curl -X POST "http://localhost:9621/documents/reprocess_failed"`

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
