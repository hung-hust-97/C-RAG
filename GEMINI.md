# C-RAG (LightRAG Enhanced) - Codebase Knowledge Base

## 1. Project Overview
**C-RAG** is an enterprise-grade Retrieval-Augmented Generation (RAG) system based on **LightRAG** (forked from `HKUDS/LightRAG`), enhanced with:
- Distributed asynchronous document processing pipelines (Kafka & Celery workers).
- Specialized Vietnamese legal document ingestion, clause-level chunking, and hierarchy tracking.
- Multi-backend hybrid storage: Knowledge Graph (Neo4j), Vector DB (Milvus / pgvector), Relational & KV (PostgreSQL, Redis), Blob storage (MinIO/S3).
- Automated citation tracking with legal reference formatting and multi-workspace support.

---

## 2. Technology Stack & Key Dependencies
- **Core Backend**: Python `>=3.10`, `setuptools`, `uv` / `pip`
- **Web API**: FastAPI, Uvicorn, Gunicorn, Pydantic v2
- **Task Queues & Streaming**:
  - `kafka_worker`: `aiokafka` (multi-stage partitioned distributed event pipeline)
  - `celery_worker`: Celery 5.3+ with Redis broker
- **Databases & Vector Engines**:
  - PostgreSQL 16+ (`asyncpg`, `pgvector`)
  - Milvus 2.6+ (`pymilvus`) backed by etcd & MinIO
  - Neo4j 5+ (`neo4j`)
  - Redis 5+ (`redis`, LLM load balancer lease lock & caching)
  - MinIO / S3 (`aioboto3`)
- **LLM & Embedding Bindings**:
  - OpenAI, Google Gemini (`google-genai`), Anthropic, Ollama, vLLM (self-hosted embed/rerank)
  - Distributed LLM Load Balancer (`lightrag/llm/load_balancer.py`) with per-scope concurrency & token leasing
- **Document Extractors**:
  - MinerU OCR (primary for PDF & images)
  - Docling (primary for Office: docx, pptx, xlsx; fallback to MinerU)
  - PyMuPDF (`fitz`), BeautifulSoup4
- **Web UI**: Bun, Vite, React 19, TypeScript, Tailwind CSS, Lucide icons, i18next

---

## 3. Directory & Architecture Map
```
.
├── lightrag/                   # Core Python library
│   ├── lightrag.py             # Main LightRAG orchestrator class
│   ├── operate.py              # Retrieval, chunking, extraction, indexing operations
│   ├── citation.py             # Citation tracking, metadata & format rendering
│   ├── legal_document/         # Vietnamese legal parsing, chunking, evaluation
│   │   ├── chunking.py         # Article & clause-aware (Điều/Khoản) semantic splitter
│   │   ├── hierarchy.py        # Legal document structure & breadcrumbs
│   │   ├── rule_extractor.py   # Regex & rule-based legal structure extractors
│   │   └── vlegal_bench.py     # Legal benchmark & evaluation harness
│   ├── pipeline/               # Multi-stage asynchronous pipeline framework
│   │   ├── coordinator.py      # Pipeline DAG coordinator & state management
│   │   ├── runner.py           # Stage execution runner & artifact logging
│   │   ├── slo.py              # Service Level Objectives & timeout monitoring
│   │   └── stages/             # Pipeline stage handlers:
│   │       ├── extraction.py       # Full-text document extraction
│   │       ├── chunking.py         # Document chunking (general & legal)
│   │       ├── chunk_embedding.py  # Chunk embedding generation
│   │       ├── entity_extraction.py# Entity & relationship extraction via LLM
│   │       ├── kg_merge.py         # Merge entities into knowledge graph
│   │       ├── kg_embedding.py     # Entity & relation embedding generation
│   │       └── finalize.py         # Final state transition & cleanup
│   ├── kg/                     # Storage backend adapters (Postgres, Milvus, Neo4j, Redis, S3)
│   ├── llm/                    # LLM bindings & distributed load balancer
│   └── api/                    # FastAPI service application
│       ├── lightrag_server.py  # Server entrypoint & OpenAPI routing
│       ├── routers/            # Endpoint routers: document, query, graph, ollama
│       └── celery_tasks.py     # Celery task definitions
├── kafka_worker/               # Distributed Kafka consumers for pipeline stages
│   ├── pipeline_main.py        # Worker entrypoint for stage consumption
│   ├── pipeline_sweeper.py     # Sweeper for auto-advancing / retrying stages
│   ├── partitioned_publish.py  # Partition-aware Kafka producer
│   └── ingress.py              # Input event ingestion gateway
├── celery_worker/              # Celery worker service for heavy OCR / extraction
├── lightrag_webui/             # React 19 + TypeScript web application
├── infra/                      # Local infrastructure configs (Milvus, etcd, etc.)
├── scripts/                    # Setup wizard and maintenance scripts
├── tests/                      # Pytest test suite
└── scratch/                    # Temporary check scripts & inspection artifacts
```

---

## 4. Key Subsystems & Workflows

### 4.1 Document Processing Pipeline (7 Stages)
Ingestion follows a strictly tracked sequential DAG:
```
[Ingress] -> EXTRACTION -> CHUNKING -> CHUNK_EMBEDDING -> ENTITY_EXTRACTION -> KG_MERGE -> KG_EMBEDDING -> FINALIZE
```
- **Identification**: Documents use `doc_id` in `doc-{uuid4}` format and `content_hash` (MD5) for deduplication. (Legacy `track_id` is completely deprecated).
- **Status lifecycle**: `UPLOADING -> EXTRACTING -> EXTRACTED -> CHUNKING -> PROCESSED` (or `FAILED`).
- **Parallel Workers**: Dockerized worker containers run general and legal pipelines separately (`worker_chunking_general`, `worker_chunking_legal`, etc.).

### 4.2 Legal Document Chunking (`doc_type="legal"`)
- Vietnamese legal documents are chunked by Article (`Điều`) and sub-clauses (`Khoản 1.`, `Điểm a)`).
- Maintains legal hierarchy metadata: `[Chương X] [Điều Y] [Khoản Z]`.
- Tables are strictly preserved and never split across chunks.

### 4.3 Database Architecture & Operational Rules
- **Milvus Vector DB (CRITICAL RULE)**:
  > **DO NOT** touch Milvus containers or database without inspecting etcd and receiving explicit user confirmation (`.agent/rules/milvus_rules.md`).
- **PostgreSQL Connection Pool**:
  - Shared singleton per process (`ClientManager._shared_instance`).
  - Controlled by `POSTGRES_MAX_CONNECTIONS` (default: 400) and `POSTGRES_MAX_INACTIVE_CONNECTION_LIFETIME` (default: 300s).
- **File & Script Locations**:
  - All test files MUST have prefix `test_` and be located in `tests/`.
  - All check/debug scripts MUST be placed in `scratch/`.

---

## 5. Development & Testing Commands
- **Environment**:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -e .[api,offline]
  ```
- **Linting & Code Quality**:
  ```bash
  ruff check .
  ```
- **Backend Tests**:
  ```bash
  python -m pytest tests                     # Offline / unit tests only
  python -m pytest tests --run-integration   # Run with external DBs
  ```
- **Frontend Commands**:
  ```bash
  cd lightrag_webui
  bun install
  bun run dev
  bun run build
  bun test
  ```
- **Docker Compose Profiles**:
  ```bash
  docker compose -f docker-compose.dev.yml up -d      # Base dev stack
  docker compose -f docker-compose.workers.yml up -d  # Kafka workers stack
  ```
- **Upstream PR Guideline**:
  - Fork target: Always target upstream **`HKUDS/LightRAG:main`** when submitting PRs.

