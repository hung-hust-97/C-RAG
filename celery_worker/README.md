# Celery Worker Module

This module provides an independent Celery worker implementation for LightRAG background processing tasks. It is designed to run in a separate Docker container from the API server, enabling independent scaling and resource allocation.

## Module Structure

```
celery_worker/
├── __init__.py          # Exposes app and tasks
├── app.py               # Celery application instance
├── tasks.py             # Task definitions (document processing, queries)
├── config.py            # Configuration loader from environment variables
├── Dockerfile           # Worker-specific Dockerfile
└── README.md            # This file
```

## Architecture

The Celery worker module is part of a three-container architecture:

1. **UI Container**: Nginx-based container serving static React application
2. **API Container**: FastAPI server handling HTTP requests
3. **Celery Worker Container**: Background task processor (this module)

### Communication Flow

```
API Container → Redis (task queue) → Celery Worker → Storage (PostgreSQL, Neo4j, Milvus)
```

## Configuration

The worker is configured via environment variables loaded by `config.py`:

### Required Variables

- `EMBEDDING_CELERY_BROKER_URL`: Redis broker URL (default: `redis://localhost:6379/0`)
- `EMBEDDING_CELERY_RESULT_BACKEND`: Redis result backend URL (default: `redis://localhost:6379/1`)

### Optional Variables

- `CELERY_CONCURRENCY`: Number of worker processes (default: `2`)
- `WORKING_DIR`: RAG storage directory (default: `/app/data/rag_storage`)
- `INPUT_DIR`: Input files directory (default: `/app/data/inputs`)
- `LIGHTRAG_LOG_DIR`: Log directory (default: `/app/data/logs`)

### Storage Configuration

The worker shares storage configuration with the API container:

- `POSTGRES_HOST`: PostgreSQL host (default: `postgres`)
- `POSTGRES_PORT`: PostgreSQL port (default: `5432`)
- `NEO4J_URI`: Neo4j connection URI (default: `bolt://neo4j:7687`)
- `MILVUS_URI`: Milvus connection URI (default: `http://milvus:19530`)

## Tasks

The worker processes the following task types:

1. **task_extract_and_enqueue**: OCR and markdown extraction for document files
2. **task_chunk_and_graph**: Chunking and entity/relation graph extraction
3. **task_execute_query**: Offline async query execution
4. **apipeline_process_enqueue_documents_task**: Multi-workspace pipeline processing

## Running Locally

### Prerequisites

- Python 3.12+
- Redis server running
- Storage services (PostgreSQL, Neo4j, Milvus) configured

### Installation

```bash
# Install dependencies
pip install -e .[api,offline]

# Set environment variables
export EMBEDDING_CELERY_BROKER_URL="redis://localhost:6379/0"
export EMBEDDING_CELERY_RESULT_BACKEND="redis://localhost:6379/1"
export CELERY_CONCURRENCY="2"
```

### Start Worker

```bash
# Start worker with default settings
celery -A celery_worker.app worker -l info

# Start with custom concurrency
celery -A celery_worker.app worker -l info --concurrency=4

# Start with prefork pool (recommended for production)
celery -A celery_worker.app worker -l info --pool=prefork --concurrency=2
```

### Monitor Worker

```bash
# Check worker status
celery -A celery_worker.app inspect active

# Check registered tasks
celery -A celery_worker.app inspect registered

# Monitor events
celery -A celery_worker.app events
```

## Docker Deployment

### Build Image

```bash
# Build from repository root
docker build -f celery_worker/Dockerfile -t lightrag-celery-worker .
```

### Run Container

```bash
docker run -d \
  --name celery_worker \
  -e EMBEDDING_CELERY_BROKER_URL="redis://redis:6379/0" \
  -e EMBEDDING_CELERY_RESULT_BACKEND="redis://redis:6379/1" \
  -e CELERY_CONCURRENCY="2" \
  -v ./data/rag_storage:/app/data/rag_storage \
  -v ./data/inputs:/app/data/inputs \
  -v ./data/logs:/app/data/logs \
  lightrag-celery-worker
```

### Docker Compose

The worker is included in the main `docker-compose.yml`:

```yaml
celery_worker:
  build:
    context: .
    dockerfile: celery_worker/Dockerfile
  restart: unless-stopped
  volumes:
    - ./data/rag_storage:/app/data/rag_storage
    - ./data/inputs:/app/data/inputs
    - ./data/logs:/app/data/logs
  environment:
    EMBEDDING_CELERY_BROKER_URL: redis://redis:6379/0
    EMBEDDING_CELERY_RESULT_BACKEND: redis://redis:6379/1
    CELERY_CONCURRENCY: 2
  depends_on:
    - redis
    - postgres
    - neo4j
    - milvus
```

Start with:

```bash
docker-compose up -d celery_worker
```

## Health Checks

The worker container includes a health check:

```bash
celery -A celery_worker.app inspect ping -d celery@$HOSTNAME
```

This verifies the worker can connect to the broker and is ready to process tasks.

## Troubleshooting

### Worker Not Starting

1. Check Redis connection:
   ```bash
   redis-cli -h localhost -p 6379 ping
   ```

2. Verify environment variables:
   ```bash
   docker exec celery_worker env | grep CELERY
   ```

3. Check logs:
   ```bash
   docker logs celery_worker
   ```

### Tasks Not Processing

1. Verify worker is registered:
   ```bash
   celery -A celery_worker.app inspect active
   ```

2. Check task queue:
   ```bash
   redis-cli -h localhost -p 6379 llen celery
   ```

3. Monitor worker logs for errors:
   ```bash
   docker logs -f celery_worker
   ```

### Storage Connection Issues

1. Verify storage services are running:
   ```bash
   docker ps | grep -E "postgres|neo4j|milvus"
   ```

2. Test connections from worker container:
   ```bash
   docker exec celery_worker nc -zv postgres 5432
   docker exec celery_worker nc -zv neo4j 7687
   docker exec celery_worker nc -zv milvus 19530
   ```

## Development

### Adding New Tasks

1. Define task in `celery_worker/tasks.py`:
   ```python
   @celery_app.task(bind=True)
   def my_new_task(self, arg1, arg2):
       # Task implementation
       pass
   ```

2. Import in API server:
   ```python
   from celery_worker.tasks import my_new_task
   
   # Enqueue task
   my_new_task.delay(arg1, arg2)
   ```

### Testing

Run tests from repository root:

```bash
# Unit tests
python -m pytest tests/test_celery_worker.py

# Integration tests (requires Redis and storage services)
python -m pytest tests/test_celery_worker.py --run-integration
```

## Migration from Old Architecture

This module replaces the previous implementation in `lightrag/api/celery_app.py` and `lightrag/api/celery_tasks.py`. The task names and signatures remain unchanged for backward compatibility.

### Import Changes

Old:
```python
from lightrag.api.celery_app import celery_app
from lightrag.api.celery_tasks import task_extract_and_enqueue
```

New:
```python
from celery_worker.app import celery_app
from celery_worker.tasks import task_extract_and_enqueue
```

## Performance Tuning

### Concurrency

Adjust `CELERY_CONCURRENCY` based on workload:

- **CPU-bound tasks**: Set to number of CPU cores
- **I/O-bound tasks**: Set to 2-4x number of CPU cores
- **LLM API calls**: Keep low (2-4) to avoid rate limits

### Worker Pool

- **prefork**: Best for CPU-bound tasks (default)
- **gevent**: Best for I/O-bound tasks
- **solo**: Single-threaded, useful for debugging

Example:
```bash
celery -A celery_worker.app worker --pool=gevent --concurrency=100
```

### Task Routing

For advanced setups, route different task types to different queues:

```python
celery_app.conf.task_routes = {
    'celery_worker.tasks.task_extract_and_enqueue': {'queue': 'ocr'},
    'celery_worker.tasks.task_chunk_and_graph': {'queue': 'graph'},
}
```

Start specialized workers:
```bash
celery -A celery_worker.app worker -Q ocr -n ocr_worker
celery -A celery_worker.app worker -Q graph -n graph_worker
```

## Security Considerations

1. **Network Isolation**: Worker doesn't expose any ports
2. **Secrets Management**: Load credentials from environment variables only
3. **Resource Limits**: Set memory limits in docker-compose to prevent OOM
4. **Logging**: Never log sensitive data (API keys, passwords)

## License

This module is part of the LightRAG project and follows the same license.
