"""
Celery tasks for C-RAG document ingestion pipeline.

Architecture:
  task_extract_and_enqueue  --> OCR / Markdown extraction for a single file
  task_chunk_and_graph      --> Chunking + Entity/Relation extraction (graph build)
  task_execute_query        --> Offline async query execution
"""

import asyncio
import logging
import os
from pathlib import Path

from lightrag.api.celery_app import celery_app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LightRAG instance factory for Celery workers
# ---------------------------------------------------------------------------

_rag_cache: dict[str, tuple[object, int]] = {}


def _run_async(coro):
    """Run an async coroutine from a synchronous Celery task."""
    try:
        # Check if there is an existing running loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If a loop is already running, run the task in a separate thread with a new loop
        # to avoid 'Event loop is closed' errors or nested event loop issues.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        # Standard case: create a new loop, run the coroutine, and close the loop.
        return asyncio.run(coro)


async def _get_rag(workspace_id: str):
    """
    Build and initialize a LightRAG instance from environment variables.
    Re-uses cached instances within the same event loop (since storage connections
    are often bound to the loop they were initialized in).
    """
    try:
        current_loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        current_loop_id = 0

    if workspace_id in _rag_cache:
        rag, loop_id = _rag_cache[workspace_id]
        if loop_id == current_loop_id:
            return rag
        else:
            logger.debug(
                f"[Celery] Event loop changed for {workspace_id} ({loop_id} -> {current_loop_id}), recreating instance."
            )

    from lightrag.api.config import global_args
    from lightrag.api.lightrag_factory import LLMConfigCache, build_rag_instance

    # Initialize configuration cache from global_args
    config_cache = LLMConfigCache(global_args)

    # Use factory to build the instance
    rag = build_rag_instance(workspace_id, global_args, config_cache)

    await rag.initialize_storages()
    _rag_cache[workspace_id] = (rag, current_loop_id)
    logger.info(
        f"[Celery] Initialized factory-based RAG instance for workspace: {workspace_id} (Loop ID: {current_loop_id})"
    )
    return rag


# ---------------------------------------------------------------------------
# Task 1: OCR / Markdown Extraction
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def task_extract_and_enqueue(self, workspace_id: str, file_path_str: str, track_id: str):
    """
    Celery Task 1 – OCR & Markdown Extraction.

    Picks up a file from disk, runs DeepSeek/Docling to produce Markdown text,
    and writes a PENDING doc_status record. Then automatically triggers
    task_chunk_and_graph to do the heavy graph work.
    """
    logger.info(f"[Celery/OCR] workspace={workspace_id} file={file_path_str}")

    async def _run():
        from lightrag.api.routers.document_routes import pipeline_enqueue_file

        rag = await _get_rag(workspace_id)
        file_path = Path(file_path_str)

        if not file_path.exists():
            logger.error(f"[Celery/OCR] File not found: {file_path_str}")
            return False

        success, _ = await pipeline_enqueue_file(rag, file_path, track_id)
        return success

    try:
        success = _run_async(_run())
        if success:
            logger.info(f"[Celery/OCR] Extraction OK → queuing graph task for {workspace_id}")
            task_chunk_and_graph.delay(workspace_id)
        else:
            logger.error(f"[Celery/OCR] Extraction FAILED for {file_path_str}")
    except Exception as exc:
        logger.exception(f"[Celery/OCR] Unexpected error for {file_path_str}: {exc}")
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task 2+3: Chunking + Entity/Relation Graph Extraction
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=1, default_retry_delay=30)
def task_chunk_and_graph(self, workspace_id: str):
    """
    Celery Task 2&3 – Chunking and Graph Build.

    Processes ALL PENDING documents in the given workspace through the
    LightRAG pipeline (chunking → entity extraction → graph upsert).
    This mirrors the logic of apipeline_process_enqueue_documents().
    """
    logger.info(f"[Celery/Graph] Starting for workspace={workspace_id}")

    async def _run():
        rag = await _get_rag(workspace_id)
        await rag.apipeline_process_enqueue_documents()

    try:
        _run_async(_run())
        logger.info(f"[Celery/Graph] Finished for workspace={workspace_id}")
    except Exception as exc:
        logger.exception(f"[Celery/Graph] Error for workspace={workspace_id}: {exc}")
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Task 4: Async User Query
# ---------------------------------------------------------------------------

@celery_app.task(bind=True)
def task_execute_query(self, workspace_id: str, query_text: str, query_params: dict):
    """
    Celery Task 4 – Offline User Query.

    Executes a RAG query outside the request-response cycle and stores the
    result in the Celery result backend (Redis) for later retrieval via
    GET /query/query_status/{task_id}.
    """
    logger.info(f"[Celery/Query] workspace={workspace_id} query={query_text[:60]!r}")

    async def _run():
        from lightrag.base import QueryParam

        rag = await _get_rag(workspace_id)

        mode = query_params.get("mode", "mix")
        top_k = query_params.get("top_k") or 40
        max_total_tokens = query_params.get("max_total_tokens") or 8192

        param = QueryParam(
            mode=mode,
            top_k=top_k,
            max_total_tokens=max_total_tokens,
        )
        response = await rag.aquery(query_text, param=param)
        return str(response)

    try:
        result = _run_async(_run())
        logger.info(f"[Celery/Query] Done for workspace={workspace_id}")
        return {"query": query_text, "response": result}
    except Exception as exc:
        logger.exception(f"[Celery/Query] Error: {exc}")
        raise

# ---------------------------------------------------------------------------
# Task 5: Pipeline Processing (Multi-Workspace)
# ---------------------------------------------------------------------------

@celery_app.task(name="lightrag.api.lightrag_server.apipeline_process_enqueue_documents_task", bind=True, max_retries=1, default_retry_delay=30)
def apipeline_process_enqueue_documents_task(self, workspace_id: str, reprocess_failed: bool = False):
    """
    Celery task to process/enqueue documents for a specific workspace.
    """
    from lightrag.kg.shared_storage import initialize_share_data
    initialize_share_data()
    
    logger.info(f"[Celery/Pipeline] Starting for workspace={workspace_id} (reprocess_failed={reprocess_failed})")

    async def _run():
        rag = await _get_rag(workspace_id)
        # Use our modified apipeline_process_enqueue_documents that supports reprocess_failed
        await rag.apipeline_process_enqueue_documents(reprocess_failed=reprocess_failed)

    try:
        _run_async(_run())
        logger.info(f"[Celery/Pipeline] Finished for workspace={workspace_id}")
    except Exception as exc:
        logger.exception(f"[Celery/Pipeline] Error for workspace={workspace_id}: {exc}")
        raise self.retry(exc=exc)
