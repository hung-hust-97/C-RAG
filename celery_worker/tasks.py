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

from celery_worker.app import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker initialization hook - Reset locks after fork
# ---------------------------------------------------------------------------

def _reset_locks_after_fork():
    """
    Reset asyncio locks after Celery fork to prevent 'bound to different event loop' errors.
    
    When Celery uses prefork mode, child processes inherit locks from parent process.
    These locks are bound to the parent's event loop and cannot be used in child processes.
    This function clears the lock caches so new locks will be created in the child's event loop.
    """
    try:
        # Import the shared storage module
        from lightrag.kg import shared_storage
        
        # Reset the keyed lock manager's async lock caches
        if hasattr(shared_storage, '_keyed_lock_manager') and shared_storage._keyed_lock_manager is not None:
            manager = shared_storage._keyed_lock_manager
            # Clear async lock caches
            if hasattr(manager, '_async_lock'):
                manager._async_lock.clear()
            if hasattr(manager, '_async_lock_count'):
                manager._async_lock_count.clear()
            if hasattr(manager, '_async_lock_cleanup_data'):
                manager._async_lock_cleanup_data.clear()
            logger.debug("[Celery] Reset keyed lock manager async locks after fork")
        
        # Reset global async locks for multiprocess mode
        if hasattr(shared_storage, '_async_locks') and shared_storage._async_locks is not None:
            # Recreate all async locks with new event loop
            shared_storage._async_locks = {
                "internal_lock": asyncio.Lock(),
                "graph_db_lock": asyncio.Lock(),
                "data_init_lock": asyncio.Lock(),
            }
            logger.debug("[Celery] Reset global async locks after fork")
        
        # Reset single-process mode locks if they exist
        if hasattr(shared_storage, '_internal_lock') and isinstance(shared_storage._internal_lock, asyncio.Lock):
            shared_storage._internal_lock = asyncio.Lock()
            logger.debug("[Celery] Reset internal lock after fork")
        
        if hasattr(shared_storage, '_data_init_lock') and isinstance(shared_storage._data_init_lock, asyncio.Lock):
            shared_storage._data_init_lock = asyncio.Lock()
            logger.debug("[Celery] Reset data init lock after fork")
            
        logger.info("[Celery] Successfully reset all asyncio locks after fork")
    except Exception as e:
        logger.warning(f"[Celery] Failed to reset locks after fork: {e}")
        import traceback
        logger.warning(traceback.format_exc())


# Register worker process init signal
from celery.signals import worker_process_init

@worker_process_init.connect
def init_worker_process(**kwargs):
    """
    Called when a new worker process is forked.
    Reset all asyncio locks to prevent event loop binding issues.
    """
    logger.info("[Celery] Worker process initialized, resetting locks...")
    _reset_locks_after_fork()


# ---------------------------------------------------------------------------
# LightRAG instance factory for Celery workers
# ---------------------------------------------------------------------------

_rag_cache: dict[tuple[str, int], tuple[object, int]] = {}


def _get_rag_key(workspace_id: str) -> tuple[str, int]:
    """Get a cache key that is both workspace and event-loop aware."""
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        loop_id = 0
    return (workspace_id, loop_id)


def _run_async(coro):
    """Run an async coroutine from a synchronous Celery task."""
    # Reset locks after fork to prevent event loop binding issues
    _reset_locks_after_fork()
    
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
    Re-uses cached instances per workspace and event loop.
    """
    cache_key = _get_rag_key(workspace_id)
    if cache_key in _rag_cache:
        rag, _ = _rag_cache[cache_key]
        logger.debug(f"[Celery] Reusing cached RAG instance for workspace: {workspace_id} (loop: {cache_key[1]})")
        return rag

    from lightrag.api.config import global_args
    from lightrag.api.lightrag_factory import LLMConfigCache, build_rag_instance

    # Initialize configuration cache from global_args
    config_cache = LLMConfigCache(global_args)

    # Use factory to build the instance
    rag = build_rag_instance(workspace_id, global_args, config_cache)

    await rag.initialize_storages()
    
    # Cache with loop_id to prevent loop-binding issues
    _rag_cache[cache_key] = (rag, 0)
    logger.info(
        f"[Celery] Initialized factory-based RAG instance for workspace: {workspace_id} (loop: {cache_key[1]})"
    )
    return rag


# ---------------------------------------------------------------------------
# Task 1: OCR / Markdown Extraction
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def task_extract_and_enqueue(self, workspace_id: str, file_path_str: str, doc_id: str):
    """
    Celery Task 1 – OCR & Markdown Extraction.

    Picks up a file from disk, runs DeepSeek/Docling to produce Markdown text,
    and writes a PENDING doc_status record. Then automatically triggers
    task_chunk_and_graph to do the heavy graph work.
    """
    logger.info(f"[Celery/OCR] workspace={workspace_id} file={file_path_str} doc_id={doc_id}")

    async def _run():
        from lightrag.api.routers.document_routes import pipeline_enqueue_file

        rag = await _get_rag(workspace_id)
        file_path = Path(file_path_str)

        if not file_path.exists():
            logger.error(f"[Celery/OCR] File not found: {file_path_str}")
            return False

        success, _ = await pipeline_enqueue_file(rag, file_path, doc_id, task_id=self.request.id)
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
        await rag.apipeline_process_enqueue_documents(task_id=self.request.id)

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
        await rag.apipeline_process_enqueue_documents(reprocess_failed=reprocess_failed, task_id=self.request.id)

    try:
        _run_async(_run())
        logger.info(f"[Celery/Pipeline] Finished for workspace={workspace_id}")
    except Exception as exc:
        logger.exception(f"[Celery/Pipeline] Error for workspace={workspace_id}: {exc}")
        raise self.retry(exc=exc)
