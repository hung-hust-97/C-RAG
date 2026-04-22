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
from celery.signals import worker_process_init, worker_process_shutdown

@worker_process_init.connect
def init_worker_process(**kwargs):
    """
    Called when a new worker process is forked.
    Reset all asyncio locks to prevent event loop binding issues.
    """
    logger.info("[Celery] Worker process initialized, resetting locks...")
    _reset_locks_after_fork()


@worker_process_shutdown.connect
def shutdown_worker_process(**kwargs):
    """
    Called when a worker process is shutting down.
    Close all PostgreSQL connection pools to prevent connection leaks.
    
    This is critical for preventing idle connection accumulation:
    - asyncpg's max_inactive_connection_lifetime only marks connections for closure
    - Connections are only closed on next acquire() call
    - Without explicit cleanup, worker processes keep connections open indefinitely
    """
    logger.info("[Celery] Worker process shutting down, closing connection pools...")
    
    # Close all cached RAG instances and their connection pools
    closed_count = 0
    for workspace_id, (rag, _) in list(_rag_cache.items()):
        try:
            # Close PostgreSQL pools in storage backends
            async def _close_pools():
                # Close KV storage pool
                if hasattr(rag, 'key_string_value_json_storage_cls'):
                    kv_storage = rag.key_string_value_json_storage_cls
                    if hasattr(kv_storage, 'db') and hasattr(kv_storage.db, 'pool'):
                        if kv_storage.db.pool is not None:
                            await kv_storage.db.pool.close()
                            logger.debug(f"[Celery] Closed KV storage pool for workspace: {workspace_id}")
                
                # Close doc status storage pool
                if hasattr(rag, 'doc_status_storage_cls'):
                    doc_storage = rag.doc_status_storage_cls
                    if hasattr(doc_storage, 'db') and hasattr(doc_storage.db, 'pool'):
                        if doc_storage.db.pool is not None:
                            await doc_storage.db.pool.close()
                            logger.debug(f"[Celery] Closed doc status storage pool for workspace: {workspace_id}")
                
                # Close graph storage pool (if PostgreSQL)
                if hasattr(rag, 'graph_storage_cls'):
                    graph_storage = rag.graph_storage_cls
                    if hasattr(graph_storage, 'db') and hasattr(graph_storage.db, 'pool'):
                        if graph_storage.db.pool is not None:
                            await graph_storage.db.pool.close()
                            logger.debug(f"[Celery] Closed graph storage pool for workspace: {workspace_id}")
                
                # Close vector storage pool (if PostgreSQL)
                if hasattr(rag, 'vector_db_storage_cls'):
                    vector_storage = rag.vector_db_storage_cls
                    if hasattr(vector_storage, 'db') and hasattr(vector_storage.db, 'pool'):
                        if vector_storage.db.pool is not None:
                            await vector_storage.db.pool.close()
                            logger.debug(f"[Celery] Closed vector storage pool for workspace: {workspace_id}")
            
            # Run the cleanup coroutine
            asyncio.run(_close_pools())
            closed_count += 1
            logger.info(f"[Celery] Closed connection pools for workspace: {workspace_id}")
            
        except Exception as e:
            logger.warning(f"[Celery] Failed to close pools for workspace {workspace_id}: {e}")
    
    # Clear the cache
    _rag_cache.clear()
    logger.info(f"[Celery] Worker shutdown complete. Closed {closed_count} workspace connection pools.")


# ---------------------------------------------------------------------------
# LightRAG instance factory for Celery workers
# ---------------------------------------------------------------------------

_rag_cache: dict[str, tuple[object, int]] = {}


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
    Re-uses cached instances per workspace (shared pool handles multi-loop scenarios).
    
    NOTE: We cache by workspace_id only, not by event loop ID, because:
    1. PostgreSQL shared pool is designed to work across event loops
    2. Creating new instances per loop wastes connections (each instance = new pool)
    3. Storage backends properly handle event loop changes internally
    """
    if workspace_id in _rag_cache:
        rag, _ = _rag_cache[workspace_id]
        logger.debug(f"[Celery] Reusing cached RAG instance for workspace: {workspace_id}")
        return rag

    from lightrag.api.config import global_args
    from lightrag.api.lightrag_factory import LLMConfigCache, build_rag_instance

    # Initialize configuration cache from global_args
    config_cache = LLMConfigCache(global_args)

    # Use factory to build the instance
    rag = build_rag_instance(workspace_id, global_args, config_cache)

    await rag.initialize_storages()
    
    # Cache without loop_id - shared pool handles cross-loop usage
    _rag_cache[workspace_id] = (rag, 0)
    logger.info(
        f"[Celery] Initialized factory-based RAG instance for workspace: {workspace_id}"
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

        success, _ = await pipeline_enqueue_file(rag, file_path, doc_id)
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
    finally:
        # Clean up RAG instance and close connection pools after task completes
        if workspace_id in _rag_cache:
            try:
                rag, _ = _rag_cache.pop(workspace_id)
                # Close pools asynchronously
                async def _cleanup():
                    # Close PostgreSQL pools if they exist
                    if hasattr(rag, 'key_string_value_json_storage_cls'):
                        storage = rag.key_string_value_json_storage_cls
                        if hasattr(storage, 'db') and hasattr(storage.db, 'pool') and storage.db.pool:
                            await storage.db.pool.close()
                    if hasattr(rag, 'doc_status_storage_cls'):
                        storage = rag.doc_status_storage_cls
                        if hasattr(storage, 'db') and hasattr(storage.db, 'pool') and storage.db.pool:
                            await storage.db.pool.close()
                
                _run_async(_cleanup())
                logger.debug(f"[Celery/OCR] Cleaned up connection pools for workspace={workspace_id}")
            except Exception as e:
                logger.warning(f"[Celery/OCR] Failed to cleanup pools for workspace={workspace_id}: {e}")


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
    finally:
        # Clean up RAG instance and close connection pools after task completes
        if workspace_id in _rag_cache:
            try:
                rag, _ = _rag_cache.pop(workspace_id)
                # Close pools asynchronously
                async def _cleanup():
                    # Close PostgreSQL pools if they exist
                    if hasattr(rag, 'key_string_value_json_storage_cls'):
                        storage = rag.key_string_value_json_storage_cls
                        if hasattr(storage, 'db') and hasattr(storage.db, 'pool') and storage.db.pool:
                            await storage.db.pool.close()
                    if hasattr(rag, 'doc_status_storage_cls'):
                        storage = rag.doc_status_storage_cls
                        if hasattr(storage, 'db') and hasattr(storage.db, 'pool') and storage.db.pool:
                            await storage.db.pool.close()
                
                _run_async(_cleanup())
                logger.debug(f"[Celery/Graph] Cleaned up connection pools for workspace={workspace_id}")
            except Exception as e:
                logger.warning(f"[Celery/Graph] Failed to cleanup pools for workspace={workspace_id}: {e}")


# ---------------------------------------------------------------------------
# Task: Process All Workspaces with EXTRACTED Documents
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, max_retries=1, default_retry_delay=30)
def task_process_all_extracted(self):
    """
    Process all workspaces that have EXTRACTED documents.
    
    This task queries the database for all workspaces with EXTRACTED documents
    and triggers task_chunk_and_graph for each workspace.
    """
    logger.info("[Celery/ProcessAll] Starting to process all workspaces with EXTRACTED documents")

    async def _run():
        import asyncpg
        import os
        
        # Get PostgreSQL connection details from environment
        host = os.getenv("POSTGRES_HOST", "postgres")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        database = os.getenv("POSTGRES_DATABASE", "crag")
        
        # Connect to PostgreSQL
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        try:
            # Get all workspaces with EXTRACTED documents
            workspaces = await conn.fetch(
                """
                SELECT workspace, COUNT(*) as count 
                FROM lightrag_doc_status 
                WHERE status = 'EXTRACTED'
                GROUP BY workspace
                ORDER BY count DESC
                """
            )
            
            if not workspaces:
                logger.info("[Celery/ProcessAll] No workspaces with EXTRACTED documents found")
                return 0
            
            logger.info(f"[Celery/ProcessAll] Found {len(workspaces)} workspaces with EXTRACTED documents")
            
            triggered_count = 0
            for row in workspaces:
                workspace = row['workspace']
                count = row['count']
                
                try:
                    task_chunk_and_graph.delay(workspace)
                    logger.info(f"[Celery/ProcessAll] Triggered task for workspace '{workspace}' ({count} docs)")
                    triggered_count += 1
                except Exception as e:
                    logger.error(f"[Celery/ProcessAll] Failed to trigger task for workspace '{workspace}': {e}")
            
            return triggered_count
            
        finally:
            await conn.close()

    try:
        triggered = _run_async(_run())
        logger.info(f"[Celery/ProcessAll] Successfully triggered {triggered} workspace tasks")
        return triggered
    except Exception as exc:
        logger.exception(f"[Celery/ProcessAll] Error: {exc}")
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
