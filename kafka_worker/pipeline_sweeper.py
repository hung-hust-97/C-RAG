"""Pipeline sweeper: detects and recovers stuck documents.

Runs periodically to find documents that are stuck between pipeline stages
and either retries them or marks them as FAILED. This is the safety net
that prevents documents from being permanently stuck.

Usage:
    python -m kafka_worker.pipeline_sweeper          # one-shot
    python -m kafka_worker.pipeline_sweeper --loop    # continuous (every 5 min)
"""
from __future__ import annotations

import asyncio
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline_sweeper")

# Documents stuck longer than this threshold are eligible for recovery
STUCK_THRESHOLD_SECONDS = int(os.getenv("PIPELINE_SWEEPER_STUCK_THRESHOLD_SECONDS", "1800"))  # 30 min
# How often the sweeper loop runs (if --loop)
SWEEP_INTERVAL_SECONDS = int(os.getenv("PIPELINE_SWEEPER_INTERVAL_SECONDS", "300"))  # 5 min


async def _get_db():
    """Get database connection pool."""
    from lightrag.kg.postgres_impl import ClientManager
    return await ClientManager.get_client()


async def sweep_stale_running(table_name: str = "lightrag_pipeline_stage_runs") -> int:
    """Reset records stuck in 'running' beyond the threshold back to failed so they can be retried."""
    db = await _get_db()
    result = await db.execute(
        f"""
        UPDATE {table_name}
        SET status = 'failed',
            error_json = jsonb_build_object(
                'message', 'Sweeper: reset stale running stage',
                'error_type', 'StaleRunningError',
                'created_at', NOW()::text
            ),
            updated_at = NOW()
        WHERE status = 'running'
          AND EXTRACT(EPOCH FROM (NOW() - COALESCE(updated_at, started_at, NOW()))) > $1
        """,
        [STUCK_THRESHOLD_SECONDS],
    )
    # Parse "UPDATE N" to get count
    count = 0
    if result and result.startswith("UPDATE"):
        try:
            count = int(result.split()[-1])
        except (ValueError, IndexError):
            pass
    if count > 0:
        logger.warning("Sweeper: reset %s stale running records (threshold=%ss)", count, STUCK_THRESHOLD_SECONDS)
    return count


async def sweep_orphaned_documents(table_name: str = "lightrag_pipeline_stage_runs") -> int:
    """Find documents that have all extraction shards done/failed but no kg_merge record.
    
    This handles the case where entity_extraction shards all completed but the
    join event for kg_merge was never published (e.g., due to worker crash or
    Kafka delivery failure).
    """
    db = await _get_db()
    
    # Find run_ids where:
    # 1. entity_extraction shards exist
    # 2. All entity_extraction shards are in terminal state (done or failed)
    # 3. No kg_merge record exists
    # 4. The entity_extraction records are old enough (beyond threshold)
    rows = await db.query(
        f"""
        WITH ee_summary AS (
            SELECT run_id,
                   MAX(shard_count) as shard_count,
                   COUNT(*) as total_shards,
                   COUNT(*) FILTER (WHERE status IN ('done', 'failed')) as terminal_shards,
                   MIN(updated_at) as oldest_update
            FROM {table_name}
            WHERE stage = 'entity_extraction'
            GROUP BY run_id
        )
        SELECT ee.run_id, ee.shard_count, ee.total_shards, ee.terminal_shards
        FROM ee_summary ee
        WHERE ee.total_shards >= ee.shard_count
          AND ee.terminal_shards = ee.total_shards
          AND NOT EXISTS (
              SELECT 1 FROM {table_name} km
              WHERE km.run_id = ee.run_id AND km.stage = 'kg_merge'
          )
          AND EXTRACT(EPOCH FROM (NOW() - ee.oldest_update)) > $1
        LIMIT 100
        """,
        [STUCK_THRESHOLD_SECONDS],
        multirows=True,
    )
    
    if not rows:
        return 0
    
    count = len(rows)
    logger.warning(
        "Sweeper: found %s documents with completed entity_extraction but missing kg_merge", count
    )
    
    # For these orphaned documents, we cannot easily re-publish Kafka events from here.
    # Instead, mark the document as FAILED in the pipeline coordinator so the
    # auto-retry mechanism can pick them up on next reprocess.
    for row in rows:
        run_id = row["run_id"]
        logger.info("Sweeper: marking orphaned document run_id=%s as needing reprocess", run_id)
        # Insert a failed kg_merge record so it can be auto-retried
        try:
            await db.execute(
                f"""
                INSERT INTO {table_name}
                    (run_id, stage, shard_index, shard_count, workspace_id, doc_id,
                     status, attempt, started_at, updated_at,
                     error_json, artifacts_json)
                SELECT
                    $1, 'kg_merge', 0, 1,
                    MIN(workspace_id), MIN(doc_id),
                    'failed', 0, NOW(), NOW(),
                    '{{"message": "Sweeper: created missing kg_merge record"}}'::jsonb,
                    '[]'::jsonb
                FROM {table_name}
                WHERE run_id = $1 AND stage = 'entity_extraction'
                ON CONFLICT (run_id, stage, shard_index) DO NOTHING
                """,
                [run_id],
            )
        except Exception as exc:
            logger.error("Sweeper: failed to create kg_merge record for %s: %s", run_id, exc)
    
    return count


async def report_pipeline_gaps(table_name: str = "lightrag_pipeline_stage_runs") -> None:
    """Log a summary of pipeline gaps for monitoring."""
    db = await _get_db()
    row = await db.query(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE status = 'done') as done,
            COUNT(*) FILTER (WHERE status = 'running') as running,
            COUNT(*) FILTER (WHERE status = 'failed') as failed
        FROM {table_name}
        """,
    )
    if row:
        logger.info(
            "Sweeper report: done=%s running=%s failed=%s",
            row.get("done", 0),
            row.get("running", 0),
            row.get("failed", 0),
        )

    # Per-stage summary
    stage_rows = await db.query(
        f"""
        SELECT stage, status, COUNT(*) as cnt
        FROM {table_name}
        GROUP BY stage, status
        ORDER BY stage, status
        """,
        multirows=True,
    )
    if stage_rows:
        for sr in stage_rows:
            logger.info(
                "  stage=%-20s status=%-10s count=%s",
                sr.get("stage"),
                sr.get("status"),
                sr.get("cnt"),
            )


async def run_sweep() -> None:
    """Run all sweep operations once."""
    logger.info("Sweeper: starting sweep cycle (threshold=%ss)", STUCK_THRESHOLD_SECONDS)
    
    try:
        await report_pipeline_gaps()
        stale_count = await sweep_stale_running()
        orphan_count = await sweep_orphaned_documents()
        
        logger.info(
            "Sweeper: cycle complete. Reset %s stale running, found %s orphaned docs.",
            stale_count,
            orphan_count,
        )
    except Exception:
        logger.exception("Sweeper: error during sweep cycle")


async def main() -> None:
    loop_mode = "--loop" in sys.argv
    
    if loop_mode:
        logger.info("Sweeper: starting in loop mode (interval=%ss)", SWEEP_INTERVAL_SECONDS)
        while True:
            await run_sweep()
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
    else:
        await run_sweep()
    
    # Cleanup
    from lightrag.kg.postgres_impl import ClientManager
    await ClientManager.force_close_shared_pool()


if __name__ == "__main__":
    asyncio.run(main())
