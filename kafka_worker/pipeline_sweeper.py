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
from typing import Any

from aiokafka import AIOKafkaProducer

from kafka_worker.config import KAFKA_BOOTSTRAP_SERVERS
from kafka_worker.partitioned_publish import publish_pipeline_event
from kafka_worker.pipeline_main import (
    STAGE_TOPICS,
    _canonical_doc_type,
    _normalize_event_doc_type,
)
from lightrag.pipeline.models import PipelineStageEvent, StageName, utc_now_iso
from lightrag.pipeline.partitioning import create_partition_allocator_from_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline_sweeper")

# Documents stuck longer than this threshold are eligible for recovery
STUCK_THRESHOLD_SECONDS = int(
    os.getenv(
        "PIPELINE_SWEEPER_STUCK_THRESHOLD_SECONDS",
        os.getenv("PIPELINE_STAGE_RUNNING_STALE_SECONDS", "10800"),
    )
)
# Max stale rows requeued in a single sweep cycle
STALE_BATCH_SIZE = int(os.getenv("PIPELINE_SWEEPER_STALE_BATCH_SIZE", "100"))
# Automatically approve paused events for the next stage once prerequisite
# stages no longer have running/paused work in the same workspace.
AUTO_APPROVE_NEXT_STAGE = os.getenv(
    "PIPELINE_SWEEPER_AUTO_APPROVE_NEXT_STAGE", "true"
).lower() in {"1", "true", "yes", "on"}
AUTO_APPROVE_BATCH_SIZE = int(
    os.getenv("PIPELINE_SWEEPER_AUTO_APPROVE_BATCH_SIZE", "500")
)
AUTO_APPROVE_REPUBLISH_COOLDOWN_SECONDS = int(
    os.getenv("PIPELINE_SWEEPER_AUTO_APPROVE_REPUBLISH_COOLDOWN_SECONDS", "900")
)
# How often the sweeper loop runs (if --loop)
SWEEP_INTERVAL_SECONDS = int(os.getenv("PIPELINE_SWEEPER_INTERVAL_SECONDS", "300"))  # 5 min

STAGE_AUTO_APPROVE_DEPENDENCIES: dict[StageName, tuple[StageName, ...]] = {
    StageName.CHUNKING: (StageName.EXTRACTION,),
    StageName.CHUNK_EMBEDDING: (StageName.CHUNKING,),
    StageName.ENTITY_EXTRACTION: (StageName.CHUNKING,),
    StageName.KG_MERGE: (StageName.ENTITY_EXTRACTION,),
    StageName.KG_EMBEDDING: (StageName.KG_MERGE,),
    StageName.FINALIZE: (StageName.CHUNK_EMBEDDING, StageName.KG_EMBEDDING),
}


async def _get_db():
    """Get database connection pool."""
    from lightrag.kg.postgres_impl import ClientManager
    return await ClientManager.get_client()


def _event_from_stale_row(row: dict[str, Any]) -> PipelineStageEvent:
    """Build a fresh retry event from the coordinator's stored event_json."""
    raw_event = row.get("event_json")
    if not raw_event:
        raise ValueError("missing event_json")

    if isinstance(raw_event, str):
        event = PipelineStageEvent.model_validate_json(raw_event)
    else:
        event = PipelineStageEvent.model_validate(raw_event)

    next_attempt = max(int(row.get("attempt") or 0) + 1, int(event.attempt or 0) + 1)
    payload = event.model_dump(mode="json")
    payload.pop("event_id", None)
    payload.update(
        {
            "run_id": row["run_id"],
            "workspace_id": row["workspace_id"],
            "doc_id": row["doc_id"],
            "stage": row["stage"],
            "shard_index": int(row.get("shard_index") or 0),
            "shard_count": int(row.get("shard_count") or event.shard_count or 1),
            "attempt": next_attempt,
            "status": "pending",
            "created_at": utc_now_iso(),
        }
    )
    return PipelineStageEvent.model_validate(payload)


def _event_from_paused_row(row: dict[str, Any]) -> PipelineStageEvent:
    """Build an approval event from a paused coordinator row."""
    raw_event = row.get("event_json")
    if not raw_event:
        raise ValueError("missing event_json")

    if isinstance(raw_event, str):
        event = PipelineStageEvent.model_validate_json(raw_event)
    else:
        event = PipelineStageEvent.model_validate(raw_event)

    payload = event.model_dump(mode="json")
    payload.update(
        {
            "run_id": row["run_id"],
            "workspace_id": row["workspace_id"],
            "doc_id": row["doc_id"],
            "stage": row["stage"],
            "shard_index": int(row.get("shard_index") or 0),
            "shard_count": int(row.get("shard_count") or event.shard_count or 1),
            "attempt": int(row.get("attempt") or event.attempt or 1),
            "status": "pending",
            "created_at": utc_now_iso(),
        }
    )
    return PipelineStageEvent.model_validate(payload)


async def _publish_stage_events(events: list[PipelineStageEvent]) -> int:
    if not events:
        return 0
    if not KAFKA_BOOTSTRAP_SERVERS:
        raise RuntimeError("KAFKA_BOOTSTRAP_SERVERS is not configured")

    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    partition_allocator = create_partition_allocator_from_env()
    published = 0
    await producer.start()
    try:
        for event in events:
            event = _normalize_event_doc_type(event)
            doc_type = _canonical_doc_type(event)
            topic = STAGE_TOPICS[event.stage_name][doc_type]
            await publish_pipeline_event(producer, partition_allocator, topic, event)
            published += 1
    finally:
        await producer.stop()
        await partition_allocator.finalize()

    return published


async def _mark_auto_approved_rows(
    rows: list[dict[str, Any]],
    table_name: str = "lightrag_pipeline_stage_runs",
) -> None:
    if not rows:
        return

    db = await _get_db()
    await db.execute(
        f"""
        WITH approved_keys AS (
            SELECT *
            FROM UNNEST($1::text[], $2::text[], $3::int[])
                 AS t(run_id, stage, shard_index)
        )
        UPDATE {table_name} sr
        SET error_json = COALESCE(sr.error_json, '{{}}'::jsonb) || jsonb_build_object(
                'auto_approved_by', 'pipeline_sweeper',
                'auto_approved_at', NOW()::text,
                'auto_approved_at_epoch', EXTRACT(EPOCH FROM NOW())
            )
        FROM approved_keys ak
        WHERE sr.run_id = ak.run_id
          AND sr.stage = ak.stage
          AND sr.shard_index = ak.shard_index
          AND sr.status = 'paused'
        """,
        [
            [str(row["run_id"]) for row in rows],
            [str(row["stage"]) for row in rows],
            [int(row["shard_index"]) for row in rows],
        ],
    )


async def auto_approve_next_stages(
    table_name: str = "lightrag_pipeline_stage_runs",
) -> int:
    """Approve paused next-stage events once prerequisite stages are clear."""
    if not AUTO_APPROVE_NEXT_STAGE:
        return 0

    db = await _get_db()
    events: list[PipelineStageEvent] = []
    approved_rows: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = {}

    for target_stage, dependency_stages in STAGE_AUTO_APPROVE_DEPENDENCIES.items():
        remaining_capacity = AUTO_APPROVE_BATCH_SIZE - len(events)
        if remaining_capacity <= 0:
            break

        rows = await db.query(
            f"""
            WITH target_workspaces AS (
                SELECT DISTINCT workspace_id
                FROM {table_name}
                WHERE stage = $1 AND status = 'paused'
            ),
            blocked_workspaces AS (
                SELECT DISTINCT workspace_id
                FROM {table_name}
                WHERE stage = ANY($2::text[])
                  AND status IN ('running', 'paused')
            )
            SELECT sr.run_id, sr.stage, sr.shard_index, sr.shard_count,
                   sr.workspace_id, sr.doc_id, sr.attempt, sr.event_json
            FROM {table_name} sr
            JOIN target_workspaces tw ON tw.workspace_id = sr.workspace_id
            LEFT JOIN blocked_workspaces bw ON bw.workspace_id = sr.workspace_id
            WHERE sr.stage = $1
              AND sr.status = 'paused'
              AND bw.workspace_id IS NULL
              AND (
                  sr.error_json IS NULL
                  OR sr.error_json->>'auto_approved_at_epoch' IS NULL
                  OR (
                      CASE
                          WHEN (sr.error_json->>'auto_approved_at_epoch') ~ '^[0-9]+(\\.[0-9]+)?$'
                          THEN (sr.error_json->>'auto_approved_at_epoch')::double precision
                          ELSE 0
                      END
                  ) <= EXTRACT(EPOCH FROM NOW()) - $3
              )
            ORDER BY sr.workspace_id, sr.run_id, sr.shard_index
            LIMIT $4
            """,
            [
                target_stage.value,
                [stage.value for stage in dependency_stages],
                AUTO_APPROVE_REPUBLISH_COOLDOWN_SECONDS,
                remaining_capacity,
            ],
            multirows=True,
        )

        for row in rows or []:
            try:
                events.append(_event_from_paused_row(row))
                approved_rows.append(row)
                stage_counts[target_stage.value] = (
                    stage_counts.get(target_stage.value, 0) + 1
                )
            except Exception as exc:
                logger.error(
                    "Sweeper: cannot auto-approve paused row run_id=%s stage=%s shard=%s: %s",
                    row.get("run_id"),
                    row.get("stage"),
                    row.get("shard_index"),
                    exc,
                )

    published = await _publish_stage_events(events)
    if published:
        await _mark_auto_approved_rows(approved_rows, table_name)
        logger.warning(
            "Sweeper: auto-approved %s paused next-stage records after previous stages cleared: %s",
            published,
            stage_counts,
        )
    return published


async def sweep_stale_running(table_name: str = "lightrag_pipeline_stage_runs") -> int:
    """Requeue records stuck in 'running' beyond the threshold at the same stage."""
    db = await _get_db()
    rows = await db.query(
        f"""
        SELECT run_id, stage, shard_index, shard_count, workspace_id, doc_id,
               attempt, event_json,
               EXTRACT(EPOCH FROM (NOW() - COALESCE(updated_at, started_at, NOW()))) AS age_seconds
        FROM {table_name}
        WHERE status = 'running'
          AND EXTRACT(EPOCH FROM (NOW() - COALESCE(updated_at, started_at, NOW()))) > $1
        ORDER BY COALESCE(updated_at, started_at, NOW()) ASC
        LIMIT $2
        """,
        [STUCK_THRESHOLD_SECONDS, STALE_BATCH_SIZE],
        multirows=True,
    )

    if not rows:
        return 0

    events: list[PipelineStageEvent] = []
    for row in rows:
        try:
            events.append(_event_from_stale_row(row))
        except Exception as exc:
            logger.error(
                "Sweeper: cannot requeue stale running record run_id=%s stage=%s shard=%s: %s",
                row.get("run_id"),
                row.get("stage"),
                row.get("shard_index"),
                exc,
            )

    published = await _publish_stage_events(events)
    if published > 0:
        logger.warning(
            "Sweeper: requeued %s stale running records at the same stage (found=%s threshold=%ss)",
            published,
            len(rows),
            STUCK_THRESHOLD_SECONDS,
        )
    return published


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
        auto_approved_count = await auto_approve_next_stages()
        
        logger.info(
            "Sweeper: cycle complete. Requeued %s stale running, found %s orphaned docs, auto-approved %s paused next-stage records.",
            stale_count,
            orphan_count,
            auto_approved_count,
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
