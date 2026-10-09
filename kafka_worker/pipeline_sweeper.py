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
CHUNK_EMBEDDING_STUCK_THRESHOLD_SECONDS = int(
    os.getenv(
        "PIPELINE_SWEEPER_STUCK_THRESHOLD_SECONDS_CHUNK_EMBEDDING",
        os.getenv(
            "PIPELINE_STAGE_RUNNING_STALE_SECONDS_CHUNK_EMBEDDING",
            str(STUCK_THRESHOLD_SECONDS),
        ),
    )
)
# Max stale rows requeued in a single sweep cycle
STALE_BATCH_SIZE = int(os.getenv("PIPELINE_SWEEPER_STALE_BATCH_SIZE", "100"))
STALE_REQUEUE_COOLDOWN_SECONDS = int(
    os.getenv("PIPELINE_SWEEPER_STALE_REQUEUE_COOLDOWN_SECONDS", "300")
)
FAILED_CHUNK_EMBEDDING_RETRY_BATCH_SIZE = int(
    os.getenv("PIPELINE_FAILED_CHUNK_EMBEDDING_RETRY_BATCH_SIZE", str(STALE_BATCH_SIZE))
)
# Automatically approve paused events for the next stage once prerequisite
# stages no longer have running/paused work in the same workspace.
AUTO_APPROVE_NEXT_STAGE = os.getenv(
    "PIPELINE_SWEEPER_AUTO_APPROVE_NEXT_STAGE", "true"
).lower() in {"1", "true", "yes", "on"}
AUTO_APPROVE_BATCH_SIZE = int(
    os.getenv("PIPELINE_SWEEPER_AUTO_APPROVE_BATCH_SIZE", "500")
)
AUTO_APPROVE_STAGE_BATCH_SIZE = int(
    os.getenv(
        "PIPELINE_SWEEPER_AUTO_APPROVE_STAGE_BATCH_SIZE",
        str(AUTO_APPROVE_BATCH_SIZE),
    )
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


def _stage_auto_approve_batch_size(stage: StageName) -> int:
    stage_key = stage.value.upper()
    for env_name in (
        f"PIPELINE_SWEEPER_AUTO_APPROVE_{stage_key}_BATCH_SIZE",
        f"PIPELINE_SWEEPER_AUTO_APPROVE_STAGE_BATCH_SIZE_{stage_key}",
    ):
        value = os.getenv(env_name)
        if value is None or not value.strip():
            continue
        try:
            return max(0, int(value))
        except ValueError:
            logger.warning("Invalid %s=%r; using global stage batch size", env_name, value)
            break
    return max(1, AUTO_APPROVE_STAGE_BATCH_SIZE)


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


async def _mark_failed_chunk_embedding_requeued_rows(
    rows: list[dict[str, Any]],
    table_name: str = "lightrag_pipeline_stage_runs",
) -> None:
    if not rows:
        return

    db = await _get_db()
    await db.execute(
        f"""
        WITH requeued_keys AS (
            SELECT *
            FROM UNNEST($1::text[], $2::text[], $3::int[])
                 AS t(run_id, stage, shard_index)
        )
        UPDATE {table_name} sr
        SET updated_at = NOW(),
            error_json = COALESCE(sr.error_json, '{{}}'::jsonb) || jsonb_build_object(
                'failed_requeued_by', 'pipeline_sweeper',
                'failed_requeued_at', NOW()::text,
                'failed_requeued_at_epoch', EXTRACT(EPOCH FROM NOW())
            )
        FROM requeued_keys rk
        WHERE sr.run_id = rk.run_id
          AND sr.stage = rk.stage
          AND sr.shard_index = rk.shard_index
          AND sr.status = 'failed'
        """,
        [
            [str(row["run_id"]) for row in rows],
            [str(row["stage"]) for row in rows],
            [int(row["shard_index"]) for row in rows],
        ],
    )


async def _mark_stale_running_requeued_rows(
    rows: list[dict[str, Any]],
    table_name: str = "lightrag_pipeline_stage_runs",
) -> None:
    if not rows:
        return

    db = await _get_db()
    await db.execute(
        f"""
        WITH requeued_keys AS (
            SELECT *
            FROM UNNEST($1::text[], $2::text[], $3::int[])
                 AS t(run_id, stage, shard_index)
        )
        UPDATE {table_name} sr
        SET error_json = COALESCE(sr.error_json, '{{}}'::jsonb) || jsonb_build_object(
                'stale_requeued_by', 'pipeline_sweeper',
                'stale_requeued_at', NOW()::text,
                'stale_requeued_at_epoch', EXTRACT(EPOCH FROM NOW())
            )
        FROM requeued_keys rk
        WHERE sr.run_id = rk.run_id
          AND sr.stage = rk.stage
          AND sr.shard_index = rk.shard_index
          AND sr.status = 'running'
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
        total_remaining_capacity = AUTO_APPROVE_BATCH_SIZE - len(events)
        if total_remaining_capacity <= 0:
            break
        stage_capacity = _stage_auto_approve_batch_size(target_stage)
        if stage_capacity <= 0:
            continue
        remaining_capacity = min(total_remaining_capacity, stage_capacity)

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
                  ) <= EXTRACT(EPOCH FROM NOW()) - $3::double precision
              )
            ORDER BY sr.workspace_id, sr.run_id, sr.shard_index
            LIMIT $4::int
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
          AND EXTRACT(EPOCH FROM (NOW() - COALESCE(updated_at, started_at, NOW()))) >
              CASE
                  WHEN stage = 'chunk_embedding' THEN $4::double precision
                  ELSE $1::double precision
              END
          AND (
              error_json IS NULL
              OR error_json->>'stale_requeued_at_epoch' IS NULL
              OR (
                  CASE
                      WHEN (error_json->>'stale_requeued_at_epoch') ~ '^[0-9]+(\\.[0-9]+)?$'
                      THEN (error_json->>'stale_requeued_at_epoch')::double precision
                      ELSE 0
                  END
              ) <= EXTRACT(EPOCH FROM NOW()) - $2::double precision
          )
        ORDER BY COALESCE(updated_at, started_at, NOW()) ASC
        LIMIT $3::int
        """,
        [
            STUCK_THRESHOLD_SECONDS,
            STALE_REQUEUE_COOLDOWN_SECONDS,
            STALE_BATCH_SIZE,
            CHUNK_EMBEDDING_STUCK_THRESHOLD_SECONDS,
        ],
        multirows=True,
    )

    if not rows:
        return 0

    events: list[PipelineStageEvent] = []
    event_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            events.append(_event_from_stale_row(row))
            event_rows.append(row)
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
        await _mark_stale_running_requeued_rows(event_rows[:published], table_name)
        logger.warning(
            "Sweeper: requeued %s stale running records at the same stage (found=%s threshold=%ss chunk_embedding_threshold=%ss cooldown=%ss)",
            published,
            len(rows),
            STUCK_THRESHOLD_SECONDS,
            CHUNK_EMBEDDING_STUCK_THRESHOLD_SECONDS,
            STALE_REQUEUE_COOLDOWN_SECONDS,
        )
    return published


async def _mark_failed_requeued_rows(
    rows: list[dict[str, Any]],
    table_name: str = "lightrag_pipeline_stage_runs",
) -> None:
    if not rows:
        return

    db = await _get_db()
    await db.execute(
        f"""
        WITH requeued_keys AS (
            SELECT *
            FROM UNNEST($1::text[], $2::text[], $3::int[])
                 AS t(run_id, stage, shard_index)
        )
        UPDATE {table_name} sr
        SET updated_at = NOW(),
            error_json = COALESCE(sr.error_json, '{{}}'::jsonb) || jsonb_build_object(
                'failed_requeued_by', 'pipeline_sweeper',
                'failed_requeued_at', NOW()::text,
                'failed_requeued_at_epoch', EXTRACT(EPOCH FROM NOW())
            )
        FROM requeued_keys rk
        WHERE sr.run_id = rk.run_id
          AND sr.stage = rk.stage
          AND sr.shard_index = rk.shard_index
          AND sr.status = 'failed'
        """,
        [
            [str(row["run_id"]) for row in rows],
            [str(row["stage"]) for row in rows],
            [int(row["shard_index"]) for row in rows],
        ],
    )


async def sweep_failed_stage_records(
    table_name: str = "lightrag_pipeline_stage_runs",
) -> int:
    """Requeue failed stage records across all stages that still have retry budget."""
    max_attempts = int(os.getenv("PIPELINE_MAX_STAGE_RETRY_ATTEMPTS", "5"))
    retry_cooldown_seconds = int(
        os.getenv("PIPELINE_FAILED_RETRY_COOLDOWN_SECONDS", "300")
    )
    batch_size = int(os.getenv("PIPELINE_SWEEPER_FAILED_RETRY_BATCH_SIZE", "500"))

    db = await _get_db()

    # Fast cleanup: mark downstream stage records with missing artifacts as max_attempts
    # because retrying downstream stages without the required S3 artifact will never succeed.
    try:
        await db.execute(
            f"""
            UPDATE {table_name}
            SET attempt = $1::int, updated_at = NOW()
            WHERE status = 'failed'
              AND attempt < $1::int
              AND stage != 'extraction'
              AND (
                  error_json::text LIKE '%Artifact not found%'
                  OR error_json->>'error_type' = 'FileNotFoundError'
                  OR error_json::text LIKE '%received no chunks%'
              )
            """,
            [max_attempts],
        )
    except Exception as exc:
        logger.warning("Sweeper: failed to mark missing artifact records as exhausted: %s", exc)

    rows = await db.query(
        f"""
        SELECT run_id, stage, shard_index, shard_count, workspace_id, doc_id,
               attempt, event_json,
               EXTRACT(EPOCH FROM (NOW() - COALESCE(updated_at, started_at, NOW()))) AS age_seconds
        FROM {table_name}
        WHERE status = 'failed'
          AND attempt < $1::int
          AND EXTRACT(EPOCH FROM (NOW() - COALESCE(updated_at, started_at, NOW()))) > $2::double precision
          AND NOT (
              stage != 'extraction'
              AND (
                  error_json::text LIKE '%Artifact not found%'
                  OR error_json->>'error_type' = 'FileNotFoundError'
                  OR error_json::text LIKE '%received no chunks%'
              )
          )
        ORDER BY COALESCE(updated_at, started_at, NOW()) ASC
        LIMIT $3::int
        """,
        [max_attempts, retry_cooldown_seconds, batch_size],
        multirows=True,
    )

    if not rows:
        return 0

    events: list[PipelineStageEvent] = []
    event_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            events.append(_event_from_stale_row(row))
            event_rows.append(row)
        except Exception as exc:
            logger.error(
                "Sweeper: cannot requeue failed record run_id=%s stage=%s shard=%s: %s",
                row.get("run_id"),
                row.get("stage"),
                row.get("shard_index"),
                exc,
            )

    published = await _publish_stage_events(events)
    if published > 0:
        await _mark_failed_requeued_rows(event_rows[:published], table_name)
        logger.warning(
            "Sweeper: requeued %s failed stage records (found=%s max_attempts=%s cooldown=%ss batch_size=%s)",
            published,
            len(rows),
            max_attempts,
            retry_cooldown_seconds,
            batch_size,
        )
    return published


async def sweep_orphaned_documents(table_name: str = "lightrag_pipeline_stage_runs") -> int:
    """Find documents that have all extraction shards done/failed but no completed kg_merge record.
    
    This handles the case where entity_extraction shards all completed but the
    join event for kg_merge was never published or processed (e.g., due to worker crash,
    Kafka delivery failure, or pending records stuck in DB).
    """
    db = await _get_db()
    stuck_pending_seconds = int(
        os.getenv("PIPELINE_SWEEPER_PENDING_STUCK_THRESHOLD_SECONDS", "300")
    )
    batch_size = int(os.getenv("PIPELINE_SWEEPER_ORPHANED_BATCH_SIZE", "1000"))
    
    # Find run_ids where:
    # 1. entity_extraction shards exist
    # 2. All entity_extraction shards are completed successfully (done)
    # 3. No running/done kg_merge record exists, and no recent pending record exists
    rows = await db.query(
        f"""
        WITH ee_summary AS (
            SELECT run_id,
                   MAX(workspace_id) as workspace_id,
                   MAX(doc_id) as doc_id,
                   MAX(shard_count) as shard_count,
                   MAX(event_json->>'doc_type') as doc_type,
                   MAX(event_json->'metadata'->>'doc_type') as meta_doc_type,
                   MAX(event_json->'metadata'->>'legal_document_id') as legal_doc_id,
                   MAX(event_json->'metadata'->>'document_name') as document_name,
                   MAX(event_json->'metadata'->>'file_path') as file_path,
                   COUNT(*) as total_shards,
                   COUNT(*) FILTER (WHERE status = 'done') as done_shards,
                   MIN(updated_at) as oldest_update
            FROM {table_name}
            WHERE stage = 'entity_extraction'
            GROUP BY run_id
        )
        SELECT ee.run_id, ee.shard_count, ee.total_shards, ee.done_shards, ee.workspace_id, ee.doc_id,
               ee.doc_type, ee.meta_doc_type, ee.legal_doc_id, ee.document_name, ee.file_path
        FROM ee_summary ee
        WHERE ee.total_shards >= ee.shard_count
          AND ee.done_shards = ee.total_shards
          AND NOT EXISTS (
              SELECT 1 FROM {table_name} km
              WHERE km.run_id = ee.run_id
                AND km.stage = 'kg_merge'
                AND (
                    km.status IN ('running', 'done')
                    OR (km.status = 'pending' AND EXTRACT(EPOCH FROM (NOW() - COALESCE(km.updated_at, km.started_at, NOW()))) < $2::double precision)
                )
          )
          AND EXTRACT(EPOCH FROM (NOW() - ee.oldest_update)) > $1::double precision
        LIMIT $3::int
        """,
        [STUCK_THRESHOLD_SECONDS, stuck_pending_seconds, batch_size],
        multirows=True,
    )
    
    if not rows:
        return 0
    
    count = len(rows)
    logger.warning(
        "Sweeper: found %s documents with completed entity_extraction needing kg_merge recovery", count
    )
    
    run_ids = [row["run_id"] for row in rows]
    artifacts_by_run: dict[str, list[str]] = {}
    
    art_rows = await db.query(
        f"""
        SELECT run_id, artifacts_json
        FROM {table_name}
        WHERE stage = 'entity_extraction'
          AND status = 'done'
          AND run_id = ANY($1::text[])
        """,
        [run_ids],
        multirows=True,
    )
    import json
    for art_row in art_rows or []:
        rid = art_row.get("run_id")
        art_list = art_row.get("artifacts_json") or []
        if isinstance(art_list, str):
            try:
                art_list = json.loads(art_list)
            except Exception:
                art_list = []
        if rid not in artifacts_by_run:
            artifacts_by_run[rid] = []
        for item in art_list:
            if isinstance(item, dict) and item.get("kind") == "extraction" and item.get("uri"):
                artifacts_by_run[rid].append(item["uri"])

    events = []
    event_rows = []
    for row in rows:
        run_id = row["run_id"]
        workspace_id = row["workspace_id"]
        doc_id = row["doc_id"]
        input_uris = artifacts_by_run.get(run_id, [])
        
        raw_doc_type = str(row.get("doc_type") or "").lower()
        meta_doc_type = str(row.get("meta_doc_type") or "").lower()
        legal_doc_id = row.get("legal_doc_id")
        is_legal = (
            raw_doc_type == "legal"
            or meta_doc_type == "legal"
            or legal_doc_id is not None
        )
        canonical_doc_type = "legal" if is_legal else "general"

        meta = {
            "input_artifact_uris": input_uris,
            "doc_type": canonical_doc_type,
        }
        if legal_doc_id is not None:
            meta["legal_document_id"] = legal_doc_id
        if row.get("document_name"):
            meta["document_name"] = row.get("document_name")
        if row.get("file_path"):
            meta["file_path"] = row.get("file_path")

        event = PipelineStageEvent(
            run_id=run_id,
            stage=StageName.KG_MERGE,
            shard_index=0,
            shard_count=1,
            workspace_id=workspace_id,
            doc_id=doc_id,
            doc_type=canonical_doc_type,
            attempt=1,
            metadata=meta,
        )
        events.append(event)
        event_rows.append((run_id, workspace_id, doc_id, event.model_dump_json()))
        
    for run_id, workspace_id, doc_id, event_json_str in event_rows:
        try:
            await db.execute(
                f"""
                INSERT INTO {table_name}
                    (run_id, stage, shard_index, shard_count, workspace_id, doc_id,
                     status, attempt, started_at, updated_at,
                     error_json, artifacts_json, event_json)
                VALUES
                    ($1, 'kg_merge', 0, 1,
                     $2, $3,
                     'pending', 1, NOW(), NOW(),
                     '{{}}'::jsonb,
                     '[]'::jsonb,
                     $4::jsonb)
                ON CONFLICT (run_id, stage, shard_index) DO UPDATE
                    SET status = 'pending',
                        updated_at = NOW(),
                        attempt = EXCLUDED.attempt,
                        event_json = EXCLUDED.event_json
                """,
                [run_id, workspace_id, doc_id, event_json_str],
            )
        except Exception as exc:
            logger.error("Sweeper: failed to insert/update pending kg_merge record for %s: %s", run_id, exc)

    if events:
        try:
            published = await _publish_stage_events(events)
            logger.info("Sweeper: successfully published %s missing kg_merge events", published)
        except Exception as exc:
            logger.error("Sweeper: failed to publish kg_merge events: %s", exc)
    
    return count


async def sweep_orphaned_kg_embedding_documents(
    table_name: str = "lightrag_pipeline_stage_runs",
) -> int:
    """Find documents that completed kg_merge but have no active/pending kg_embedding record.
    
    This handles the case where kg_merge finished successfully but the downstream
    kg_embedding event was lost or dropped.
    """
    db = await _get_db()
    stuck_pending_seconds = int(
        os.getenv("PIPELINE_SWEEPER_PENDING_STUCK_THRESHOLD_SECONDS", "300")
    )
    batch_size = int(os.getenv("PIPELINE_SWEEPER_ORPHANED_BATCH_SIZE", "1000"))

    # Find run_ids where:
    # 1. kg_merge is done and has non-empty artifacts_json
    # 2. No active (running/done) kg_embedding record exists, and no recent pending record exists
    rows = await db.query(
        f"""
        SELECT km.run_id, km.workspace_id, km.doc_id, km.artifacts_json,
               km.event_json->>'doc_type' as doc_type,
               km.event_json->'metadata'->>'doc_type' as meta_doc_type,
               km.event_json->'metadata'->>'legal_document_id' as legal_doc_id,
               km.event_json->'metadata'->>'document_name' as document_name,
               km.event_json->'metadata'->>'file_path' as file_path
        FROM {table_name} km
        WHERE km.stage = 'kg_merge'
          AND km.status = 'done'
          AND jsonb_array_length(km.artifacts_json) > 0
          AND NOT EXISTS (
              SELECT 1 FROM {table_name} ke
              WHERE ke.run_id = km.run_id
                AND ke.stage = 'kg_embedding'
                AND (
                    ke.status IN ('running', 'done')
                    OR (ke.status = 'pending' AND EXTRACT(EPOCH FROM (NOW() - COALESCE(ke.updated_at, ke.started_at, NOW()))) < $2::double precision)
                )
          )
          AND EXTRACT(EPOCH FROM (NOW() - km.updated_at)) > $1::double precision
        ORDER BY km.updated_at ASC
        LIMIT $3::int
        """,
        [STUCK_THRESHOLD_SECONDS, stuck_pending_seconds, batch_size],
        multirows=True,
    )

    if not rows:
        return 0

    count = len(rows)
    logger.warning(
        "Sweeper: found %s documents with completed kg_merge needing kg_embedding recovery", count
    )

    events = []
    event_rows = []
    import json
    for row in rows:
        run_id = row["run_id"]
        workspace_id = row["workspace_id"]
        doc_id = row["doc_id"]
        art_list = row.get("artifacts_json") or []
        if isinstance(art_list, str):
            try:
                art_list = json.loads(art_list)
            except Exception:
                art_list = []

        payload_uri = None
        for item in art_list:
            if isinstance(item, dict) and item.get("kind") == "kg_vector_payload" and item.get("uri"):
                payload_uri = item["uri"]
                break

        if not payload_uri:
            continue

        raw_doc_type = str(row.get("doc_type") or "").lower()
        meta_doc_type = str(row.get("meta_doc_type") or "").lower()
        legal_doc_id = row.get("legal_doc_id")
        is_legal = (
            raw_doc_type == "legal"
            or meta_doc_type == "legal"
            or legal_doc_id is not None
        )
        canonical_doc_type = "legal" if is_legal else "general"

        meta = {
            "kg_vector_payload_artifact_uri": payload_uri,
            "doc_type": canonical_doc_type,
        }
        if legal_doc_id is not None:
            meta["legal_document_id"] = legal_doc_id
        if row.get("document_name"):
            meta["document_name"] = row.get("document_name")
        if row.get("file_path"):
            meta["file_path"] = row.get("file_path")

        event = PipelineStageEvent(
            run_id=run_id,
            stage=StageName.KG_EMBEDDING,
            shard_index=0,
            shard_count=1,
            workspace_id=workspace_id,
            doc_id=doc_id,
            doc_type=canonical_doc_type,
            attempt=1,
            artifact_uri=payload_uri,
            metadata=meta,
        )
        events.append(event)
        event_rows.append((run_id, workspace_id, doc_id, event.model_dump_json()))

    for run_id, workspace_id, doc_id, event_json_str in event_rows:
        try:
            await db.execute(
                f"""
                INSERT INTO {table_name}
                    (run_id, stage, shard_index, shard_count, workspace_id, doc_id,
                     status, attempt, started_at, updated_at,
                     error_json, artifacts_json, event_json)
                VALUES
                    ($1, 'kg_embedding', 0, 1,
                     $2, $3,
                     'pending', 1, NOW(), NOW(),
                     '{{}}'::jsonb,
                     '[]'::jsonb,
                     $4::jsonb)
                ON CONFLICT (run_id, stage, shard_index) DO UPDATE
                    SET status = 'pending',
                        updated_at = NOW(),
                        attempt = EXCLUDED.attempt,
                        event_json = EXCLUDED.event_json
                """,
                [run_id, workspace_id, doc_id, event_json_str],
            )
        except Exception as exc:
            logger.error("Sweeper: failed to insert/update pending kg_embedding record for %s: %s", run_id, exc)

    if events:
        try:
            published = await _publish_stage_events(events)
            logger.info("Sweeper: successfully published %s missing kg_embedding events", published)
        except Exception as exc:
            logger.error("Sweeper: failed to publish kg_embedding events: %s", exc)

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
    logger.info(
        "Sweeper: starting sweep cycle (threshold=%ss chunk_embedding_threshold=%ss)",
        STUCK_THRESHOLD_SECONDS,
        CHUNK_EMBEDDING_STUCK_THRESHOLD_SECONDS,
    )
    
    try:
        await report_pipeline_gaps()
        stale_count = await sweep_stale_running()
        failed_stage_count = await sweep_failed_stage_records()
        orphan_kg_merge_count = await sweep_orphaned_documents()
        orphan_kg_embedding_count = await sweep_orphaned_kg_embedding_documents()
        auto_approved_count = await auto_approve_next_stages()
        
        logger.info(
            "Sweeper: cycle complete. Requeued %s stale running, requeued %s failed records, found %s orphaned kg_merge, found %s orphaned kg_embedding, auto-approved %s paused next-stage records.",
            stale_count,
            failed_stage_count,
            orphan_kg_merge_count,
            orphan_kg_embedding_count,
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
