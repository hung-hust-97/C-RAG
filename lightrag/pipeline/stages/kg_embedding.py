from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from lightrag.pipeline.models import PipelineStageEvent, StageName, StageResult
from lightrag.utils import logger

from .chunk_embedding import upsert_vectors_with_retry


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.lower() in {"1", "true", "yes", "on"}


async def run_kg_embedding_stage(event: PipelineStageEvent, context: Any) -> StageResult:
    if not event.artifact_uri:
        raise ValueError("kg_embedding stage requires artifact_uri")

    stage_started = time.perf_counter()
    rag = await context.get_rag(event.workspace_id, profile=event.stage)
    read_started = time.perf_counter()
    rows = await context.artifact_store.read_jsonl(event.artifact_uri)
    read_seconds = time.perf_counter() - read_started
    entity_payload = {
        row["id"]: row["data"] for row in rows if row.get("kind") == "entity"
    }
    relation_payload = {
        row["id"]: row["data"] for row in rows if row.get("kind") == "relation"
    }

    entity_seconds = 0.0
    relation_seconds = 0.0

    async def _upsert_entities() -> None:
        nonlocal entity_seconds
        if not entity_payload:
            return
        started = time.perf_counter()
        await upsert_vectors_with_retry(
            vector_storage=rag.entities_vdb,
            payload=entity_payload,
            label=(
                f"entities doc_id={event.doc_id} "
                f"shard={event.shard_index}/{event.shard_count}"
            ),
        )
        await rag.entities_vdb.index_done_callback()
        entity_seconds = time.perf_counter() - started

    async def _upsert_relations() -> None:
        nonlocal relation_seconds
        if not relation_payload:
            return
        started = time.perf_counter()
        await upsert_vectors_with_retry(
            vector_storage=rag.relationships_vdb,
            payload=relation_payload,
            label=(
                f"relations doc_id={event.doc_id} "
                f"shard={event.shard_index}/{event.shard_count}"
            ),
        )
        await rag.relationships_vdb.index_done_callback()
        relation_seconds = time.perf_counter() - started

    if (
        _env_bool("PIPELINE_KG_EMBEDDING_PARALLEL_KINDS", False)
        and entity_payload
        and relation_payload
    ):
        await asyncio.gather(_upsert_entities(), _upsert_relations())
    else:
        await _upsert_entities()
        await _upsert_relations()

    total_seconds = time.perf_counter() - stage_started
    logger.info(
        "[Pipeline] KG embedding timings doc_id=%s shard=%s/%s rows=%s entities=%s relations=%s read=%.3fs entity_upsert=%.3fs relation_upsert=%.3fs total=%.3fs",
        event.doc_id,
        event.shard_index,
        event.shard_count,
        len(rows),
        len(entity_payload),
        len(relation_payload),
        read_seconds,
        entity_seconds,
        relation_seconds,
        total_seconds,
    )

    finalize_event = event.next_event(
        StageName.FINALIZE,
        metadata={
            "completed_dependency": StageName.KG_EMBEDDING.value,
            "kg_embedding_shard_index": event.shard_index,
            "kg_embedding_shard_count": event.shard_count,
        },
    )
    return StageResult(
        counts={
            "entities": len(entity_payload),
            "relations": len(relation_payload),
            "vector_rows": len(rows),
        },
        finalize_event=finalize_event,
        finalize_after_stages=[StageName.CHUNK_EMBEDDING, StageName.KG_EMBEDDING],
        message=(
            f"embedded {len(entity_payload)} entities and "
            f"{len(relation_payload)} relations from shard "
            f"{event.shard_index + 1}/{event.shard_count}"
        ),
    )
