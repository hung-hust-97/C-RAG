from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from lightrag.pipeline.models import PipelineStageEvent, StageName, StageResult
from lightrag.utils import logger

from .common import filter_chunks, rows_to_chunks


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _is_transient_vector_upsert_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "timeout",
            "timed out",
            "too many requests",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "connection reset",
            "connection refused",
            "connection aborted",
            "connection error",
            "server disconnected",
            "server closed",
            "remote protocol error",
            "httpx",
            "aiohttp",
            "api",
            "embedding",
            "service not ready",
            "segmentidallocator",
            "getsegmentid failed",
            "failremainrequest",
            "syncsegmentid failed",
            "abnormal",
            "quota exceeded",
            "memory quota",
            "force deny",
            "message size too large",
        )
    )


async def upsert_vectors_with_retry(
    *,
    vector_storage: Any,
    payload: dict[str, Any],
    label: str,
) -> None:
    attempts = max(1, _env_int("PIPELINE_VECTOR_UPSERT_RETRY_ATTEMPTS", 8))
    delay = max(1, _env_int("PIPELINE_VECTOR_UPSERT_RETRY_DELAY_SECONDS", 3))
    max_delay = max(delay, _env_int("PIPELINE_VECTOR_UPSERT_RETRY_MAX_DELAY_SECONDS", 30))

    for attempt in range(1, attempts + 1):
        try:
            await vector_storage.upsert(payload)
            return
        except Exception as exc:
            if attempt >= attempts or not _is_transient_vector_upsert_error(exc):
                raise
            sleep_seconds = min(max_delay, delay * (2 ** (attempt - 1)))
            logger.warning(
                "[Pipeline] Transient vector upsert error for %s; retrying attempt=%s/%s sleep_seconds=%s error=%s",
                label,
                attempt,
                attempts,
                sleep_seconds,
                exc,
            )
            await asyncio.sleep(sleep_seconds)


async def run_chunk_embedding_stage(
    event: PipelineStageEvent, context: Any
) -> StageResult:
    if not event.artifact_uri:
        raise ValueError("chunk_embedding stage requires artifact_uri")

    stage_started = time.perf_counter()
    rag = await context.get_rag(event.workspace_id, profile=event.stage)
    read_started = time.perf_counter()
    rows = await context.artifact_store.read_jsonl(event.artifact_uri)
    read_seconds = time.perf_counter() - read_started
    chunks = rows_to_chunks(rows)
    target_chunk_ids = event.metadata.get("chunk_ids")
    filtered = filter_chunks(chunks, target_chunk_ids)
    if not filtered:
        if rows and target_chunk_ids:
            logger.warning(
                "[Pipeline] Skipping obsolete chunk_embedding shard: none of the %d chunk_ids match current artifact %s %s",
                len(target_chunk_ids),
                event.artifact_uri,
                event.log_context(),
            )
            return StageResult(
                counts={"chunks": 0},
                message="Skipped obsolete shard: chunk_ids not found in current artifact",
            )
        raise ValueError("chunk_embedding stage received no chunks")
    chunks = filtered

    upsert_started = time.perf_counter()
    await upsert_vectors_with_retry(
        vector_storage=rag.chunks_vdb,
        payload=chunks,
        label=f"chunks doc_id={event.doc_id}",
    )
    upsert_seconds = time.perf_counter() - upsert_started
    index_started = time.perf_counter()
    await rag.chunks_vdb.index_done_callback()
    index_seconds = time.perf_counter() - index_started
    total_seconds = time.perf_counter() - stage_started

    logger.info(
        "[Pipeline] Chunk embedding timings doc_id=%s shard=%s/%s rows=%s chunks=%s read=%.3fs upsert=%.3fs index=%.3fs total=%.3fs",
        event.doc_id,
        event.shard_index,
        event.shard_count,
        len(rows),
        len(chunks),
        read_seconds,
        upsert_seconds,
        index_seconds,
        total_seconds,
    )

    finalize_event = event.next_event(
        StageName.FINALIZE,
        metadata={
            "completed_dependency": StageName.CHUNK_EMBEDDING.value,
            "chunk_embedding_shard_index": event.shard_index,
            "chunk_embedding_shard_count": event.shard_count,
        },
    )
    return StageResult(
        counts={"chunks": len(chunks)},
        finalize_event=finalize_event,
        finalize_after_stages=[StageName.CHUNK_EMBEDDING, StageName.KG_EMBEDDING],
        message=f"embedded {len(chunks)} chunks",
    )
