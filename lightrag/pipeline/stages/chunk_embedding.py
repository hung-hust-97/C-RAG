from __future__ import annotations

import asyncio
import os
from typing import Any

from lightrag.pipeline.models import PipelineStageEvent, StageName, StageResult
from lightrag.utils import logger

from .common import filter_chunks, rows_to_chunks


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _is_transient_milvus_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "service not ready",
            "segmentidallocator",
            "getsegmentid failed",
            "failremainrequest",
            "syncsegmentid failed",
            "abnormal",
        )
    )


async def upsert_vectors_with_retry(
    *,
    vector_storage: Any,
    payload: dict[str, Any],
    label: str,
) -> None:
    attempts = max(1, _env_int("PIPELINE_VECTOR_UPSERT_RETRY_ATTEMPTS", 5))
    delay = max(1, _env_int("PIPELINE_VECTOR_UPSERT_RETRY_DELAY_SECONDS", 5))

    for attempt in range(1, attempts + 1):
        try:
            await vector_storage.upsert(payload)
            return
        except Exception as exc:
            if attempt >= attempts or not _is_transient_milvus_error(exc):
                raise
            sleep_seconds = delay * attempt
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

    rag = await context.get_rag(event.workspace_id, profile=event.stage)
    rows = await context.artifact_store.read_jsonl(event.artifact_uri)
    chunks = rows_to_chunks(rows)
    chunks = filter_chunks(chunks, event.metadata.get("chunk_ids"))
    if not chunks:
        raise ValueError("chunk_embedding stage received no chunks")

    await upsert_vectors_with_retry(
        vector_storage=rag.chunks_vdb,
        payload=chunks,
        label=f"chunks doc_id={event.doc_id}",
    )
    await rag.chunks_vdb.index_done_callback()

    finalize_event = event.next_event(
        StageName.FINALIZE,
        metadata={"completed_dependency": StageName.CHUNK_EMBEDDING.value},
    )
    return StageResult(
        counts={"chunks": len(chunks)},
        finalize_event=finalize_event,
        finalize_after_stages=[StageName.CHUNK_EMBEDDING, StageName.KG_EMBEDDING],
        message=f"embedded {len(chunks)} chunks",
    )
