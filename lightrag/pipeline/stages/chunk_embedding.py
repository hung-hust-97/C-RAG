from __future__ import annotations

from typing import Any

from lightrag.pipeline.models import PipelineStageEvent, StageName, StageResult

from .common import filter_chunks, rows_to_chunks


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

    await rag.chunks_vdb.upsert(chunks)
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
