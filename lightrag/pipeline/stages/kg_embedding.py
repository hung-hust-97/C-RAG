from __future__ import annotations

from typing import Any

from lightrag.pipeline.models import PipelineStageEvent, StageName, StageResult


async def run_kg_embedding_stage(event: PipelineStageEvent, context: Any) -> StageResult:
    if not event.artifact_uri:
        raise ValueError("kg_embedding stage requires artifact_uri")

    rag = await context.get_rag(event.workspace_id, profile=event.stage)
    rows = await context.artifact_store.read_jsonl(event.artifact_uri)
    entity_payload = {
        row["id"]: row["data"] for row in rows if row.get("kind") == "entity"
    }
    relation_payload = {
        row["id"]: row["data"] for row in rows if row.get("kind") == "relation"
    }

    if entity_payload:
        await rag.entities_vdb.upsert(entity_payload)
        await rag.entities_vdb.index_done_callback()
    if relation_payload:
        await rag.relationships_vdb.upsert(relation_payload)
        await rag.relationships_vdb.index_done_callback()

    finalize_event = event.next_event(
        StageName.FINALIZE,
        metadata={"completed_dependency": StageName.KG_EMBEDDING.value},
    )
    return StageResult(
        counts={
            "entities": len(entity_payload),
            "relations": len(relation_payload),
        },
        finalize_event=finalize_event,
        finalize_after_stages=[StageName.CHUNK_EMBEDDING, StageName.KG_EMBEDDING],
        message=(
            f"embedded {len(entity_payload)} entities and "
            f"{len(relation_payload)} relations"
        ),
    )
