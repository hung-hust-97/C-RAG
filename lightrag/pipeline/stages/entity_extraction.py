from __future__ import annotations

import asyncio
from collections import Counter
import os
from typing import Any

from lightrag.legal_document.extraction_mode import ExtractionMode
from lightrag.legal_document.extraction_mode import classify_chunk_extraction_mode
from lightrag.pipeline.models import PipelineStageEvent, StageName, StageResult

from .common import (
    filter_chunks,
    get_rag_for_event,
    rows_to_chunks,
    serialize_extraction_results,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


async def run_entity_extraction_stage(
    event: PipelineStageEvent, context: Any
) -> StageResult:
    if not event.artifact_uri:
        raise ValueError("entity_extraction stage requires artifact_uri")

    rag = await get_rag_for_event(context, event, profile=event.stage)
    rows = await context.artifact_store.read_jsonl(event.artifact_uri)
    chunks = rows_to_chunks(rows)
    target_chunk_ids = event.metadata.get("chunk_ids")
    filtered = filter_chunks(chunks, target_chunk_ids)
    if not filtered:
        if rows and target_chunk_ids:
            from lightrag.utils import logger
            logger.warning(
                "[Pipeline] Skipping obsolete entity_extraction shard: none of the %d chunk_ids match current artifact %s %s",
                len(target_chunk_ids),
                event.artifact_uri,
                event.log_context(),
            )
            return StageResult(
                counts={"chunks": 0},
                message="Skipped obsolete shard: chunk_ids not found in current artifact",
            )
        raise ValueError("entity_extraction stage received no chunks")
    chunks = filtered

    doc_types = {chunk.get("doc_type", "general") for chunk in chunks.values()}
    doc_type = "legal" if doc_types == {"legal"} else "general"
    mode_counts: Counter[str] = Counter()
    if doc_type == "legal":
        mode_counts.update(
            classify_chunk_extraction_mode(chunk).value for chunk in chunks.values()
        )

    original_llm_max_async = rag.llm_model_max_async
    original_addon_params = dict(rag.addon_params)
    max_async_override = _env_int("PIPELINE_ENTITY_EXTRACTION_MAX_ASYNC", 54)
    if max_async_override > 0:
        rag.llm_model_max_async = min(rag.llm_model_max_async, max_async_override)
    if event.debug_sync or context.debug_sync:
        rag.llm_model_max_async = 1
    rag.addon_params["entity_extraction_skip_failed_chunks"] = False
    try:
        extraction_results = await rag._process_extract_entities(
            chunks,
            pipeline_status={"latest_message": "", "history_messages": []},
            pipeline_status_lock=asyncio.Lock(),
            doc_type=doc_type,
        )
    finally:
        rag.llm_model_max_async = original_llm_max_async
        rag.addon_params = original_addon_params

    artifact = await context.artifact_store.write_jsonl(
        run_id=event.run_id,
        stage=event.stage_name,
        name=f"{event.doc_id}.extraction.{event.shard_index}.jsonl",
        rows=serialize_extraction_results(extraction_results),
        kind="extraction",
        metadata={
            "doc_id": event.doc_id,
            "workspace_id": event.workspace_id,
            "shard_index": event.shard_index,
        },
    )
    manifest = await context.artifact_store.write_manifest(
        run_id=event.run_id,
        stage=event.stage_name,
        name=f"{event.doc_id}.extraction.{event.shard_index}.manifest.json",
        manifest={
            "stage": event.stage_name.value,
            "doc_id": event.doc_id,
            "workspace_id": event.workspace_id,
            "shard_index": event.shard_index,
            "shard_count": event.shard_count,
            "input_artifact": event.artifact_uri,
            "output_artifact": artifact.model_dump(mode="json"),
            "counts": {
                "chunks": len(chunks),
                "result_batches": len(extraction_results),
                "vector_only_chunks": mode_counts.get(
                    ExtractionMode.VECTOR_ONLY.value, 0
                ),
                "rule_only_chunks": mode_counts.get(
                    ExtractionMode.KG_RULE_ONLY.value, 0
                ),
                "llm_chunks": mode_counts.get(ExtractionMode.KG_LLM.value, 0),
            },
        },
    )

    join_event = event.next_event(
        StageName.KG_MERGE,
        metadata={
            "chunks_artifact_uri": event.metadata.get("chunks_artifact_uri")
            or event.artifact_uri,
        },
    )
    return StageResult(
        output_artifacts=[artifact],
        manifest=manifest,
        counts={
            "chunks": len(chunks),
            "result_batches": len(extraction_results),
            "vector_only_chunks": mode_counts.get(
                ExtractionMode.VECTOR_ONLY.value, 0
            ),
            "rule_only_chunks": mode_counts.get(ExtractionMode.KG_RULE_ONLY.value, 0),
            "llm_chunks": mode_counts.get(ExtractionMode.KG_LLM.value, 0),
        },
        join_next_event=join_event,
        message=f"extracted {len(extraction_results)} chunk result batches",
    )
