from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from lightrag.operate import merge_nodes_and_edges

from lightrag.pipeline.models import PipelineStageEvent, StageName, StageResult
from lightrag.utils import logger

from .common import (
    build_entity_vector_payload,
    build_relation_vector_payload,
    deserialize_extraction_results,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


async def _read_extraction_results(event: PipelineStageEvent, context: Any) -> list[Any]:
    artifact_uris = event.metadata.get("input_artifact_uris") or []
    if event.artifact_uri:
        artifact_uris.append(event.artifact_uri)
    if not artifact_uris:
        raise ValueError("kg_merge stage requires extraction artifact URIs")

    results = []
    for uri in artifact_uris:
        rows = await context.artifact_store.read_jsonl(uri)
        results.extend(deserialize_extraction_results(rows))
    return results


def _merge_global_config(rag: Any) -> dict[str, Any]:
    return {
        "workspace": rag.workspace,
        "tokenizer": rag.tokenizer,
        "summary_context_size": rag.summary_context_size,
        "summary_max_tokens": rag.summary_max_tokens,
        "force_llm_summary_on_merge": rag.force_llm_summary_on_merge,
        "llm_model_func": rag.llm_model_func,
        "llm_model_max_async": rag.llm_model_max_async,
        "addon_params": rag.addon_params,
        "summary_length_recommended": rag.summary_length_recommended,
        "source_ids_limit_method": rag.source_ids_limit_method,
        "max_source_ids_per_entity": rag.max_source_ids_per_entity,
        "max_source_ids_per_relation": rag.max_source_ids_per_relation,
        "max_file_paths": rag.max_file_paths,
        "file_path_more_placeholder": rag.file_path_more_placeholder,
    }


def _build_vector_payload(
    event: PipelineStageEvent, merge_result: dict[str, list[dict[str, Any]]] | None
) -> list[dict[str, Any]]:
    if not merge_result:
        return []

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    workspace = event.workspace_id

    for node in [
        *merge_result.get("entities", []),
        *merge_result.get("added_entities", []),
    ]:
        if not isinstance(node, dict):
            continue
        entity_name = node.get("entity_name") or node.get("entity_id")
        if not entity_name:
            continue
        vector_id, payload = build_entity_vector_payload(
            str(entity_name), node, workspace
        )
        if vector_id in seen_ids:
            continue
        seen_ids.add(vector_id)
        rows.append({"kind": "entity", "id": vector_id, "data": payload})

    for edge in merge_result.get("relations", []):
        if not isinstance(edge, dict):
            continue
        src_id = edge.get("src_id")
        tgt_id = edge.get("tgt_id")
        if not src_id or not tgt_id:
            continue
        src_id = str(src_id)
        tgt_id = str(tgt_id)
        if not edge.get("directed") and src_id > tgt_id:
            src_id, tgt_id = tgt_id, src_id
        vector_id, payload = build_relation_vector_payload(
            src_id, tgt_id, edge, workspace
        )
        if vector_id in seen_ids:
            continue
        seen_ids.add(vector_id)
        rows.append({"kind": "relation", "id": vector_id, "data": payload})

    logger.info(
        "[Pipeline] KG merge vector payload built doc_id=%s entities=%s relations=%s rows=%s",
        event.doc_id,
        sum(1 for row in rows if row.get("kind") == "entity"),
        sum(1 for row in rows if row.get("kind") == "relation"),
        len(rows),
    )
    return rows


def _shard_rows(rows: list[dict[str, Any]], shard_size: int) -> list[list[dict[str, Any]]]:
    shard_size = max(1, shard_size)
    if not rows:
        return [[]]
    return [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]


async def run_kg_merge_stage(event: PipelineStageEvent, context: Any) -> StageResult:
    stage_started = time.perf_counter()
    rag = await context.get_rag(event.workspace_id, profile=StageName.KG_MERGE.value)

    read_started = time.perf_counter()
    extraction_results = await _read_extraction_results(event, context)
    read_seconds = time.perf_counter() - read_started
    if not extraction_results:
        raise ValueError("kg_merge stage received no extraction results")

    pipeline_status = {"latest_message": "", "history_messages": []}
    pipeline_status_lock = asyncio.Lock()
    original_llm_max_async = rag.llm_model_max_async
    max_async_override = _env_int("PIPELINE_KG_MERGE_MAX_ASYNC", 0)
    if max_async_override > 0:
        rag.llm_model_max_async = min(rag.llm_model_max_async, max_async_override)
    try:
        merge_started = time.perf_counter()
        merge_result = await merge_nodes_and_edges(
            chunk_results=extraction_results,
            knowledge_graph_inst=rag.chunk_entity_relation_graph,
            entity_vdb=None,
            relationships_vdb=None,
            global_config=_merge_global_config(rag),
            full_entities_storage=rag.full_entities,
            full_relations_storage=rag.full_relations,
            doc_id=event.doc_id,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=rag.llm_response_cache,
            entity_chunks_storage=rag.entity_chunks,
            relation_chunks_storage=rag.relation_chunks,
            current_file_number=1,
            total_files=1,
            file_path=event.metadata.get("document_name")
            or event.metadata.get("file_path")
            or event.doc_id,
        )
        merge_seconds = time.perf_counter() - merge_started
    finally:
        rag.llm_model_max_async = original_llm_max_async

    payload_started = time.perf_counter()
    vector_rows = _build_vector_payload(event, merge_result)
    entity_count = sum(1 for row in vector_rows if row.get("kind") == "entity")
    relation_count = sum(1 for row in vector_rows if row.get("kind") == "relation")

    shard_size = max(1, _env_int("PIPELINE_KG_EMBEDDING_SHARD_SIZE", 256))
    row_shards = _shard_rows(vector_rows, shard_size)
    shard_count = len(row_shards)
    payload_artifacts = []
    next_events = []
    for shard_index, shard_rows in enumerate(row_shards):
        shard_entities = sum(1 for row in shard_rows if row.get("kind") == "entity")
        shard_relations = sum(1 for row in shard_rows if row.get("kind") == "relation")
        if shard_count == 1:
            artifact_name = f"{event.doc_id}.kg-vector-payload.jsonl"
        else:
            artifact_name = (
                f"{event.doc_id}.kg-vector-payload."
                f"{shard_index + 1:05d}-of-{shard_count:05d}.jsonl"
            )
        payload_artifact = await context.artifact_store.write_jsonl(
            run_id=event.run_id,
            stage=event.stage_name,
            name=artifact_name,
            rows=shard_rows,
            kind="kg_vector_payload",
            metadata={
                "doc_id": event.doc_id,
                "workspace_id": event.workspace_id,
                "shard_index": shard_index,
                "shard_count": shard_count,
                "entities": shard_entities,
                "relations": shard_relations,
                "vector_rows": len(shard_rows),
            },
        )
        payload_artifacts.append(payload_artifact)
        next_events.append(
            event.next_event(
                StageName.KG_EMBEDDING,
                artifact_uri=payload_artifact.uri,
                shard_index=shard_index,
                shard_count=shard_count,
                counts={
                    "entities": shard_entities,
                    "relations": shard_relations,
                    "vector_rows": len(shard_rows),
                },
                metadata={
                    "kg_embedding_shard_size": shard_size,
                    "kg_vector_payload_artifact_uri": payload_artifact.uri,
                },
            )
        )
    payload_seconds = time.perf_counter() - payload_started
    total_seconds = time.perf_counter() - stage_started

    manifest = await context.artifact_store.write_manifest(
        run_id=event.run_id,
        stage=event.stage_name,
        name=f"{event.doc_id}.kg-merge.manifest.json",
        manifest={
            "stage": event.stage_name.value,
            "doc_id": event.doc_id,
            "workspace_id": event.workspace_id,
            "input_artifact_uris": event.metadata.get("input_artifact_uris", []),
            "output_artifact": payload_artifacts[0].model_dump(mode="json"),
            "output_artifacts": [
                artifact.model_dump(mode="json") for artifact in payload_artifacts
            ],
            "counts": {
                "entities": entity_count,
                "relations": relation_count,
                "vector_rows": len(vector_rows),
                "shards": shard_count,
            },
            "timings_seconds": {
                "read_extraction_artifacts": round(read_seconds, 6),
                "merge_graph": round(merge_seconds, 6),
                "write_vector_payload": round(payload_seconds, 6),
                "total": round(total_seconds, 6),
            },
        },
    )

    logger.info(
        "[Pipeline] KG merge timings doc_id=%s extraction_results=%s entities=%s relations=%s vector_rows=%s shards=%s shard_size=%s read=%.3fs merge=%.3fs payload=%.3fs total=%.3fs",
        event.doc_id,
        len(extraction_results),
        entity_count,
        relation_count,
        len(vector_rows),
        shard_count,
        shard_size,
        read_seconds,
        merge_seconds,
        payload_seconds,
        total_seconds,
    )
    return StageResult(
        output_artifacts=payload_artifacts,
        manifest=manifest,
        counts={
            "entities": entity_count,
            "relations": relation_count,
            "vector_rows": len(vector_rows),
            "shards": shard_count,
        },
        next_events=next_events,
        message=f"merged KG and prepared {len(vector_rows)} vector rows in {shard_count} shards",
    )
