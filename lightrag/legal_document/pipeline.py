from __future__ import annotations

import copy
from typing import Any, Awaitable, Callable

from lightrag.constants import DEFAULT_LEGAL_ENTITY_TYPES, GRAPH_FIELD_SEP
from lightrag.legal_document.extraction import extract_legal_cross_chunk_relations
from lightrag.legal_document.hierarchy import append_legal_hierarchy_results
from lightrag.legal_document.status import load_legal_document_statuses
from lightrag.legal_document.types import LegalExtractionResult
from lightrag.utils import logger


def _chunk_ids_from_source(source_id: Any) -> list[str]:
    if not source_id:
        return []
    return [
        chunk_id.strip()
        for chunk_id in str(source_id).split(GRAPH_FIELD_SEP)
        if chunk_id.strip()
    ]


def _apply_legal_status_to_chunks(
    chunks: dict[str, Any],
    status_by_document_id: dict[str, dict[str, Any]],
) -> None:
    for chunk in chunks.values():
        if not isinstance(chunk, dict):
            continue
        metadata = status_by_document_id.get(str(chunk.get("legal_document_id") or ""))
        if not metadata:
            continue
        chunk.update(metadata)
        context = chunk.setdefault("context", {})
        if isinstance(context, dict):
            context.update(
                {
                    key: metadata[key]
                    for key in (
                        "legal_status",
                        "legal_status_code",
                        "legal_status_priority",
                    )
                    if key in metadata
                }
            )


def _apply_legal_status_to_extraction_results(
    llm_results: list,
    chunks: dict[str, Any],
) -> None:
    for maybe_nodes, maybe_edges in llm_results:
        if isinstance(maybe_nodes, dict):
            for entity_items in maybe_nodes.values():
                for entity_item in entity_items:
                    _apply_legal_status_to_item(entity_item, chunks)
        elif isinstance(maybe_nodes, list):
            for entity_item in maybe_nodes:
                _apply_legal_status_to_item(entity_item, chunks)

        if isinstance(maybe_edges, dict):
            for edge_items in maybe_edges.values():
                for edge_item in edge_items:
                    _apply_legal_status_to_item(edge_item, chunks)
        elif isinstance(maybe_edges, list):
            for edge_item in maybe_edges:
                _apply_legal_status_to_item(edge_item, chunks)


def _apply_legal_status_to_item(item: Any, chunks: dict[str, Any]) -> None:
    if not isinstance(item, dict):
        return

    status_records = []
    legal_document_ids = []
    for chunk_id in _chunk_ids_from_source(item.get("source_id")):
        chunk = chunks.get(chunk_id)
        if not isinstance(chunk, dict):
            continue
        if chunk.get("legal_document_id"):
            legal_document_ids.append(str(chunk["legal_document_id"]))
        if chunk.get("legal_status"):
            status_records.append(chunk)

    if not status_records:
        return

    from lightrag.legal_document.status import best_legal_status_metadata

    metadata = best_legal_status_metadata(status_records)
    if metadata:
        if legal_document_ids:
            metadata["legal_document_ids"] = GRAPH_FIELD_SEP.join(
                dict.fromkeys(legal_document_ids)
            )
        item.update(metadata)


async def extract_legal_entities(
    chunks: dict[str, Any],
    global_config: dict[str, Any],
    pipeline_status: dict | None,
    pipeline_status_lock: Any,
    llm_response_cache: Any,
    text_chunks_storage: Any,
    extract_entities_func: Callable[..., Awaitable[list]],
    parse_extraction_result: Callable[..., Awaitable[tuple[dict, dict, dict]]],
) -> list:
    """Run legal-document KG extraction and append legal hierarchy edges."""

    legal_config = copy.deepcopy(global_config)
    legal_config.setdefault("addon_params", {})["entity_types"] = (
        DEFAULT_LEGAL_ENTITY_TYPES
    )
    legal_config.setdefault("addon_params", {})["conditional_gleaning"] = True
    status_by_document_id = await load_legal_document_statuses(
        [
            chunk.get("legal_document_id")
            for chunk in chunks.values()
            if isinstance(chunk, dict)
        ]
    )
    _apply_legal_status_to_chunks(chunks, status_by_document_id)

    logger.info(
        "Calling standard extract_entities for domain entities with legal prompts "
        f"({len(DEFAULT_LEGAL_ENTITY_TYPES)} legal entity types)"
    )
    llm_results = await extract_entities_func(
        chunks,
        legal_config,
        pipeline_status,
        pipeline_status_lock,
        llm_response_cache,
        text_chunks_storage,
        prompt_prefix="legal_entity_extraction",
        structured_response_format=LegalExtractionResult,
        force_directed_relations=True,
        conditional_gleaning=True,
    )

    cross_chunk_edges = await extract_legal_cross_chunk_relations(
        chunks,
        llm_results,
        legal_config,
        parse_extraction_result,
        llm_response_cache,
        text_chunks_storage,
    )
    if cross_chunk_edges:
        llm_results.append(({}, cross_chunk_edges))
        logger.info(f"Added {len(cross_chunk_edges)} legal cross-chunk relations")

    _apply_legal_status_to_extraction_results(llm_results, chunks)

    return await append_legal_hierarchy_results(chunks, llm_results)
