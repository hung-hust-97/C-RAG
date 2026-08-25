from __future__ import annotations

import copy
import os
import time
from collections import Counter
from typing import Any, Awaitable, Callable

from lightrag.constants import DEFAULT_LEGAL_ENTITY_TYPES, GRAPH_FIELD_SEP
from lightrag.legal_document.extraction_mode import ExtractionMode
from lightrag.legal_document.extraction_mode import classify_chunk_extraction_mode
from lightrag.legal_document.extraction import extract_legal_cross_chunk_relations
from lightrag.legal_document.hierarchy import append_legal_hierarchy_results
from lightrag.legal_document.rule_extractor import extract_legal_rule_kg
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


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _count_result_items(results: list) -> tuple[int, int]:
    entities = 0
    relations = 0
    for maybe_nodes, maybe_edges in results:
        if isinstance(maybe_nodes, dict):
            entities += len(maybe_nodes)
        elif isinstance(maybe_nodes, list):
            entities += sum(
                1
                for item in maybe_nodes
                if isinstance(item, dict) and item.get("entity_name")
            )

        if isinstance(maybe_edges, dict):
            relations += len(maybe_edges)
        elif isinstance(maybe_edges, list):
            relations += sum(
                1
                for item in maybe_edges
                if isinstance(item, dict) and item.get("src_id") and item.get("tgt_id")
            )
    return entities, relations


def _merge_mode_counts(modes_by_chunk: dict[str, ExtractionMode]) -> Counter:
    counter = Counter(mode.value for mode in modes_by_chunk.values())
    return counter


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
    """Run optimized legal-document KG extraction and append hierarchy edges."""

    started = time.perf_counter()
    legal_config = copy.deepcopy(global_config)
    legal_config.setdefault("addon_params", {})["entity_types"] = (
        DEFAULT_LEGAL_ENTITY_TYPES
    )
    legal_config.setdefault("addon_params", {})["conditional_gleaning"] = True
    doc_id = next(
        (
            chunk.get("full_doc_id")
            for chunk in chunks.values()
            if isinstance(chunk, dict) and chunk.get("full_doc_id")
        ),
        "",
    )
    workspace_id = str(global_config.get("workspace") or "")

    status_by_document_id = await load_legal_document_statuses(
        [
            chunk.get("legal_document_id")
            for chunk in chunks.values()
            if isinstance(chunk, dict)
        ]
    )
    _apply_legal_status_to_chunks(chunks, status_by_document_id)

    modes_by_chunk = {
        chunk_id: classify_chunk_extraction_mode(chunk)
        for chunk_id, chunk in chunks.items()
        if isinstance(chunk, dict)
    }
    mode_counts = _merge_mode_counts(modes_by_chunk)
    kg_chunks = {
        chunk_id: chunk
        for chunk_id, chunk in chunks.items()
        if modes_by_chunk.get(chunk_id) != ExtractionMode.VECTOR_ONLY
    }
    llm_chunks = {
        chunk_id: chunk
        for chunk_id, chunk in kg_chunks.items()
        if modes_by_chunk.get(chunk_id) == ExtractionMode.KG_LLM
    }

    rule_results = []
    for chunk_id, chunk in kg_chunks.items():
        try:
            maybe_nodes, maybe_edges = extract_legal_rule_kg(chunk_id, chunk)
        except Exception as exc:
            logger.warning(
                "[LegalExtraction] rule extractor failed chunk_id=%s: %s",
                chunk_id,
                exc,
            )
            continue
        if maybe_nodes or maybe_edges:
            rule_results.append((maybe_nodes, maybe_edges))

    rule_entities, rule_relations = _count_result_items(rule_results)

    llm_results = []
    llm_seconds = 0.0
    error_count = 0
    if llm_chunks:
        llm_started = time.perf_counter()
        logger.info(
            "Calling standard extract_entities for %s/%s legal chunks with legal prompts "
            "(%s legal entity types)",
            len(llm_chunks),
            len(chunks),
            len(DEFAULT_LEGAL_ENTITY_TYPES),
        )
        try:
            llm_results = await extract_entities_func(
                llm_chunks,
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
        except Exception as exc:
            error_count += 1
            if not _env_bool("LEGAL_EXTRACTION_FALLBACK_TO_RULE", True):
                raise
            logger.warning(
                "[LegalExtraction] LLM extraction failed for doc_id=%s; "
                "continuing with rule results only: %s",
                doc_id,
                exc,
            )
        finally:
            llm_seconds = time.perf_counter() - llm_started

    llm_entities, llm_relations = _count_result_items(llm_results)
    all_results = rule_results + llm_results

    if not all_results:
        logger.info(
            "[LegalExtraction] doc_id=%s workspace_id=%s chunks=%s vector_only=%s "
            "rule_only=%s llm=%s llm_calls=0 rule_entities=0 rule_relations=0 "
            "llm_entities=0 llm_relations=0 duration=%.3fs llm_duration=0.000s errors=%s",
            doc_id,
            workspace_id,
            len(chunks),
            mode_counts.get(ExtractionMode.VECTOR_ONLY.value, 0),
            mode_counts.get(ExtractionMode.KG_RULE_ONLY.value, 0),
            mode_counts.get(ExtractionMode.KG_LLM.value, 0),
            time.perf_counter() - started,
            error_count,
        )
        return []

    if _env_bool("LEGAL_EXTRACTION_ENABLE_CROSS_CHUNK_RELATIONS", False) and llm_chunks:
        cross_chunk_edges = await extract_legal_cross_chunk_relations(
            kg_chunks,
            all_results,
            legal_config,
            parse_extraction_result,
            llm_response_cache,
            text_chunks_storage,
        )
        if cross_chunk_edges:
            all_results.append(({}, cross_chunk_edges))
            logger.info(f"Added {len(cross_chunk_edges)} legal cross-chunk relations")

    _apply_legal_status_to_extraction_results(all_results, chunks)

    duration = time.perf_counter() - started
    logger.info(
        "[LegalExtraction] doc_id=%s workspace_id=%s chunks=%s vector_only=%s "
        "rule_only=%s llm=%s llm_calls=%s rule_entities=%s rule_relations=%s "
        "llm_entities=%s llm_relations=%s duration=%.3fs llm_duration=%.3fs errors=%s",
        doc_id,
        workspace_id,
        len(chunks),
        mode_counts.get(ExtractionMode.VECTOR_ONLY.value, 0),
        mode_counts.get(ExtractionMode.KG_RULE_ONLY.value, 0),
        mode_counts.get(ExtractionMode.KG_LLM.value, 0),
        len(llm_chunks),
        rule_entities,
        rule_relations,
        llm_entities,
        llm_relations,
        duration,
        llm_seconds,
        error_count,
    )
    return await append_legal_hierarchy_results(kg_chunks, all_results)
