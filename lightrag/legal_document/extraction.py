from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Awaitable, Callable

from lightrag.base import BaseKVStorage
from lightrag.constants import GRAPH_FIELD_SEP, DEFAULT_SUMMARY_LANGUAGE
from lightrag.extraction_utils import should_retry_without_response_format
from lightrag.legal_document.types import (
    LEGAL_EXTRACTION_MAX_DESCRIPTION_CHARS,
    LEGAL_EXTRACTION_MAX_ENTITIES,
    LEGAL_EXTRACTION_MAX_RELATIONS,
    LegalExtractionResult,
)
from lightrag.utils import (
    Tokenizer,
    compute_mdhash_id,
    logger,
    split_string_by_multi_markers,
    update_chunk_cache_list,
    use_llm_func_with_cache,
)


def build_structured_legal_extraction_prompts(
    content: str,
    entity_types: list[str],
    language: str,
) -> tuple[str, str]:
    relation_keywords = (
        "APPLIES_TO, PENALIZED_BY, REGULATED_BY, HAS_OBLIGATION, HAS_RIGHT, "
        "IS_CONDITION_FOR, REPLACES, MODIFIES"
    )
    system_prompt = f"""---Role---
You are a Legal Knowledge Graph Specialist extracting structured legal entities and relationships.

---Instructions---
Extract only facts that are directly supported by the input text.
Use these entity types when possible: {", ".join(entity_types)}. If none applies, use Other.
Relationship keywords should prefer: {relation_keywords}.
Return at most {LEGAL_EXTRACTION_MAX_ENTITIES} entities and {LEGAL_EXTRACTION_MAX_RELATIONS} relationships.
Prioritize legally operative subjects, obligations, rights, penalties, conditions, deadlines, authorities, and actions.
Skip repeated boilerplate, generic terms, and entities that do not participate in a legal relation.
Keep each description under {LEGAL_EXTRACTION_MAX_DESCRIPTION_CHARS} characters.
All legal relationships must be directed when they express regulation, obligation, right, condition, modification, replacement, or penalty.
Write entity names, descriptions, and relationship descriptions in {language}; keep standardized relationship keywords in English.
Do not create entities or relationships that are not grounded in the input text.
"""
    user_prompt = f"""Extract legal entities and directed legal relationships from the input text.

Entity output fields:
- entity_name
- entity_type
- description

Relation output fields:
- src_id
- tgt_id
- keywords
- description
- directed

Input text:
```
{content}
```
"""
    return system_prompt, user_prompt


def build_structured_legal_continue_prompt(content: str, language: str) -> str:
    return f"""Based on the previous structured extraction, return only missed or corrected legal entities and relationships.
Do not repeat valid items from the previous extraction.
Write text fields in {language}; keep relationship keywords in English.

Input text:
```
{content}
```
"""


def _truncate_text_by_tokens(tokenizer: Tokenizer, text: str, max_tokens: int) -> str:
    encoded = tokenizer.encode(text)
    if len(encoded) <= max_tokens:
        return text
    return tokenizer.decode(encoded[:max_tokens]).strip()


def _legal_article_group_key(chunk: dict[str, Any]) -> tuple[str, str] | None:
    context = chunk.get("context", {}) if isinstance(chunk, dict) else {}
    article = (context.get("article") or "").strip()
    if not article:
        return None
    return ((context.get("chapter") or "").strip(), article)


def _collect_llm_entities_by_chunk(
    chunks: dict[str, Any], llm_results: list
) -> dict[str, dict[str, dict]]:
    chunk_entities: dict[str, dict[str, dict]] = defaultdict(dict)
    for maybe_nodes, _maybe_edges in llm_results:
        if not isinstance(maybe_nodes, dict):
            continue
        for entity_name, entity_items in maybe_nodes.items():
            for entity_item in entity_items:
                if not isinstance(entity_item, dict):
                    continue
                source_id = entity_item.get("source_id") or ""
                for chunk_id in split_string_by_multi_markers(
                    source_id, [GRAPH_FIELD_SEP]
                ):
                    chunk_id = chunk_id.strip()
                    if chunk_id in chunks:
                        chunk_entities[chunk_id][entity_name] = entity_item
    return chunk_entities


async def extract_legal_cross_chunk_relations(
    chunks: dict[str, Any],
    llm_results: list,
    global_config: dict[str, Any],
    parse_extraction_result: Callable[..., Awaitable[tuple[dict, dict, dict]]],
    llm_response_cache: BaseKVStorage | None = None,
    text_chunks_storage: BaseKVStorage | None = None,
) -> list[dict]:
    use_llm_func: callable | None = global_config.get("llm_model_func")
    tokenizer: Tokenizer | None = global_config.get("tokenizer")
    if use_llm_func is None or tokenizer is None:
        return []

    chunk_entities = _collect_llm_entities_by_chunk(chunks, llm_results)
    if not chunk_entities:
        return []

    article_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for chunk_id, chunk in chunks.items():
        group_key = _legal_article_group_key(chunk)
        if group_key is not None and chunk_id in chunk_entities:
            article_groups[group_key].append(chunk_id)

    language = global_config.get("addon_params", {}).get(
        "language", DEFAULT_SUMMARY_LANGUAGE
    )
    max_input_tokens = int(global_config.get("max_extract_input_tokens", 20480))
    all_edges: list[dict] = []

    for (chapter, article), group_chunk_ids in article_groups.items():
        group_chunk_ids = sorted(
            group_chunk_ids,
            key=lambda cid: chunks[cid].get("chunk_order_index", 0),
        )
        if len(group_chunk_ids) < 2:
            continue

        entities_by_chunk = {
            chunk_id: chunk_entities.get(chunk_id, {}) for chunk_id in group_chunk_ids
        }
        if sum(1 for entities in entities_by_chunk.values() if entities) < 2:
            continue

        entity_names = sorted(
            {
                entity_name
                for entities in entities_by_chunk.values()
                for entity_name in entities.keys()
            }
        )
        if len(entity_names) < 2:
            continue

        group_source_id = GRAPH_FIELD_SEP.join(group_chunk_ids)
        context_label = " > ".join(part for part in [chapter, article] if part)
        chunk_sections = []
        for chunk_id in group_chunk_ids:
            chunk = chunks[chunk_id]
            chunk_content = _truncate_text_by_tokens(
                tokenizer, chunk.get("content", ""), 500
            )
            chunk_entity_names = ", ".join(sorted(entities_by_chunk[chunk_id].keys()))
            chunk_sections.append(
                f"Chunk ID: {chunk_id}\nEntities in this chunk: {chunk_entity_names}\nContent:\n{chunk_content}"
            )

        system_prompt = f"""You are a Legal Knowledge Graph Specialist.
Infer only cross-chunk legal relationships that are directly supported by the supplied chunks.
You must not create new entities. Relation endpoints must be chosen from the existing entity list.
Use directed legal relationships and keep keywords in English.
Write descriptions in {language}.
"""
        user_prompt = f"""Legal section: {context_label}

Existing entities:
{json.dumps(entity_names, ensure_ascii=False)}

Chunks:
{chr(10).join(chunk_sections)}

Return only relationships connecting existing entities, preferably when the evidence spans multiple chunks.
Each relation must include source_ids containing the chunk IDs that support it.
Use keywords such as APPLIES_TO, PENALIZED_BY, REGULATED_BY, HAS_OBLIGATION, HAS_RIGHT, IS_CONDITION_FOR, REPLACES, MODIFIES.
"""
        prompt_tokens = len(tokenizer.encode(system_prompt + user_prompt))
        if prompt_tokens > max_input_tokens:
            logger.warning(
                f"Skipping legal cross-chunk relation pass for {context_label}: input tokens {prompt_tokens} > {max_input_tokens}"
            )
            continue

        cache_keys_collector: list[str] = []
        doc_ids = sorted(
            {
                str(
                    chunks[chunk_id].get("full_doc_id")
                    or chunks[chunk_id].get("doc_id")
                    or chunks[chunk_id].get("legal_document_id")
                    or ""
                )
                for chunk_id in group_chunk_ids
            }
        )
        group_cache_id = compute_mdhash_id(
            ":".join(
                [
                    ",".join(doc_ids),
                    chapter,
                    article,
                    ",".join(group_chunk_ids),
                    ",".join(entity_names),
                ]
            ),
            prefix="legal-cross-",
            workspace=global_config.get("workspace", "default"),
        )
        try:
            result, timestamp = await use_llm_func_with_cache(
                user_prompt,
                use_llm_func,
                system_prompt=system_prompt,
                llm_response_cache=llm_response_cache,
                cache_type="extract",
                chunk_id=group_chunk_ids[0],
                cache_keys_collector=cache_keys_collector,
                response_format=LegalExtractionResult,
            )
        except Exception as exc:
            if not should_retry_without_response_format(exc):
                raise
            logger.warning(
                f"Structured cross-chunk relation extraction unsupported for {context_label}, falling back to JSON prompt: {exc}"
            )
            result, timestamp = await use_llm_func_with_cache(
                user_prompt,
                use_llm_func,
                system_prompt=system_prompt,
                llm_response_cache=llm_response_cache,
                cache_type="extract",
                chunk_id=group_chunk_ids[0],
                cache_keys_collector=cache_keys_collector,
            )

        _nodes, cross_edges, _parse_stats = await parse_extraction_result(
            result,
            group_cache_id,
            timestamp,
            "legal_cross_chunk",
            force_directed_relations=True,
            return_stats=True,
        )

        allowed_entities = set(entity_names)
        allowed_chunk_ids = set(group_chunk_ids)
        for edge_items in cross_edges.values():
            for edge in edge_items:
                if (
                    edge.get("src_id") not in allowed_entities
                    or edge.get("tgt_id") not in allowed_entities
                ):
                    continue
                edge_source_ids = [
                    source_id
                    for source_id in split_string_by_multi_markers(
                        edge.get("source_id", ""), [GRAPH_FIELD_SEP]
                    )
                    if source_id in allowed_chunk_ids
                ]
                edge["source_id"] = (
                    GRAPH_FIELD_SEP.join(edge_source_ids)
                    if edge_source_ids
                    else group_source_id
                )
                edge["directed"] = True
                all_edges.append(edge)

        if cache_keys_collector and text_chunks_storage is not None:
            for chunk_id in group_chunk_ids:
                await update_chunk_cache_list(
                    chunk_id,
                    text_chunks_storage,
                    cache_keys_collector,
                    "legal_cross_chunk_relation",
                )

    return all_edges
