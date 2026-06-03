from __future__ import annotations

from collections import defaultdict

from lightrag.base import BaseKVStorage
from lightrag.constants import GRAPH_FIELD_SEP, DEFAULT_ENTITY_NAME_MAX_LENGTH
from lightrag.extraction_utils import (
    empty_extraction_parse_stats,
    json_loads_best_effort,
    looks_like_json_payload,
)
from lightrag.prompt import PROMPTS
from lightrag.utils import (
    fix_tuple_delimiter_corruption,
    is_float_regex,
    logger,
    sanitize_and_normalize_extracted_text,
    split_string_by_multi_markers,
)


def truncate_entity_identifier(
    identifier: str, limit: int, chunk_key: str, identifier_role: str
) -> str:
    """Truncate entity identifiers that exceed the configured length limit."""

    if len(identifier) <= limit:
        return identifier

    display_value = identifier[:limit]
    preview = identifier[:20]
    logger.warning(
        "%s: %s len %d > %d chars (Name: '%s...')",
        chunk_key,
        identifier_role,
        len(identifier),
        limit,
        preview,
    )
    return display_value


async def handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    if len(record_attributes) != 4 or "entity" not in record_attributes[0]:
        if len(record_attributes) > 1 and "entity" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/4 fields on ENTITY `{record_attributes[1]}` @ `{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        entity_name = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )

        if not entity_name or not entity_name.strip() or len(entity_name.strip()) < 2:
            logger.info(
                f"Empty or single-character entity name found after sanitization, ignoring. Original: '{record_attributes[1]}'"
            )
            return None

        entity_type = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )
        if any(char in entity_type for char in ["<", ">", "|", "#", ";"]):
            cleaned_entity_type = "".join(
                char
                for char in entity_type
                if char.isalnum() or char in {" ", "_", "-", ","}
            ).strip()
            if cleaned_entity_type:
                entity_type = cleaned_entity_type

        if not entity_type.strip() or any(
            char in entity_type for char in ["'", "(", ")", "<", ">", "|", "/", "\\"]
        ):
            logger.warning(
                f"Entity extraction error: invalid entity type in: {record_attributes}"
            )
            return None

        if "," in entity_type:
            original = entity_type
            tokens = [t.strip() for t in entity_type.split(",")]
            non_empty = [t for t in tokens if t]
            if not non_empty:
                logger.warning(
                    f"Entity extraction error: all tokens empty after comma-split: '{original}'"
                )
                return None
            entity_type = non_empty[0]
            logger.warning(
                f"Entity type contains comma, taking first non-empty token: '{original}' -> '{entity_type}'"
            )

        entity_type = entity_type.replace(" ", "").lower()
        entity_description = sanitize_and_normalize_extracted_text(record_attributes[3])

        if not entity_description.strip():
            logger.warning(
                f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
            )
            return None

        return dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.error(
            f"Entity extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.error(
            f"Entity extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
):
    if len(record_attributes) != 5 or "relation" not in record_attributes[0]:
        if len(record_attributes) > 1 and "relation" in record_attributes[0]:
            logger.warning(
                f"{chunk_key}: LLM output format error; found {len(record_attributes)}/5 fields on RELATION `{record_attributes[1]}`~`{record_attributes[2] if len(record_attributes) > 2 else 'N/A'}`"
            )
            logger.debug(record_attributes)
        return None

    try:
        source = sanitize_and_normalize_extracted_text(
            record_attributes[1], remove_inner_quotes=True
        )
        target = sanitize_and_normalize_extracted_text(
            record_attributes[2], remove_inner_quotes=True
        )

        if not source:
            logger.info(
                f"Empty source entity found after sanitization. Original: '{record_attributes[1]}'"
            )
            return None

        if not target:
            logger.info(
                f"Empty target entity found after sanitization. Original: '{record_attributes[2]}'"
            )
            return None

        if source == target:
            logger.debug(
                f"Relationship source and target are the same in: {record_attributes}"
            )
            return None

        edge_keywords = sanitize_and_normalize_extracted_text(
            record_attributes[3], remove_inner_quotes=True
        )
        edge_keywords = edge_keywords.replace("，", ",")

        edge_description = sanitize_and_normalize_extracted_text(record_attributes[4])
        if not edge_description.strip():
            logger.warning(
                f"Relationship extraction error: empty description for relation '{source}'~'{target}' in chunk '{chunk_key}'"
            )
            return None

        weight = (
            float(record_attributes[-1].strip('"').strip("'"))
            if is_float_regex(record_attributes[-1].strip('"').strip("'"))
            else 1.0
        )

        return dict(
            src_id=source,
            tgt_id=target,
            weight=weight,
            description=edge_description,
            keywords=edge_keywords,
            source_id=chunk_key,
            file_path=file_path,
            timestamp=timestamp,
        )

    except ValueError as e:
        logger.warning(
            f"Relationship extraction failed due to encoding issues in chunk {chunk_key}: {e}"
        )
        return None
    except Exception as e:
        logger.warning(
            f"Relationship extraction failed with unexpected error in chunk {chunk_key}: {e}"
        )
        return None


async def process_extraction_result(
    result: str,
    chunk_key: str,
    timestamp: int,
    file_path: str = "unknown_source",
    tuple_delimiter: str = "<|#|>",
    completion_delimiter: str = "<|COMPLETE|>",
    force_directed_relations: bool = False,
    return_stats: bool = False,
) -> tuple[dict, dict] | tuple[dict, dict, dict[str, int]]:
    """Process one LLM extraction response in JSON or delimiter format."""

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    stats = empty_extraction_parse_stats()

    if looks_like_json_payload(result):
        try:
            payload = json_loads_best_effort(result)
            if isinstance(payload, list):
                payload = {"relations": payload}
            if isinstance(payload, dict):
                json_payload_seen = any(
                    key in payload
                    for key in ("entities", "relations", "relationships", "edges")
                )
                for entity_item in payload.get("entities", []) or []:
                    if not isinstance(entity_item, dict):
                        stats["dropped_records"] += 1
                        continue
                    record_attributes = [
                        "entity",
                        entity_item.get("entity_name") or entity_item.get("name") or "",
                        entity_item.get("entity_type") or entity_item.get("type") or "",
                        entity_item.get("description")
                        or entity_item.get("entity_description")
                        or "",
                    ]
                    entity_data = await handle_single_entity_extraction(
                        record_attributes, chunk_key, timestamp, file_path
                    )
                    if entity_data is None:
                        stats["dropped_records"] += 1
                        continue
                    truncated_name = truncate_entity_identifier(
                        entity_data["entity_name"],
                        DEFAULT_ENTITY_NAME_MAX_LENGTH,
                        chunk_key,
                        "Entity name",
                    )
                    entity_data["entity_name"] = truncated_name
                    maybe_nodes[truncated_name].append(entity_data)
                    stats["json_records"] += 1

                relation_items = (
                    payload.get("relations")
                    or payload.get("relationships")
                    or payload.get("edges")
                    or []
                )
                for relation_item in relation_items:
                    if not isinstance(relation_item, dict):
                        stats["dropped_records"] += 1
                        continue
                    record_attributes = [
                        "relation",
                        relation_item.get("src_id")
                        or relation_item.get("source_entity")
                        or relation_item.get("source")
                        or "",
                        relation_item.get("tgt_id")
                        or relation_item.get("target_entity")
                        or relation_item.get("target")
                        or "",
                        relation_item.get("keywords")
                        or relation_item.get("relationship_keywords")
                        or "",
                        relation_item.get("description")
                        or relation_item.get("relationship_description")
                        or "",
                    ]
                    relationship_data = await handle_single_relationship_extraction(
                        record_attributes, chunk_key, timestamp, file_path
                    )
                    if relationship_data is None:
                        stats["dropped_records"] += 1
                        continue

                    source_ids = relation_item.get("source_ids")
                    if source_ids is None:
                        source_ids = relation_item.get("source_id")
                    if isinstance(source_ids, str):
                        source_ids = [
                            source_id.strip()
                            for source_id in split_string_by_multi_markers(
                                source_ids, [GRAPH_FIELD_SEP, ","]
                            )
                            if source_id.strip()
                        ]
                    if isinstance(source_ids, list):
                        valid_source_ids = [
                            str(source_id).strip()
                            for source_id in source_ids
                            if str(source_id).strip()
                        ]
                        if valid_source_ids:
                            relationship_data["source_id"] = GRAPH_FIELD_SEP.join(
                                valid_source_ids
                            )

                    if force_directed_relations or relation_item.get("directed"):
                        relationship_data["directed"] = True

                    truncated_source = truncate_entity_identifier(
                        relationship_data["src_id"],
                        DEFAULT_ENTITY_NAME_MAX_LENGTH,
                        chunk_key,
                        "Relation entity",
                    )
                    truncated_target = truncate_entity_identifier(
                        relationship_data["tgt_id"],
                        DEFAULT_ENTITY_NAME_MAX_LENGTH,
                        chunk_key,
                        "Relation entity",
                    )
                    relationship_data["src_id"] = truncated_source
                    relationship_data["tgt_id"] = truncated_target
                    maybe_edges[(truncated_source, truncated_target)].append(
                        relationship_data
                    )
                    stats["json_records"] += 1

                if json_payload_seen:
                    if stats["dropped_records"] or stats["parse_errors"]:
                        logger.info(
                            "%s: extraction parse stats json_records=%d delimiter_records=%d salvaged_records=%d dropped_records=%d parse_errors=%d",
                            chunk_key,
                            stats["json_records"],
                            stats["delimiter_records"],
                            stats["salvaged_records"],
                            stats["dropped_records"],
                            stats["parse_errors"],
                        )
                    result_tuple = (dict(maybe_nodes), dict(maybe_edges))
                    return (*result_tuple, stats) if return_stats else result_tuple
        except Exception as e:
            stats["parse_errors"] += 1
            logger.warning(f"{chunk_key}: failed to parse structured extraction: {e}")

    if completion_delimiter not in result:
        logger.warning(
            f"{chunk_key}: Complete delimiter can not be found in extraction result"
        )

    records = split_string_by_multi_markers(
        result,
        ["\n", completion_delimiter, completion_delimiter.lower()],
    )

    fixed_records = []
    for record in records:
        record = record.strip()
        if record is None:
            continue
        entity_records = split_string_by_multi_markers(
            record, [f"{tuple_delimiter}entity{tuple_delimiter}"]
        )
        for entity_record in entity_records:
            if not entity_record.startswith("entity") and not entity_record.startswith(
                "relation"
            ):
                entity_record = f"entity<|{entity_record}"
            entity_relation_records = split_string_by_multi_markers(
                entity_record,
                [
                    f"{tuple_delimiter}relationship{tuple_delimiter}",
                    f"{tuple_delimiter}relation{tuple_delimiter}",
                ],
            )
            for entity_relation_record in entity_relation_records:
                if not entity_relation_record.startswith(
                    "entity"
                ) and not entity_relation_record.startswith("relation"):
                    entity_relation_record = (
                        f"relation{tuple_delimiter}{entity_relation_record}"
                    )
                fixed_records = fixed_records + [entity_relation_record]

    if len(fixed_records) != len(records):
        logger.warning(
            f"{chunk_key}: LLM output format error; find LLM use {tuple_delimiter} as record separators instead new-line"
        )

    for record in fixed_records:
        record = record.strip()
        if record is None:
            continue

        delimiter_core = tuple_delimiter[2:-2]
        record = fix_tuple_delimiter_corruption(record, delimiter_core, tuple_delimiter)
        if delimiter_core != delimiter_core.lower():
            delimiter_core = delimiter_core.lower()
            record = fix_tuple_delimiter_corruption(
                record, delimiter_core, tuple_delimiter
            )

        record_attributes = split_string_by_multi_markers(record, [tuple_delimiter])

        if record_attributes and "entity" in record_attributes[0]:
            if len(record_attributes) > 4:
                record_attributes = record_attributes[:3] + [
                    tuple_delimiter.join(record_attributes[3:])
                ]
                stats["salvaged_records"] += 1
            elif len(record_attributes) < 4:
                stats["dropped_records"] += 1
                logger.warning(
                    f"{chunk_key}: LLM output format error; found {len(record_attributes)}/4 fields on ENTITY"
                )
                continue
        elif record_attributes and (
            "relation" in record_attributes[0]
            or "relationship" in record_attributes[0]
        ):
            if len(record_attributes) > 5:
                record_attributes = record_attributes[:4] + [
                    tuple_delimiter.join(record_attributes[4:])
                ]
                stats["salvaged_records"] += 1
            elif len(record_attributes) == 4:
                record_attributes = record_attributes[:3] + [
                    "RELATED_TO",
                    record_attributes[3],
                ]
                stats["salvaged_records"] += 1
            elif len(record_attributes) < 5:
                stats["dropped_records"] += 1
                logger.warning(
                    f"{chunk_key}: LLM output format error; found {len(record_attributes)}/5 fields on RELATION"
                )
                continue

        entity_data = await handle_single_entity_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if entity_data is not None:
            truncated_name = truncate_entity_identifier(
                entity_data["entity_name"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Entity name",
            )
            entity_data["entity_name"] = truncated_name
            maybe_nodes[truncated_name].append(entity_data)
            stats["delimiter_records"] += 1
            continue

        relationship_data = await handle_single_relationship_extraction(
            record_attributes, chunk_key, timestamp, file_path
        )
        if relationship_data is not None:
            if force_directed_relations:
                relationship_data["directed"] = True
            truncated_source = truncate_entity_identifier(
                relationship_data["src_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            truncated_target = truncate_entity_identifier(
                relationship_data["tgt_id"],
                DEFAULT_ENTITY_NAME_MAX_LENGTH,
                chunk_key,
                "Relation entity",
            )
            relationship_data["src_id"] = truncated_source
            relationship_data["tgt_id"] = truncated_target
            maybe_edges[(truncated_source, truncated_target)].append(relationship_data)
            stats["delimiter_records"] += 1
            continue

        if record:
            stats["dropped_records"] += 1

    if stats["salvaged_records"] or stats["dropped_records"] or stats["parse_errors"]:
        logger.info(
            "%s: extraction parse stats json_records=%d delimiter_records=%d salvaged_records=%d dropped_records=%d parse_errors=%d",
            chunk_key,
            stats["json_records"],
            stats["delimiter_records"],
            stats["salvaged_records"],
            stats["dropped_records"],
            stats["parse_errors"],
        )

    result_tuple = (dict(maybe_nodes), dict(maybe_edges))
    return (*result_tuple, stats) if return_stats else result_tuple


async def rebuild_from_extraction_result(
    text_chunks_storage: BaseKVStorage,
    extraction_result: str,
    chunk_id: str,
    timestamp: int,
) -> tuple[dict, dict]:
    """Parse cached extraction results using the shared extraction parser."""

    chunk_data = await text_chunks_storage.get_by_id(chunk_id)
    file_path = (
        chunk_data.get("file_path", "unknown_source")
        if chunk_data
        else "unknown_source"
    )

    return await process_extraction_result(
        extraction_result,
        chunk_id,
        timestamp,
        file_path,
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    )
