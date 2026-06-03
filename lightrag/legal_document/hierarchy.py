from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import logger


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        value = row[key]
    except Exception:
        value = getattr(row, key, default)
    return default if value is None else value


def _coerce_legal_document_id(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    clean_value = str(value)
    if clean_value.startswith("legal-"):
        clean_value = clean_value[len("legal-") :]
    if not clean_value.isdigit():
        return None
    return int(clean_value)


def _node(
    name: str,
    entity_type: str,
    description: str,
    source_id: str = "",
    **extra: Any,
) -> dict[str, Any]:
    data = {
        "entity_name": name,
        "entity_type": entity_type,
        "description": description,
        "source_id": source_id,
    }
    data.update(extra)
    return data


def _edge(
    src_id: str,
    tgt_id: str,
    keywords: str,
    description: str,
    source_id: str = "",
    **extra: Any,
) -> dict[str, Any]:
    data = {
        "src_id": src_id,
        "tgt_id": tgt_id,
        "keywords": keywords,
        "weight": 1.0,
        "description": description,
        "source_id": source_id,
        "directed": True,
    }
    data.update(extra)
    return data


def _legal_document_id_from_chunks(chunks: dict[str, Any]) -> Any:
    for chunk in chunks.values():
        if isinstance(chunk, dict) and chunk.get("legal_document_id"):
            return chunk["legal_document_id"]
    return None


async def _load_legal_document_metadata(
    legal_document_id: Any,
) -> tuple[str, str, list[Any]]:
    from lightrag.kg.postgres_impl import ClientManager

    db = await ClientManager.get_client()
    if db is None:
        return "", "", []

    tvpl_key = ""
    legal_status = ""
    related_docs: list[Any] = []

    async def _query_db(connection):
        nonlocal tvpl_key, legal_status, related_docs

        row = await connection.fetchrow(
            """
            SELECT tvpl_key, status::text AS status
            FROM tbl_legal_document
            WHERE id = $1
            """,
            legal_document_id,
        )
        if row:
            tvpl_key = _row_value(row, "tvpl_key", "")
            legal_status = _row_value(row, "status", "")

        related_docs = await connection.fetch(
            """SELECT r.main_doc_id, r.related_doc_id, r.relation_type,
                      m.tvpl_key as main_tvpl_key, d.tvpl_key as related_tvpl_key
               FROM tbl_relate_legal_document r
               LEFT JOIN tbl_legal_document m ON r.main_doc_id = m.id
               LEFT JOIN tbl_legal_document d ON r.related_doc_id = d.id
               WHERE r.main_doc_id = $1 OR r.related_doc_id = $1""",
            legal_document_id,
        )

    await db._run_with_retry(_query_db)
    return tvpl_key, legal_status, list(related_docs)


def _append_unique_structure_node(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    created_nodes: set[str],
    parent: str,
    node_name: str,
    entity_type: str,
    relation_type: str,
    description: str,
    source_id: str,
    extra: dict[str, Any] | None = None,
) -> None:
    if node_name in created_nodes:
        return

    extra = extra or {}
    nodes.append(_node(node_name, entity_type, description, source_id, **extra))
    created_nodes.add(node_name)
    edges.append(
        _edge(
            parent,
            node_name,
            relation_type,
            f"{parent} contains {node_name}",
            source_id,
            **extra,
        )
    )


def _iter_extracted_entities(llm_results: list) -> Iterable[tuple[str, dict[str, Any]]]:
    for maybe_nodes, _maybe_edges in llm_results:
        if isinstance(maybe_nodes, dict):
            for entity_name, entity_items in maybe_nodes.items():
                items = entity_items if isinstance(entity_items, list) else [entity_items]
                for item in items:
                    if isinstance(item, dict):
                        yield entity_name, item
        elif isinstance(maybe_nodes, list):
            for item in maybe_nodes:
                if isinstance(item, dict) and item.get("entity_name"):
                    yield item["entity_name"], item


def _level_name(parent_node: str) -> str:
    if "#" not in parent_node:
        return "document root"
    parts = parent_node.split("#")
    if len(parts) == 5:
        return f"point {parts[4]}"
    if len(parts) == 4:
        return f"clause {parts[3]}"
    if len(parts) == 3:
        return f"article {parts[2]}"
    if len(parts) == 2:
        return f"chapter {parts[1]}"
    return "legal section"


async def append_legal_hierarchy_results(
    chunks: dict[str, Any],
    llm_results: list,
) -> list:
    """Append legal document structure nodes and directed hierarchy edges."""

    raw_legal_document_id = _legal_document_id_from_chunks(chunks)
    legal_document_id = _coerce_legal_document_id(raw_legal_document_id)
    if legal_document_id is None:
        logger.warning("No legal_document_id found in chunks for legal hierarchy")
        return llm_results

    tvpl_key, legal_status, related_docs = await _load_legal_document_metadata(
        legal_document_id
    )
    if not tvpl_key:
        logger.warning(f"tvpl_key not found for legal_document_id {legal_document_id}")
        tvpl_key = f"doc_{legal_document_id}"

    legal_extra = {
        "legal_document_id": str(legal_document_id),
    }
    if legal_status:
        legal_extra["legal_status"] = legal_status

    hierarchy_nodes: list[dict[str, Any]] = [
        _node(
            tvpl_key,
            "LEGAL_DOCUMENT",
            f"Legal document with tvpl_key {tvpl_key}",
            **legal_extra,
        )
    ]
    hierarchy_edges: list[dict[str, Any]] = []
    created_nodes = {tvpl_key}

    for rel in related_docs:
        main_doc_id = _row_value(rel, "main_doc_id")
        related_doc_id = _row_value(rel, "related_doc_id")
        rel_type = _row_value(rel, "relation_type", "RELATES_TO")
        main_tvpl = _row_value(rel, "main_tvpl_key") or f"doc_{main_doc_id}"
        related_tvpl = _row_value(rel, "related_tvpl_key") or f"doc_{related_doc_id}"

        for tkey in (main_tvpl, related_tvpl):
            if tkey not in created_nodes:
                hierarchy_nodes.append(
                    _node(
                        tkey,
                        "LEGAL_DOCUMENT",
                        f"Legal document with tvpl_key {tkey}",
                    )
                )
                created_nodes.add(tkey)

        hierarchy_edges.append(
            _edge(
                main_tvpl,
                related_tvpl,
                rel_type,
                f"Document {main_tvpl} has relation {rel_type} with {related_tvpl}",
            )
        )

    chunk_deepest_node: dict[str, str] = {}
    for chunk_id, chunk in chunks.items():
        if not isinstance(chunk, dict):
            continue

        context = chunk.get("context", {})
        if not context:
            chunk_deepest_node[chunk_id] = tvpl_key
            continue

        current_parent = tvpl_key
        for context_key, entity_type, relation_type in (
            ("chapter", "CHAPTER", "HAS_CHAPTER"),
            ("article", "ARTICLE", "HAS_ARTICLE"),
            ("clauses", "CLAUSE", "HAS_CLAUSE"),
            ("points", "POINT", "HAS_POINT"),
        ):
            context_value = context.get(context_key, "")
            if not context_value:
                continue
            node_name = f"{current_parent}#{context_value}"
            _append_unique_structure_node(
                nodes=hierarchy_nodes,
                edges=hierarchy_edges,
                created_nodes=created_nodes,
                parent=current_parent,
                node_name=node_name,
                entity_type=entity_type,
                relation_type=relation_type,
                description=f"{entity_type.title()} {context_value} of {tvpl_key}",
                source_id=chunk_id,
                extra=legal_extra,
            )
            current_parent = node_name

        chunk_deepest_node[chunk_id] = current_parent

    entity_to_chunks: dict[str, set[str]] = defaultdict(set)
    extracted_entity_names: set[str] = set()
    for entity_name, entity_item in _iter_extracted_entities(llm_results):
        extracted_entity_names.add(entity_name)
        source_id = entity_item.get("source_id", "")
        if not source_id:
            continue
        for chunk_id in source_id.split(GRAPH_FIELD_SEP):
            chunk_id = chunk_id.strip()
            if chunk_id in chunks:
                entity_to_chunks[entity_name].add(chunk_id)

    linked_edges: set[tuple[str, str, str]] = set()
    for entity_name, chunk_ids in entity_to_chunks.items():
        for chunk_id in chunk_ids:
            parent_node = chunk_deepest_node.get(chunk_id, tvpl_key)
            chunk = chunks.get(chunk_id)

            if isinstance(chunk, dict):
                context = chunk.get("context", {})
                current_path = tvpl_key
                structural_paths: dict[str, str] = {}
                for context_key in ("chapter", "article", "clauses", "points"):
                    context_value = context.get(context_key, "")
                    if context_value:
                        current_path = f"{current_path}#{context_value}"
                        structural_paths[context_value] = current_path

                if entity_name in structural_paths:
                    structural_node = structural_paths[entity_name]
                    edge_key = (
                        entity_name,
                        structural_node,
                        "IS_STRUCTURAL_PART_OF",
                    )
                    if edge_key not in linked_edges:
                        linked_edges.add(edge_key)
                        hierarchy_edges.append(
                            _edge(
                                entity_name,
                                structural_node,
                                "IS_STRUCTURAL_PART_OF",
                                f"Entity {entity_name} corresponds to structural node {structural_node} of legal document {tvpl_key}",
                                chunk_id,
                                **legal_extra,
                            )
                        )

            if entity_name == parent_node:
                continue

            edge_key = (entity_name, parent_node, "MENTIONED_IN")
            if edge_key in linked_edges:
                continue
            linked_edges.add(edge_key)
            hierarchy_edges.append(
                _edge(
                    entity_name,
                    parent_node,
                    "MENTIONED_IN",
                    f"Entity {entity_name} is mentioned in {_level_name(parent_node)} of legal document {tvpl_key}",
                    chunk_id,
                    **legal_extra,
                )
            )

    for entity_name in extracted_entity_names:
        if entity_name != tvpl_key and entity_name not in entity_to_chunks:
            hierarchy_edges.append(
                _edge(
                    entity_name,
                    tvpl_key,
                    "MENTIONED_IN",
                    f"Entity {entity_name} is mentioned in legal document {tvpl_key}",
                    **legal_extra,
                )
            )

    llm_results.append((hierarchy_nodes, hierarchy_edges))
    logger.info(
        f"Added {len(hierarchy_nodes)} hierarchy nodes and {len(hierarchy_edges)} hierarchy edges"
    )
    return llm_results
