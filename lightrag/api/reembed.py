"""Utilities for rebuilding vector embeddings from existing workspace data."""

from __future__ import annotations

import asyncio
from typing import Any

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import compute_mdhash_id, logger


async def reembed_workspace_vectors(
    rag: LightRAG,
    batch_size: int = 256,
) -> dict[str, int]:
    """Rebuild chunk/entity/relation vectors from current workspace storages.

    This operation keeps documents and graph data intact, but recreates the
    vector storages so they match the active embedding model configuration.
    """
    if batch_size <= 0:
        batch_size = 256

    # Collect chunk ids from processed documents.
    processed_docs = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)
    preprocessed_docs = await rag.doc_status.get_docs_by_status(DocStatus.PREPROCESSED)

    doc_status_map: dict[str, Any] = {}
    doc_status_map.update(processed_docs or {})
    doc_status_map.update(preprocessed_docs or {})

    chunk_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    for doc_status in doc_status_map.values():
        chunks_list = getattr(doc_status, "chunks_list", None) or []
        for chunk_id in chunks_list:
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                chunk_ids.append(chunk_id)

    # Recreate vector storages first to guarantee a clean index schema.
    await asyncio.gather(
        rag.chunks_vdb.drop(),
        rag.entities_vdb.drop(),
        rag.relationships_vdb.drop(),
    )

    chunks_upserted = 0
    entities_upserted = 0
    relations_upserted = 0

    # Rebuild chunk vectors from KV text chunks.
    for i in range(0, len(chunk_ids), batch_size):
        batch_ids = chunk_ids[i : i + batch_size]
        batch_rows = await rag.text_chunks.get_by_ids(batch_ids)
        batch_payload: dict[str, dict[str, Any]] = {}
        for chunk_id, chunk_row in zip(batch_ids, batch_rows):
            if chunk_row:
                batch_payload[chunk_id] = chunk_row

        if batch_payload:
            await rag.chunks_vdb.upsert(batch_payload)
            chunks_upserted += len(batch_payload)

    # Rebuild entity vectors from graph nodes.
    nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
    for i in range(0, len(nodes), batch_size):
        batch_nodes = nodes[i : i + batch_size]
        batch_payload: dict[str, dict[str, Any]] = {}
        for node in batch_nodes:
            entity_name = (
                node.get("entity_name")
                or node.get("entity_id")
                or node.get("id")
                or ""
            )
            if not entity_name:
                continue

            description = node.get("description", "")
            batch_payload[compute_mdhash_id(entity_name, prefix="ent-")] = {
                "content": f"{entity_name}\n{description}",
                "entity_name": entity_name,
                "source_id": node.get("source_id", ""),
                "description": description,
                "entity_type": node.get("entity_type", ""),
                "file_path": node.get("file_path", ""),
            }

        if batch_payload:
            await rag.entities_vdb.upsert(batch_payload)
            entities_upserted += len(batch_payload)

    # Rebuild relationship vectors from graph edges.
    edges = await rag.chunk_entity_relation_graph.get_all_edges()
    for i in range(0, len(edges), batch_size):
        batch_edges = edges[i : i + batch_size]
        batch_payload: dict[str, dict[str, Any]] = {}
        for edge in batch_edges:
            src_id = edge.get("src_id") or edge.get("source") or ""
            tgt_id = edge.get("tgt_id") or edge.get("target") or ""
            if not src_id or not tgt_id:
                continue

            keywords = edge.get("keywords", "")
            description = edge.get("description", "")
            batch_payload[compute_mdhash_id(f"{src_id}{tgt_id}", prefix="rel-")] = {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "source_id": edge.get("source_id", ""),
                "content": f"{keywords}\t{src_id}\n{tgt_id}\n{description}",
                "keywords": keywords,
                "description": description,
                "weight": edge.get("weight", 1.0),
                "file_path": edge.get("file_path", ""),
            }

        if batch_payload:
            await rag.relationships_vdb.upsert(batch_payload)
            relations_upserted += len(batch_payload)

    # Persist vector indices if backend requires callback flush.
    await asyncio.gather(
        rag.chunks_vdb.index_done_callback(),
        rag.entities_vdb.index_done_callback(),
        rag.relationships_vdb.index_done_callback(),
    )

    logger.info(
        "[%s] Re-embedding completed: chunks=%s, entities=%s, relations=%s",
        rag.workspace,
        chunks_upserted,
        entities_upserted,
        relations_upserted,
    )

    return {
        "documents": len(doc_status_map),
        "chunks": chunks_upserted,
        "entities": entities_upserted,
        "relations": relations_upserted,
    }
