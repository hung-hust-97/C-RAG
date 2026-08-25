from __future__ import annotations

from typing import Any
import os
import aioboto3
from botocore.exceptions import ClientError
from lightrag.utils import logger
from lightrag.kg.postgres_impl import ClientManager
from lightrag.kg.postgres_impl import to_uuid

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import compute_mdhash_id


def chunks_to_rows(chunks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"chunk_id": chunk_id, "chunk": chunk} for chunk_id, chunk in chunks.items()
    ]


def rows_to_chunks(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for row in rows:
        chunk_id = row["chunk_id"]
        chunks[chunk_id] = row["chunk"]
    return chunks


def filter_chunks(
    chunks: dict[str, dict[str, Any]], chunk_ids: list[str] | None
) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return chunks
    wanted = set(chunk_ids)
    return {chunk_id: chunk for chunk_id, chunk in chunks.items() if chunk_id in wanted}


def serialize_extraction_results(results: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for index, result in enumerate(results):
        maybe_nodes, maybe_edges = result
        rows.append(
            {
                "index": index,
                "nodes": maybe_nodes,
                "edges": _serialize_edges(maybe_edges),
            }
        )
    return rows


def deserialize_extraction_results(rows: list[dict[str, Any]]) -> list[tuple[Any, Any]]:
    rows = sorted(rows, key=lambda row: row.get("index", 0))
    results = []
    for row in rows:
        results.append((row.get("nodes", {}), _deserialize_edges(row.get("edges", []))))
    return results


def _serialize_edges(edges: Any) -> Any:
    if isinstance(edges, dict):
        serialized = []
        for key, value in edges.items():
            if isinstance(key, tuple):
                src, tgt = key
            elif isinstance(key, list) and len(key) == 2:
                src, tgt = key
            else:
                parts = str(key).split(GRAPH_FIELD_SEP, 1)
                src = parts[0]
                tgt = parts[1] if len(parts) > 1 else ""
            serialized.append({"src": src, "tgt": tgt, "items": value})
        return serialized
    return edges


def _deserialize_edges(edges: Any) -> Any:
    if isinstance(edges, list) and all(
        isinstance(item, dict) and "items" in item for item in edges
    ):
        return {
            (item.get("src", ""), item.get("tgt", "")): item.get("items", [])
            for item in edges
        }
    return edges


def build_entity_vector_payload(
    entity_name: str, node_data: dict[str, Any], workspace: str
) -> tuple[str, dict[str, Any]]:
    description = node_data.get("description", "")
    entity_type = node_data.get("entity_type", "UNKNOWN")
    source_id = node_data.get("source_id", "")
    file_path = node_data.get("file_path", "unknown_source")
    payload = {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "content": f"{entity_name}\n{description}",
        "source_id": source_id,
        "file_path": file_path,
    }
    for key in (
        "legal_status",
        "effective_date",
        "expiry_date",
        "issuing_authority",
        "document_number",
    ):
        if key in node_data:
            payload[key] = node_data[key]
    return compute_mdhash_id(entity_name, prefix="ent-", workspace=workspace), payload


def build_relation_vector_payload(
    src_id: str, tgt_id: str, edge_data: dict[str, Any], workspace: str
) -> tuple[str, dict[str, Any]]:
    keywords = edge_data.get("keywords", "")
    description = edge_data.get("description", "")
    payload = {
        "src_id": src_id,
        "tgt_id": tgt_id,
        "source_id": edge_data.get("source_id", ""),
        "content": f"{keywords}\t{src_id}\n{tgt_id}\n{description}",
        "keywords": keywords,
        "description": description,
        "weight": edge_data.get("weight", 1.0),
        "file_path": edge_data.get("file_path", "unknown_source"),
    }
    return compute_mdhash_id(
        src_id + tgt_id, prefix="rel-", workspace=workspace
    ), payload


async def read_file_from_storage(
    workspace_id: str, doc_type: str, document_id: str, document_name: str
) -> bytes:
    """Đọc file gốc từ MinIO/S3."""
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    access_key = os.environ.get("S3_ACCESS_KEY") or os.environ.get("S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_KEY") or os.environ.get(
        "S3_SECRET_ACCESS_KEY"
    )
    bucket = os.environ.get("S3_BUCKET", "lightrag-markdown")

    if (
        doc_type == "legal"
        and document_name.startswith("law_")
        and document_name.endswith(".pdf")
    ):
        bucket = "legal-docs-pdf"
        object_key = document_name
    else:
        object_key = f"full_docs/{workspace_id}/{document_id}/{document_name}"

    if not all([endpoint_url, access_key, secret_key]):
        raise ValueError("Missing S3 credentials in environment variables")

    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ) as s3:
        try:
            logger.info(f"[MinIO] Downloading s3://{bucket}/{object_key}")
            response = await s3.get_object(Bucket=bucket, Key=object_key)
            async with response["Body"] as stream:
                data = await stream.read()
            logger.info(
                f"[MinIO] Successfully read {len(data)} bytes from s3://{bucket}/{object_key}"
            )
            return data
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = f"File not found in MinIO: s3://{bucket}/{object_key}. Code: {error_code}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e


def _s3_credentials() -> tuple[str | None, str | None, str | None]:
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    access_key = os.environ.get("S3_ACCESS_KEY") or os.environ.get("S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_KEY") or os.environ.get(
        "S3_SECRET_ACCESS_KEY"
    )
    return endpoint_url, access_key, secret_key


async def read_legal_pdf_from_storage(doc_id: str) -> bytes:
    """Read a legal source PDF from the fixed MinIO location."""
    endpoint_url, access_key, secret_key = _s3_credentials()
    bucket = os.environ.get("LEGAL_DOCS_PDF_BUCKET", "legal-docs-pdf")
    object_key = f"law_{doc_id}.pdf"

    if not all([endpoint_url, access_key, secret_key]):
        raise ValueError("Missing S3 credentials in environment variables")

    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ) as s3:
        try:
            logger.info(f"[MinIO] Downloading legal PDF s3://{bucket}/{object_key}")
            response = await s3.get_object(Bucket=bucket, Key=object_key)
            async with response["Body"] as stream:
                data = await stream.read()
            logger.info(
                f"[MinIO] Successfully read {len(data)} bytes from s3://{bucket}/{object_key}"
            )
            return data
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = (
                f"Legal PDF not found in MinIO: s3://{bucket}/{object_key}. "
                f"Code: {error_code}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e


async def write_text_to_storage(
    object_key: str,
    content: str,
    *,
    bucket: str | None = None,
    content_type: str = "text/plain; charset=utf-8",
) -> str:
    """Write UTF-8 text to MinIO/S3 and return the object key."""
    endpoint_url, access_key, secret_key = _s3_credentials()
    target_bucket = bucket or os.environ.get("S3_BUCKET", "lightrag-markdown")

    if not all([endpoint_url, access_key, secret_key]):
        raise ValueError("Missing S3 credentials in environment variables")

    payload = content.encode("utf-8")
    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ) as s3:
        await s3.put_object(
            Bucket=target_bucket,
            Key=object_key,
            Body=payload,
            ContentType=content_type,
        )
    logger.info(
        f"[MinIO] Uploaded {len(payload)} bytes to s3://{target_bucket}/{object_key}"
    )
    return object_key


async def update_legal_document_metadata(
    legal_document_id: int,
    workspace_id: str,
    content_path: str,
) -> None:
    """Update extracted HTML metadata for tbl_legal_document."""
    ws_uuid = to_uuid(workspace_id)
    db = await ClientManager.get_client()
    async with db.pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE "tbl_legal_document"
            SET "content_path" = $1,
                "updated_at" = NOW()
            WHERE "id" = $2 AND "workspace_id" = $3
            """,
            content_path,
            legal_document_id,
            ws_uuid,
        )


async def update_document_metadata(
    document_id: str,
    workspace_id: str,
    content_path: str,
    content_hash: str,
    content_length: int,
    content_summary: str,
) -> None:
    """Cập nhật metadata vào tbl_document."""
    doc_uuid = to_uuid(document_id)
    ws_uuid = to_uuid(workspace_id)

    db = await ClientManager.get_client()
    async with db.pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE "tbl_document"
            SET "content_path" = $1,
                "content_hash" = $2,
                "content_length" = $3,
                "content_summary" = $4,
                "update_time" = NOW()
            WHERE "document_id" = $5 AND "workspace_id" = $6
            """,
            content_path,
            content_hash,
            content_length,
            content_summary,
            doc_uuid,
            ws_uuid,
        )


async def check_duplicate_by_content_hash(
    content_hash: str, workspace_id: str
) -> str | None:
    """Kiểm tra hash duplicate."""
    ws_uuid = to_uuid(workspace_id)
    db = await ClientManager.get_client()
    async with db.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT "document_id" 
            FROM "tbl_document" 
            WHERE "content_hash" = $1 AND "workspace_id" = $2 
            LIMIT 1
            """,
            content_hash,
            ws_uuid,
        )
        if row:
            return str(row["document_id"])
    return None


async def mark_document_as_duplicated(document_id: str, workspace_id: str) -> None:
    """Đánh dấu file DUPLICATED."""
    doc_uuid = to_uuid(document_id)
    ws_uuid = to_uuid(workspace_id)
    db = await ClientManager.get_client()
    async with db.pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE "tbl_document"
            SET "status" = 'DUPLICATED', "update_time" = NOW()
            WHERE "document_id" = $1 AND "workspace_id" = $2
            """,
            doc_uuid,
            ws_uuid,
        )
