from __future__ import annotations

import asyncio
import inspect
import os
from datetime import datetime, timezone
from functools import partial
from typing import Any

from lightrag.base import DocStatus
from lightrag.operate import select_chunking_function
from lightrag.utils import compute_mdhash_id, logger

from lightrag.pipeline.models import PipelineStageEvent, StageName, StageResult

from .common import chunks_to_rows, get_rag_for_event

TERMINAL_SUCCESS_STATUSES = {"PROCESSED", "COMPLETED", "DUPLICATED"}


def _status_get(status_doc: Any, field: str, default: Any = None) -> Any:
    value = getattr(status_doc, field, None)
    if value is not None:
        return value
    if isinstance(status_doc, dict):
        return status_doc.get(field, default)
    return default


def _normalize_status(status: Any) -> str:
    if hasattr(status, "value"):
        status = status.value
    return str(status or "").strip().upper()


def _is_terminal_success_status(status: Any) -> bool:
    return _normalize_status(status) in TERMINAL_SUCCESS_STATUSES


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning(
            "[Pipeline] Invalid %s=%r, using default=%s",
            name,
            os.getenv(name),
            default,
        )
        return default


def _ensure_event_doc_id(event: PipelineStageEvent) -> None:
    if event.doc_id:
        return
    legal_document_id = event.metadata.get("legal_document_id")
    if legal_document_id is not None:
        event.doc_id = f"legal-{legal_document_id}"
        logger.warning(
            "[Pipeline] Repaired empty doc_id for legal chunking event run_id=%s legal_document_id=%s",
            event.run_id,
            legal_document_id,
        )
        return
    raise ValueError("Pipeline chunking event requires doc_id or legal_document_id")


def _compute_chunk_id(content: str, event: PipelineStageEvent) -> str:
    workspace = event.workspace_id
    if event.generation_id is not None:
        if not event.storage_workspace_id:
            raise ValueError(
                "Generation chunking event requires storage_workspace_id"
            )
        workspace = event.storage_workspace_id
    return compute_mdhash_id(content, prefix="chunk-", workspace=workspace)


async def _read_s3_text(object_key: str) -> str:
    import aioboto3
    from botocore.exceptions import ClientError

    endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    access_key = os.environ.get("S3_ACCESS_KEY") or os.environ.get("S3_ACCESS_KEY_ID")
    secret_key = os.environ.get("S3_SECRET_KEY") or os.environ.get(
        "S3_SECRET_ACCESS_KEY"
    )
    bucket = os.environ.get("S3_BUCKET", "lightrag-markdown")
    if not all([endpoint_url, access_key, secret_key]):
        raise ValueError(
            "S3 configuration is required to load pipeline document content"
        )

    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    ) as s3:
        try:
            response = await s3.get_object(Bucket=bucket, Key=object_key)
            async with response["Body"] as stream:
                return (await stream.read()).decode("utf-8")
        except ClientError as exc:
            raise FileNotFoundError(
                f"Document artifact not found: s3://{bucket}/{object_key}"
            ) from exc


def _normalize_s3_object_key(value: Any) -> str | None:
    object_key = str(value or "").strip()
    if not object_key:
        return None
    if object_key.startswith("s3://"):
        bucket_and_key = object_key[5:]
        if "/" not in bucket_and_key:
            return None
        object_key = bucket_and_key.split("/", 1)[1]
    return object_key.lstrip("/") or None


async def _legal_content_object_key(
    event: PipelineStageEvent,
    rag: Any,
    legal_document_id: Any,
) -> str | None:
    for metadata_key in ("content_path", "contentPath", "s3_key", "object_key"):
        object_key = _normalize_s3_object_key(event.metadata.get(metadata_key))
        if object_key:
            return object_key

    db = getattr(getattr(rag, "doc_status", None), "db", None)
    if db is None:
        return None

    try:
        from lightrag.kg.postgres_impl import to_uuid

        row = await db.query(
            'SELECT content_path FROM "tbl_legal_document" WHERE workspace_id=$1 AND id=$2',
            [to_uuid(event.workspace_id), int(legal_document_id)],
        )
    except Exception as exc:
        logger.warning(
            "[Pipeline] Failed to load legal content_path doc_id=%s workspace_id=%s: %s",
            event.doc_id,
            event.workspace_id,
            exc,
        )
        return None

    return _normalize_s3_object_key(row.get("content_path") if row else None)


async def _load_content(event: PipelineStageEvent, context: Any, rag: Any) -> str:
    if isinstance(event.metadata.get("content"), str):
        return event.metadata["content"]

    content_artifact_uri = event.metadata.get("content_artifact_uri")
    if content_artifact_uri:
        content_artifact = await context.artifact_store.read_json(content_artifact_uri)
        content = content_artifact.get("content")
        if not isinstance(content, str):
            raise ValueError(
                f"content_artifact_uri does not contain text content: {content_artifact_uri}"
            )
        return content

    document_name = event.metadata.get("document_name") or event.metadata.get(
        "file_path"
    )
    workspace_name = event.metadata.get("workspace_name") or event.workspace_id
    doc_type = event.metadata.get("doc_type", "general")
    legal_document_id = event.metadata.get("legal_document_id")

    if doc_type == "legal":
        if legal_document_id is None:
            raise ValueError("legal_document_id is required for legal document content")
        object_key = await _legal_content_object_key(event, rag, legal_document_id)
        if not object_key:
            object_key = f"legal_docs/{workspace_name}/{legal_document_id}/{legal_document_id}.html"
    else:
        object_key = f"full_docs/{event.workspace_id}/{event.doc_id}/{event.doc_id}.md"
        if document_name and str(document_name).endswith((".md", ".html", ".txt")):
            object_key = (
                f"full_docs/{event.workspace_id}/{event.doc_id}/{document_name}"
            )

    if document_name or doc_type == "legal":
        return await _read_s3_text(object_key)

    content_data = await rag.full_docs.get_by_id(event.doc_id)
    if content_data and content_data.get("content"):
        return content_data["content"]

    return await _read_s3_text(object_key)


def _select_chunking_func(event: PipelineStageEvent, rag: Any, file_path: str):
    doc_type = event.metadata.get("doc_type", "general")
    doc_extension = os.path.splitext(file_path or "")[1].lower().lstrip(".")
    return select_chunking_function(doc_type, doc_extension)


async def _run_chunking_func(
    chunking_func: Any,
    rag: Any,
    content: str,
    doc_id: str,
) -> Any:
    args = (
        rag.tokenizer,
        content,
        None,
        False,
        rag.chunk_overlap_token_size,
        rag.chunk_token_size,
    )
    kwargs = {"doc_id": doc_id}

    if inspect.iscoroutinefunction(chunking_func):
        result = await chunking_func(*args, **kwargs)
    else:
        loop = asyncio.get_running_loop()
        call = partial(chunking_func, *args, **kwargs)
        result = await loop.run_in_executor(None, call)

    if inspect.isawaitable(result):
        result = await result
    return result


async def run_chunking_stage(event: PipelineStageEvent, context: Any) -> StageResult:
    _ensure_event_doc_id(event)
    rag = await get_rag_for_event(context, event, profile=event.stage)
    status_doc = await rag.doc_status.get_by_id(event.doc_id)
    current_status = _status_get(status_doc, "status")
    if _is_terminal_success_status(current_status):
        logger.warning(
            "[Pipeline] Skipping chunking for terminal document doc_id=%s workspace_id=%s status=%s run_id=%s",
            event.doc_id,
            event.workspace_id,
            current_status,
            event.run_id,
        )
        return StageResult(message=f"skipped terminal document status={current_status}")

    file_path = (
        event.metadata.get("document_name")
        or event.metadata.get("file_path")
        or event.metadata.get("content_path")
        or event.metadata.get("s3_key")
        or event.metadata.get("object_key")
        or event.doc_id
    )
    doc_type = event.metadata.get("doc_type", "general")
    legal_document_id = event.metadata.get("legal_document_id")
    content = await _load_content(event, context, rag)
    if not content.strip():
        raise ValueError(f"Document content is empty for doc_id={event.doc_id}")

    chunking_func = _select_chunking_func(event, rag, file_path)
    chunking_result = await _run_chunking_func(
        chunking_func,
        rag,
        content,
        event.doc_id,
    )
    if not isinstance(chunking_result, (list, tuple)):
        raise TypeError(
            f"chunking_func must return a list or tuple, got {type(chunking_result)}"
        )

    chunks: dict[str, dict[str, Any]] = {}
    for index, chunk in enumerate(chunking_result):
        if "chunk_order_index" not in chunk:
            chunk["chunk_order_index"] = index
        chunk_id = _compute_chunk_id(chunk["content"], event)
        chunks[chunk_id] = {
            **chunk,
            "full_doc_id": event.doc_id,
            "file_path": file_path,
            "doc_type": doc_type,
            "llm_cache_list": [],
            "legal_document_id": legal_document_id,
        }

    if not chunks:
        raise ValueError(f"No chunks generated for doc_id={event.doc_id}")

    chunk_ids = list(chunks.keys())
    await rag.text_chunks.upsert(chunks)
    metadata = {}
    if status_doc is not None:
        metadata = _status_get(status_doc, "metadata", {})
    await rag.doc_status.upsert(
        {
            event.doc_id: {
                "status": DocStatus.CHUNKING,
                "chunks_count": len(chunks),
                "chunks_list": chunk_ids,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": file_path,
                "metadata": {
                    **metadata,
                    "pipeline_run_id": event.run_id,
                    "pipeline_stage": event.stage_name.value,
                },
            }
        }
    )

    chunks_artifact = await context.artifact_store.write_jsonl(
        run_id=event.run_id,
        stage=event.stage_name,
        name=f"{event.doc_id}.chunks.jsonl",
        rows=chunks_to_rows(chunks),
        kind="chunks",
        metadata={"doc_id": event.doc_id, "workspace_id": event.workspace_id},
    )
    manifest = await context.artifact_store.write_manifest(
        run_id=event.run_id,
        stage=event.stage_name,
        name=f"{event.doc_id}.manifest.json",
        manifest={
            "stage": event.stage_name.value,
            "doc_id": event.doc_id,
            "workspace_id": event.workspace_id,
            "output_artifact": chunks_artifact.model_dump(mode="json"),
            "counts": {"chunks": len(chunks)},
        },
    )

    chunk_embedding_shard_size = _env_int("PIPELINE_CHUNK_EMBEDDING_SHARD_SIZE", 512)
    if event.debug_sync or context.debug_sync:
        chunk_embedding_shard_size = len(chunk_ids)
    chunk_embedding_shard_size = max(1, chunk_embedding_shard_size)
    chunk_embedding_shard_count = (
        len(chunk_ids) + chunk_embedding_shard_size - 1
    ) // chunk_embedding_shard_size

    next_events = []
    for shard_index in range(chunk_embedding_shard_count):
        start = shard_index * chunk_embedding_shard_size
        end = start + chunk_embedding_shard_size
        shard_chunk_ids = chunk_ids[start:end]
        next_events.append(
            event.next_event(
                StageName.CHUNK_EMBEDDING,
                artifact_uri=chunks_artifact.uri,
                shard_index=shard_index,
                shard_count=chunk_embedding_shard_count,
                counts={"chunks": len(shard_chunk_ids)},
                metadata={
                    "chunks_artifact_uri": chunks_artifact.uri,
                    "chunk_ids": shard_chunk_ids,
                    "chunk_embedding_shard_size": chunk_embedding_shard_size,
                },
            )
        )

    shard_size = _env_int("PIPELINE_EXTRACT_CHUNK_SHARD_SIZE", 32)
    legal_single_shard = False
    if legal_single_shard or event.debug_sync or context.debug_sync:
        shard_size = len(chunk_ids)
    shard_size = max(1, shard_size)
    shard_count = (len(chunk_ids) + shard_size - 1) // shard_size

    for shard_index in range(shard_count):
        start = shard_index * shard_size
        end = start + shard_size
        shard_chunk_ids = chunk_ids[start:end]
        next_events.append(
            event.next_event(
                StageName.ENTITY_EXTRACTION,
                artifact_uri=chunks_artifact.uri,
                shard_index=shard_index,
                shard_count=shard_count,
                counts={"chunks": len(shard_chunk_ids)},
                metadata={
                    "chunks_artifact_uri": chunks_artifact.uri,
                    "chunk_ids": shard_chunk_ids,
                    "legal_single_shard": legal_single_shard,
                },
            )
        )

    logger.info(
        "[Pipeline] Chunking produced %s chunks for doc_id=%s doc_type=%s chunk_embedding_shards=%s extraction_shards=%s",
        len(chunks),
        event.doc_id,
        doc_type,
        chunk_embedding_shard_count,
        shard_count,
    )
    return StageResult(
        output_artifacts=[chunks_artifact],
        manifest=manifest,
        counts={"chunks": len(chunks)},
        next_events=next_events,
        message=f"chunked {len(chunks)} chunks",
    )
