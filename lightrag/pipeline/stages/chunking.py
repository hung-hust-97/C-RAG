from __future__ import annotations

import asyncio
import inspect
import os
from datetime import date, datetime, time as dt_time, timezone
from functools import partial
from typing import Any

from lightrag.base import DocStatus
from lightrag.legal_document.chunking import looks_like_html
from lightrag.operate import select_chunking_function
from lightrag.utils import compute_mdhash_id, logger

from lightrag.pipeline.legal_kg import legal_kg_enabled
from lightrag.pipeline.models import PipelineStageEvent, StageName, StageResult
from lightrag.pipeline.s3_throttle import read_body

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


def _to_unix_timestamp(val: Any) -> int:
    if not val:
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, datetime):
        return int(val.replace(tzinfo=timezone.utc).timestamp())
    if isinstance(val, date):
        return int(datetime.combine(val, dt_time.min, tzinfo=timezone.utc).timestamp())
    if isinstance(val, str):
        val = val.strip()
        if not val or val == "None" or val == "0":
            return 0
        if val.isdigit():
            return int(val)
        try:
            dt = datetime.fromisoformat(val)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            return 0
    return 0


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


def _extract_legal_document_id(event: PipelineStageEvent) -> int | None:
    val = event.metadata.get("legal_document_id")
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            pass

    for candidate in (event.doc_id, event.run_id):
        if not candidate:
            continue
        c_str = str(candidate).strip()
        if c_str.isdigit():
            return int(c_str)
        if c_str.startswith("legal-"):
            suffix = c_str.removeprefix("legal-")
            if suffix.isdigit():
                return int(suffix)

    for path_key in ("content_path", "file_path", "document_name"):
        path_val = event.metadata.get(path_key)
        if path_val and isinstance(path_val, str):
            import re
            m = re.search(r"/(\d+)(?:/\1)?\.(?:html|htm|md|txt)$", path_val)
            if m:
                return int(m.group(1))

    return None


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
                return (await read_body(stream)).decode("utf-8")
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

        row = None
        if event.workspace_id:
            try:
                row = await db.query(
                    'SELECT content_path FROM "tbl_legal_document" WHERE workspace_id=$1 AND id=$2',
                    [to_uuid(event.workspace_id), int(legal_document_id)],
                )
            except Exception:
                row = None

        if not row:
            row = await db.query(
                'SELECT content_path FROM "tbl_legal_document" WHERE id=$1',
                [int(legal_document_id)],
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
    doc_type = event.metadata.get("doc_type") or event.doc_type or ("legal" if os.environ.get("PIPELINE_DOC_TYPE") == "legal" else "general")
    legal_document_id = _extract_legal_document_id(event)
    if legal_document_id is not None:
        doc_type = "legal"
        event.metadata["legal_document_id"] = legal_document_id

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


def _select_chunking_func(
    event: PipelineStageEvent, rag: Any, file_path: str, content: str = ""
):
    doc_type = event.metadata.get("doc_type", "general")
    doc_extension = os.path.splitext(file_path or "")[1].lower().lstrip(".")
    # Legal content is read from legal_docs/.../<id>.html but file_path is the
    # document title (no extension), so detect HTML from the content itself.
    if doc_type == "legal" and looks_like_html(content):
        doc_extension = "html"
    return select_chunking_function(doc_type, doc_extension)


async def _run_chunking_func(
    chunking_func: Any,
    rag: Any,
    content: str,
    doc_id: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    args = (
        rag.tokenizer,
        content,
        None,
        False,
        rag.chunk_overlap_token_size,
        rag.chunk_token_size,
    )
    kwargs = {"doc_id": doc_id, **(extra_kwargs or {})}

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
    doc_type = event.metadata.get("doc_type") or event.doc_type or ("legal" if os.environ.get("PIPELINE_DOC_TYPE") == "legal" else "general")
    legal_document_id = _extract_legal_document_id(event)
    if legal_document_id is not None:
        doc_type = "legal"
        event.metadata["legal_document_id"] = legal_document_id

    content = await _load_content(event, context, rag)
    if not content.strip():
        raise ValueError(f"Document content is empty for doc_id={event.doc_id}")

    extra_kwargs = {}
    if doc_type == "legal" and legal_document_id is not None:
        db = getattr(getattr(rag, "doc_status", None), "db", None)
        if db is None:
            try:
                from lightrag.kg.postgres_impl import ClientManager
                db = await ClientManager.get_client()
            except Exception:
                db = None

        if db is not None:
            try:
                from lightrag.kg.postgres_impl import to_uuid
                from lightrag.legal_document.status import normalize_legal_document_status

                legal_row = None
                if event.workspace_id:
                    try:
                        legal_row = await db.query(
                            'SELECT status, lien_quan_hieu_luc_tung_phan, lien_quan_dan_chieu, document_code, title, document_type, effective_date, status_date, issuance_date FROM "tbl_legal_document" WHERE workspace_id=$1 AND id=$2',
                            [to_uuid(event.workspace_id), int(legal_document_id)],
                        )
                    except Exception:
                        legal_row = None

                if not legal_row:
                    legal_row = await db.query(
                        'SELECT status, lien_quan_hieu_luc_tung_phan, lien_quan_dan_chieu, document_code, title, document_type, effective_date, status_date, issuance_date FROM "tbl_legal_document" WHERE id=$1',
                        [int(legal_document_id)],
                    )

                if legal_row:
                    raw_status = legal_row.get("status")
                    norm_status = normalize_legal_document_status(raw_status)
                    doc_status_code = norm_status.get("legal_status_code") or (
                        str(raw_status).upper() if raw_status else "UNKNOWN"
                    )

                    extra_kwargs["legal_status"] = norm_status.get("legal_status") or raw_status
                    extra_kwargs["legal_status_code"] = doc_status_code
                    extra_kwargs["legal_status_priority"] = norm_status.get("legal_status_priority", 40)
                    extra_kwargs["lien_quan_hieu_luc_tung_phan"] = legal_row.get(
                        "lien_quan_hieu_luc_tung_phan"
                    )
                    extra_kwargs["lien_quan_dan_chieu"] = legal_row.get(
                        "lien_quan_dan_chieu"
                    )
                    if legal_row.get("document_code"):
                        extra_kwargs["document_number"] = legal_row.get("document_code")
                    if legal_row.get("title"):
                        extra_kwargs["document_title"] = legal_row.get("title")
                    if legal_row.get("document_type"):
                        extra_kwargs["document_type"] = legal_row.get("document_type")

                    eff_date = legal_row.get("effective_date")
                    iss_date = legal_row.get("issuance_date")
                    stat_date = legal_row.get("status_date")
                    extra_kwargs["effective_date"] = eff_date
                    extra_kwargs["issuance_date"] = iss_date
                    extra_kwargs["status_date"] = stat_date

                    effective_from = _to_unix_timestamp(eff_date or iss_date)
                    effective_to = 0
                    if doc_status_code in (
                        "EXPIRED",
                        "EXPIRED_TOTALLY",
                        "HET_HIEU_LUC",
                        "CEASED_EFFECT",
                        "NO_LONGER_SUITABLE",
                    ):
                        effective_to = _to_unix_timestamp(stat_date)
                    extra_kwargs["effective_from"] = effective_from
                    extra_kwargs["effective_to"] = effective_to
            except Exception as e:
                logger.debug(
                    "[Pipeline] Could not load partial expiry / reference metadata doc_id=%s: %s",
                    event.doc_id,
                    e,
                )

    chunking_func = _select_chunking_func(event, rag, file_path, content)
    chunking_result = await _run_chunking_func(
        chunking_func,
        rag,
        content,
        event.doc_id,
        extra_kwargs=extra_kwargs,
    )
    if not isinstance(chunking_result, (list, tuple)):
        raise TypeError(
            f"chunking_func must return a list or tuple, got {type(chunking_result)}"
        )

    chunks: dict[str, dict[str, Any]] = {}
    from lightrag.legal_document.chunking import build_parent_section
    from lightrag.legal_document.status import normalize_legal_document_status

    def _resolve_parent_section(chunk_data: dict[str, Any]) -> str | None:
        direct = chunk_data.get("parent_section")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        ctx = chunk_data.get("context")
        if isinstance(ctx, dict):
            ctx_ps = ctx.get("parent_section")
            if isinstance(ctx_ps, str) and ctx_ps.strip():
                return ctx_ps.strip()
            built = build_parent_section(
                part=str(ctx.get("part") or ""),
                chapter=str(ctx.get("chapter") or ""),
                section=str(ctx.get("section") or ""),
                article=str(ctx.get("article") or ""),
            )
            if built:
                return built
            breadcrumb = ctx.get("breadcrumb")
            if isinstance(breadcrumb, list) and breadcrumb:
                joined = " > ".join(str(b).strip() for b in breadcrumb if str(b).strip())
                if joined:
                    return joined
        return None

    for index, chunk in enumerate(chunking_result):
        if "chunk_order_index" not in chunk:
            chunk["chunk_order_index"] = index
        chunk_id = _compute_chunk_id(chunk["content"], event)
        parent_section = _resolve_parent_section(chunk)

        is_legal_doc = str(doc_type).strip().lower() == "legal"
        if is_legal_doc:
            chunk_status = (
                chunk.get("clause_status")
                or extra_kwargs.get("legal_status_code")
                or extra_kwargs.get("legal_status")
                or "UNKNOWN"
            )
            if chunk_status and isinstance(chunk_status, str):
                norm = normalize_legal_document_status(chunk_status)
                chunk_status = norm.get("legal_status_code") or chunk_status
            legal_status_val = extra_kwargs.get("legal_status") or chunk_status
            legal_status_code_val = chunk_status
        else:
            chunk_status = None
            legal_status_val = None
            legal_status_code_val = None

        chunk_eff_from = chunk.get("effective_from") or extra_kwargs.get("effective_from", 0)
        chunk_eff_to = chunk.get("effective_to") or extra_kwargs.get("effective_to", 0)

        chunks[chunk_id] = {
            **chunk,
            "full_doc_id": event.doc_id,
            "file_path": file_path,
            "doc_type": doc_type,
            "parent_section": parent_section,
            "llm_cache_list": [],
            "legal_document_id": legal_document_id,
            "document_type": chunk.get("document_type") or extra_kwargs.get("document_type") or "",
            "status": chunk_status,
            "legal_status_code": legal_status_code_val,
            "legal_status": legal_status_val,
            "effective_from": _to_unix_timestamp(chunk_eff_from),
            "effective_to": _to_unix_timestamp(chunk_eff_to),
            "effective_date": str(extra_kwargs.get("effective_date") or ""),
            "status_date": str(extra_kwargs.get("status_date") or ""),
            "issuance_date": str(extra_kwargs.get("issuance_date") or ""),
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

    shard_count = 0
    if legal_kg_enabled(event):
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
