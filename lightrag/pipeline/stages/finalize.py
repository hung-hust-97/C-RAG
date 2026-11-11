from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Literal

from lightrag.base import DocStatus
from lightrag.pipeline.models import PipelineStageEvent, StageResult
from lightrag.utils import logger

from kafka_worker.config import (
    KMS_FAILED_LEGAL_TOPIC,
    KMS_FAILED_TOPIC,
    KMS_PROCESSED_LEGAL_TOPIC,
    KMS_PROCESSED_TOPIC,
    PIPELINE_FAILED_TOPIC,
)
from kafka_worker.models import KmsEvent
from kafka_worker.partitioned_publish import publish_pipeline_event

from .common import get_rag_for_event


DocType = Literal["general", "legal"]


def _status_get(status_doc: Any, field: str, default: Any = None) -> Any:
    value = getattr(status_doc, field, None)
    if value is not None:
        return value

    if isinstance(status_doc, dict):
        return status_doc.get(field, default)

    return default


def _get_metadata_value(
    metadata: dict[str, Any], *keys: str, default: Any = None
) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return default


def _detect_doc_type(event: PipelineStageEvent) -> DocType:
    raw_doc_type = str(getattr(event, "doc_type", "general") or "general").lower()
    metadata_doc_type = str(event.metadata.get("doc_type", "") or "").lower()

    if (
        raw_doc_type == "legal"
        or metadata_doc_type == "legal"
        or event.metadata.get("legal_document_id") is not None
        or event.metadata.get("legalDocumentId") is not None
    ):
        return "legal"

    return "general"


def _get_kms_topic(doc_type: DocType, has_error: bool) -> str:
    if doc_type == "legal":
        return KMS_FAILED_LEGAL_TOPIC if has_error else KMS_PROCESSED_LEGAL_TOPIC

    return KMS_FAILED_TOPIC if has_error else KMS_PROCESSED_TOPIC


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _build_base_kms_event(
    event: PipelineStageEvent,
    doc_type: DocType,
    error_info: dict[str, Any] | None,
) -> KmsEvent:
    try:
        kms_event = KmsEvent.model_validate(event.metadata)
    except Exception:
        kms_event = KmsEvent(
            workspace_id=event.workspace_id,
            workspace_name=_get_metadata_value(
                event.metadata,
                "workspaceName",
                "workspace_name",
                default="",
            ),
            document_name=_get_metadata_value(
                event.metadata,
                "documentName",
                "document_name",
                default="",
            ),
            doc_type=doc_type,
            status=DocStatus.FAILED if error_info else DocStatus.PROCESSED,
        )

    kms_event.doc_type = doc_type

    if error_info:
        kms_event.status = DocStatus.FAILED
        kms_event.message = error_info.get("message") or "Pipeline failed"
    elif str(getattr(event, "status", "")).upper() == "DUPLICATED":
        kms_event.status = DocStatus.DUPLICATED
        kms_event.message = "Tài liệu trùng lặp về nội dung"
    else:
        kms_event.status = DocStatus.PROCESSED
        kms_event.message = ""

    return kms_event


def _prepare_general_kms_event(
    kms_event: KmsEvent,
    event: PipelineStageEvent,
) -> KmsEvent:
    document_id = (
        kms_event.document_id
        or event.doc_id
        or _get_metadata_value(event.metadata, "document_id", "documentId")
    )

    kms_event.document_id = document_id
    kms_event.legal_document_id = None

    return kms_event


def _prepare_legal_kms_event(
    kms_event: KmsEvent,
    event: PipelineStageEvent,
) -> KmsEvent:
    legal_document_id = kms_event.legal_document_id or _get_metadata_value(
        event.metadata,
        "legal_document_id",
        "legalDocumentId",
    )

    parsed_legal_id = _safe_int(legal_document_id)
    if parsed_legal_id is None and event.doc_id:
        clean_doc_id = event.doc_id.removeprefix("legal-")
        parsed_legal_id = _safe_int(clean_doc_id)

    if parsed_legal_id is not None:
        kms_event.legal_document_id = parsed_legal_id

    kms_event.document_id = None

    return kms_event


def _prepare_kms_event(
    event: PipelineStageEvent,
    doc_type: DocType,
    error_info: dict[str, Any] | None,
) -> KmsEvent:
    kms_event = _build_base_kms_event(event, doc_type, error_info)

    if doc_type == "legal":
        return _prepare_legal_kms_event(kms_event, event)

    return _prepare_general_kms_event(kms_event, event)


def _get_doc_id_for_log(kms_event: KmsEvent, event: PipelineStageEvent) -> Any:
    return kms_event.document_id or kms_event.legal_document_id or event.doc_id


def _write_failed_document_log(
    event: PipelineStageEvent,
    topic: str,
    doc_id: Any,
    error_message: str,
) -> None:
    log_dir = os.environ.get("LIGHTRAG_LOG_DIR", "/app/data/logs")
    os.makedirs(log_dir, exist_ok=True)

    failed_log_path = os.path.join(log_dir, "failed_documents.log")

    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        stage_name = event.stage_name.value if event.stage_name else "unknown"

        with open(failed_log_path, "a", encoding="utf-8") as file:
            file.write(
                f"[{timestamp}] "
                f"DOC_ID: {doc_id} | "
                f"TOPIC: {topic} | "
                f"STAGE: {stage_name} | "
                f"ERROR: {error_message}\n"
            )
    except Exception as log_err:
        logger.error(f"[Egress] Failed to write to {failed_log_path}: {log_err}")


def _log_kms_publish_result(
    event: PipelineStageEvent,
    kms_event: KmsEvent,
    topic: str,
    has_error: bool,
) -> None:
    doc_id = _get_doc_id_for_log(kms_event, event)

    if has_error:
        logger.error(
            f"[Egress] Published {kms_event.status.value} to {topic} "
            f"for doc {doc_id} | Reason: {kms_event.message}"
        )
        _write_failed_document_log(event, topic, doc_id, kms_event.message)
        return

    logger.info(
        f"[Egress] Published {kms_event.status.value} to {topic} for doc {doc_id}"
    )


async def _publish_kms_event(
    event: PipelineStageEvent,
    context: Any,
    doc_type: DocType,
    error_info: dict[str, Any] | None,
) -> None:
    kms_event = _prepare_kms_event(event, doc_type, error_info)
    topic = _get_kms_topic(doc_type, has_error=error_info is not None)

    payload = kms_event.model_dump_json(
        by_alias=True,
        exclude_none=True,
    ).encode("utf-8")

    doc_id = _get_doc_id_for_log(kms_event, event)
    key = str(doc_id).encode("utf-8") if doc_id else None
    await context.producer.send_and_wait(topic, payload, key=key)

    _log_kms_publish_result(
        event=event,
        kms_event=kms_event,
        topic=topic,
        has_error=error_info is not None,
    )


async def _publish_pipeline_failed_event(
    event: PipelineStageEvent,
    context: Any,
    doc_type: DocType,
    error_info: dict[str, Any],
) -> None:
    if not getattr(context, "partition_allocator", None):
        return

    failure_event = event.model_copy(deep=True)
    failure_event.status = "failed"
    failure_event.doc_type = doc_type
    failure_event.metadata = {
        **failure_event.metadata,
        "doc_type": doc_type,
        "error": error_info,
    }

    await publish_pipeline_event(
        context.producer,
        context.partition_allocator,
        PIPELINE_FAILED_TOPIC,
        failure_event,
    )


async def _publish_finalize_events(
    event: PipelineStageEvent,
    context: Any,
    doc_type: DocType,
    error_info: dict[str, Any] | None,
) -> None:
    if not getattr(context, "producer", None):
        return

    try:
        await _publish_kms_event(
            event=event,
            context=context,
            doc_type=doc_type,
            error_info=error_info,
        )

        if error_info:
            await _publish_pipeline_failed_event(
                event=event,
                context=context,
                doc_type=doc_type,
                error_info=error_info,
            )

    except Exception as exc:
        logger.error(
            f"[Egress] Failed to publish event to Kafka in finalize stage: {exc}"
        )


async def _get_chunks_count(event: PipelineStageEvent, context: Any) -> int:
    try:
        rag = await get_rag_for_event(context, event, profile=event.stage)
        status_doc = await rag.doc_status.get_by_id(event.doc_id)
    except Exception:
        return 0

    if status_doc is None:
        return 0

    chunks_list = _status_get(status_doc, "chunks_list", [])
    chunks_count = _status_get(status_doc, "chunks_count", len(chunks_list))

    return int(chunks_count or 0)


def _build_failed_result(
    event: PipelineStageEvent, error_info: dict[str, Any]
) -> StageResult:
    error_msg = error_info.get("message") or "Pipeline failed"
    error_stage = error_info.get("stage") or "unknown"

    return StageResult(
        counts={},
        message=(
            f"finalized failed doc_id={event.doc_id} "
            f"from stage={error_stage} error={error_msg}"
        ),
    )


def _build_success_result(event: PipelineStageEvent, chunks_count: int) -> StageResult:
    return StageResult(
        counts={"chunks": chunks_count},
        message=f"finalized doc_id={event.doc_id}",
    )


async def run_finalize_stage(event: PipelineStageEvent, context: Any) -> StageResult:
    doc_type = _detect_doc_type(event)
    error_info = event.metadata.get("error")

    await _publish_finalize_events(
        event=event,
        context=context,
        doc_type=doc_type,
        error_info=error_info,
    )

    if error_info:
        try:
            from lightrag.pipeline.models import StageName
            rag = await get_rag_for_event(
                context, event, profile=StageName.FINALIZE.value
            )
            await rag.doc_status.upsert(
                {
                    event.doc_id: {
                        "status": DocStatus.FAILED,
                        "error_msg": error_info.get("message", "Pipeline failed"),
                        "error_stage": error_info.get("stage", "unknown"),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                }
            )
        except Exception as exc:
            logger.warning("[Pipeline] Finalize failed to update doc_status: %s", exc)
        return _build_failed_result(event, error_info)

    if getattr(context, "artifact_store", None):
        cleanup_enabled = os.getenv("PIPELINE_CLEANUP_ARTIFACTS_ON_FINALIZE", "true").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if cleanup_enabled:
            try:
                cleaned_count = await context.artifact_store.cleanup_run_artifacts(
                    event.run_id
                )
                if cleaned_count > 0:
                    logger.info(
                        "[Pipeline] Cleaned up %d intermediate artifacts for run_id=%s doc_id=%s",
                        cleaned_count,
                        event.run_id,
                        event.doc_id,
                    )
            except Exception as exc:
                logger.warning(
                    "[Pipeline] Failed to cleanup intermediate artifacts for run_id=%s doc_id=%s: %s",
                    event.run_id,
                    event.doc_id,
                    exc,
                )

    chunks_count = await _get_chunks_count(event, context)

    return _build_success_result(event, chunks_count)
