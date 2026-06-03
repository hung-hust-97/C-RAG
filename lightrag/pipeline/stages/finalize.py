from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from lightrag.base import DocStatus
from lightrag.pipeline.models import PipelineStageEvent, StageResult


def _status_get(status_doc: Any, field: str, default: Any = None) -> Any:
    value = getattr(status_doc, field, None)
    if value is not None:
        return value
    if isinstance(status_doc, dict):
        return status_doc.get(field, default)
    return default


async def run_finalize_stage(event: PipelineStageEvent, context: Any) -> StageResult:
    rag = await context.get_rag(event.workspace_id, profile=event.stage)
    status_doc = await rag.doc_status.get_by_id(event.doc_id)
    chunks_list = []
    chunks_count = 0
    metadata = {}
    file_path = event.metadata.get("document_name") or event.metadata.get("file_path")
    content_summary = None
    content_length = None
    created_at = None

    if status_doc is not None:
        chunks_list = _status_get(status_doc, "chunks_list", [])
        chunks_count = _status_get(status_doc, "chunks_count", len(chunks_list))
        metadata = _status_get(status_doc, "metadata", {})
        file_path = file_path or _status_get(status_doc, "file_path")
        content_summary = _status_get(status_doc, "content_summary")
        content_length = _status_get(status_doc, "content_length")
        created_at = _status_get(status_doc, "created_at")

    await rag.doc_status.upsert(
        {
            event.doc_id: {
                "status": DocStatus.PROCESSED,
                "error_msg": None,
                "error_stage": None,
                "chunks_count": chunks_count,
                "chunks_list": chunks_list,
                "content_summary": content_summary,
                "content_length": content_length,
                "created_at": created_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": file_path or event.doc_id,
                "metadata": {
                    **metadata,
                    "pipeline_run_id": event.run_id,
                    "pipeline_stage": event.stage_name.value,
                },
            }
        }
    )
    await rag._insert_done()
    return StageResult(
        counts={"chunks": int(chunks_count or 0)},
        message=f"finalized doc_id={event.doc_id}",
    )
