import hashlib
import traceback
from datetime import datetime, timezone
from typing import Any

from lightrag.pipeline.models import PipelineStageEvent, StageName, StageResult
from lightrag.pipeline.stages.common import (
    check_duplicate_by_content_hash,
    get_rag_for_event,
    mark_document_as_duplicated,
    read_file_from_storage,
    read_legal_pdf_from_storage,
    update_document_metadata,
    update_legal_document_metadata,
    write_text_to_storage,
)
from lightrag.pipeline.stages.extractors import extract_content
from lightrag.utils import logger
from lightrag.utils_convert import convert_document
from lightrag.utils_pdf import iter_pdf_grayscale_batches


def _create_markdown_content(content: str, title: str, add_title: bool) -> str:
    """Tạo nội dung Markdown với Frontmatter."""
    current_time = datetime.now(timezone.utc).isoformat()
    frontmatter = f'---\ntitle: "{title}"\ncreated_at: "{current_time}"\n---\n\n'

    cleaned_content = content
    if cleaned_content.startswith("---"):
        end_idx = cleaned_content.find("---", 3)
        if end_idx != -1:
            cleaned_content = cleaned_content[end_idx + 3 :].lstrip()

    if add_title and not cleaned_content.startswith("# "):
        cleaned_content = f"# {title}\n\n{cleaned_content}"

    return frontmatter + cleaned_content


async def run_extraction_stage(event: PipelineStageEvent, context: Any) -> StageResult:
    rag = await get_rag_for_event(context, event, profile=event.stage)
    document_id = event.doc_id
    workspace_id = event.workspace_id
    document_name = (
        event.metadata.get("document_name")
        or event.metadata.get("file_path")
        or document_id
    )
    doc_type = event.metadata.get("doc_type") or getattr(event, "doc_type", "general")

    logger.info(
        f"[Extract Stage] Processing document: {document_id} | "
        f"Workspace: {workspace_id} | Filename: {document_name} | Type: {doc_type}"
    )

    if doc_type == "legal":
        return await _run_legal_extraction_stage(event, document_id, workspace_id, rag)

    return await _run_general_extraction_stage(
        event, document_id, workspace_id, document_name, doc_type, rag
    )


def _legal_source_doc_id(event: PipelineStageEvent, document_id: str) -> str:
    legal_document_id = event.metadata.get("legal_document_id")
    if legal_document_id is not None:
        return str(legal_document_id)
    if document_id.startswith("legal-"):
        return document_id.removeprefix("legal-")
    return document_id


async def _run_general_extraction_stage(
    event: PipelineStageEvent,
    document_id: str,
    workspace_id: str,
    document_name: str,
    doc_type: str,
    rag: Any,
) -> StageResult:
    try:
        # Step 1: Read from MinIO
        file_bytes = await read_file_from_storage(
            workspace_id, doc_type, document_id, document_name
        )
    except Exception as e:
        error_msg = f"Error reading original file from MinIO: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    try:
        # Step 2: OCR / Extract
        if document_name.lower().endswith(".pdf"):
            content = await _extract_pdf_content_by_batches(
                file_bytes,
                document_name,
                pages_per_batch=5,
                dpi=130,
                jpeg_quality=30,
                max_page_width_px=1400,
            )
        else:
            content, _ = await extract_content(file_bytes, document_name)
    except Exception as e:
        error_msg = f"Text extraction failed for file {document_name}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg) from e

    # Step 3: Check duplicates
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
    duplicate_id = await check_duplicate_by_content_hash(content_hash, workspace_id)

    if duplicate_id and duplicate_id != document_id:
        logger.warning(
            f"[Extract Stage] Content duplicate found! This matches document_id: {duplicate_id}"
        )
        await mark_document_as_duplicated(document_id, workspace_id)
        # Finalize since it's duplicate
        next_event = event.next_event(StageName.FINALIZE)
        next_event.status = "DUPLICATED"
        return StageResult(
            message=f"Duplicate of {duplicate_id}", next_events=[next_event]
        )

    # Step 4: Write to full_docs
    title = document_name.rsplit(".", 1)[0]
    md_content = _create_markdown_content(content, title, add_title=True)
    await rag.full_docs.upsert(
        {document_id: {"content": md_content, "doc_type": doc_type}}
    )

    # Step 5: Inject extraction metadata into event for KMS callback
    content_path = f"full_docs/{workspace_id}/{document_id}/{document_id}.md"
    content_summary = (
        content[:200].replace("\n", " ") + "..."
        if len(content) > 200
        else content.replace("\n", " ")
    )
    event.metadata["content_path"] = content_path
    event.metadata["content_hash"] = content_hash
    event.metadata["content_length"] = len(content)
    event.metadata["content_summary"] = content_summary

    await write_text_to_storage(
        content_path,
        md_content,
        content_type="text/markdown; charset=utf-8",
    )

    await update_document_metadata(
        document_id=document_id,
        workspace_id=workspace_id,
        content_path=content_path,
        content_hash=content_hash,
        content_length=len(content),
        content_summary=content_summary,
    )

    next_events = []
    if str(event.metadata.get("status", "")).upper() == "FAILED":
        next_events.append(event.next_event(StageName.CHUNKING))

    logger.info(
        "[Extract Stage] Document extracted; waiting for KMS chunking approval doc_id=%s content_path=%s",
        document_id,
        content_path,
    )
    return StageResult(
        next_events=next_events,
        message="Extraction completed successfully; reported EXTRACTED to KMS",
    )


async def _run_legal_extraction_stage(
    event: PipelineStageEvent,
    document_id: str,
    workspace_id: str,
    rag: Any,
) -> StageResult:
    legal_source_id = _legal_source_doc_id(event, document_id)
    try:
        legal_document_id = int(legal_source_id)
    except (TypeError, ValueError):
        error_msg = (
            "Legal extraction requires numeric legal_document_id "
            f"for tbl_legal_document lookup, got {legal_source_id!r}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    pdf_name = f"law_{legal_source_id}.pdf"

    try:
        file_bytes = await read_legal_pdf_from_storage(legal_source_id)
    except Exception as e:
        error_msg = f"Error reading legal PDF from MinIO: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    try:
        if pdf_name.lower().endswith(".pdf"):
            markdown_content = await _extract_pdf_content_by_batches(
                file_bytes,
                pdf_name,
                pages_per_batch=5,
                dpi=130,
                jpeg_quality=30,
                max_page_width_px=1400,
            )
        else:
            markdown_content, _ = await extract_content(file_bytes, pdf_name)
    except Exception as e:
        error_msg = f"Legal OCR extraction failed for {pdf_name}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg) from e

    html_content = convert_document(markdown_content)

    # Save to MinIO at legal_docs/{workspace_name}/{doc_id}/{doc_id}.html
    workspace_name = event.metadata.get("workspace_name")
    if not workspace_name:
        try:
            import uuid

            clean_id = str(workspace_id).split(":")[-1]
            if len(clean_id.replace("-", "")) == 32:
                workspace_uuid = uuid.UUID(clean_id)
            else:
                workspace_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, str(workspace_id))

            db = getattr(getattr(rag, "doc_status", None), "db", None) or getattr(
                getattr(rag, "full_docs", None), "db", None
            )
            if db is not None:
                row = await db.query(
                    "SELECT name FROM tbl_workspace WHERE id = $1", [workspace_uuid]
                )
                if row and isinstance(row, dict):
                    workspace_name = row.get("name")
        except Exception as e:
            logger.warning(f"Failed to query workspace name from DB: {e}")

    if not workspace_name:
        workspace_name = "unknown"

    content_path = (
        f"legal_docs/{workspace_name}/{legal_source_id}/{legal_source_id}.html"
    )

    await write_text_to_storage(
        content_path,
        html_content,
        content_type="text/html; charset=utf-8",
    )

    # Inject path for KMS Kafka consumer callback synchronization
    event.metadata["content_path"] = content_path

    await update_legal_document_metadata(
        legal_document_id,
        workspace_id,
        content_path,
    )

    next_events = []
    if str(event.metadata.get("status", "")).upper() == "FAILED":
        next_events.append(event.next_event(StageName.CHUNKING))

    logger.info(
        "[Extract Stage] Legal document extracted to HTML; waiting for KMS chunking approval doc_id=%s legal_document_id=%s content_path=%s",
        document_id,
        legal_document_id,
        content_path,
    )
    return StageResult(
        next_events=next_events,
        message="Legal extraction completed successfully; reported EXTRACTED to KMS",
    )


async def _extract_pdf_content_by_batches(
    file_bytes: bytes,
    pdf_name: str,
    *,
    pages_per_batch: int = 5,
    dpi: int = 130,
    jpeg_quality: int = 30,
    max_page_width_px: int = 1400,
) -> str:
    """
    Chuyển toàn bộ PDF sang grayscale, chia batch nhỏ,
    OCR từng batch rồi nối markdown lại.
    """
    contents: list[str] = []

    for batch in iter_pdf_grayscale_batches(
        file_bytes,
        source_filename=pdf_name,
        pages_per_batch=pages_per_batch,
        dpi=dpi,
        jpeg_quality=jpeg_quality,
        max_page_width_px=max_page_width_px,
    ):
        logger.info(
            "[Extract Stage] OCR PDF batch: "
            f"file={batch.filename}, pages={batch.start_page}-{batch.end_page}, "
            f"size={len(batch.content)} bytes"
        )

        batch_content, _ = await extract_content(
            batch.content,
            batch.filename,
        )

        if batch_content and batch_content.strip():
            contents.append(
                "\n\n".join(
                    [
                        f"<!-- OCR batch {batch.index}, pages {batch.start_page}-{batch.end_page} -->",
                        batch_content.strip(),
                    ]
                )
            )

    return "\n\n".join(contents).strip()
