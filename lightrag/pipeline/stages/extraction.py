import hashlib
import traceback
from typing import Any
from datetime import datetime, timezone

from lightrag.utils import logger
from lightrag.pipeline.models import PipelineStageEvent, StageName, StageResult
from lightrag.pipeline.stages.common import (
    read_file_from_storage,
    update_document_metadata,
    check_duplicate_by_content_hash,
    mark_document_as_duplicated,
)
from lightrag.pipeline.stages.extractors import extract_content

def _create_markdown_content(content: str, title: str, add_title: bool) -> str:
    """Tạo nội dung Markdown với Frontmatter."""
    current_time = datetime.now(timezone.utc).isoformat()
    frontmatter = f"---\ntitle: \"{title}\"\ncreated_at: \"{current_time}\"\n---\n\n"
    
    cleaned_content = content
    if cleaned_content.startswith("---"):
        end_idx = cleaned_content.find("---", 3)
        if end_idx != -1:
            cleaned_content = cleaned_content[end_idx + 3 :].lstrip()

    if add_title and not cleaned_content.startswith("# "):
        cleaned_content = f"# {title}\n\n{cleaned_content}"

    return frontmatter + cleaned_content

async def run_extraction_stage(event: PipelineStageEvent, context: Any) -> StageResult:
    rag = await context.get_rag(event.workspace_id, profile=event.stage)
    document_id = event.doc_id
    workspace_id = event.workspace_id
    document_name = event.metadata.get("document_name") or event.metadata.get("file_path") or document_id
    doc_type = getattr(event, "doc_type", "general")
    
    logger.info(
        f"[Extract Stage] Processing document: {document_id} | "
        f"Workspace: {workspace_id} | Filename: {document_name} | Type: {doc_type}"
    )

    try:
        # Step 1: Read from MinIO
        file_bytes = await read_file_from_storage(workspace_id, doc_type, document_id, document_name)
    except Exception as e:
        error_msg = f"Error reading original file from MinIO: {e}"
        logger.error(error_msg)
        return StageResult(message=error_msg, next_events=[])

    try:
        # Step 2: OCR / Extract
        content, _ = await extract_content(file_bytes, document_name)
    except Exception as e:
        error_msg = f"Text extraction failed for file {document_name}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return StageResult(message=error_msg, next_events=[])

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
        return StageResult(message=f"Duplicate of {duplicate_id}", next_events=[next_event])

    # Step 4: Write to full_docs
    title = document_name.rsplit(".", 1)[0]
    md_content = _create_markdown_content(content, title, add_title=True)
    await rag.full_docs.set_by_id(document_id, {"content": md_content, "doc_type": doc_type})

    # Step 5: Update metadata in Postgres
    content_summary = content[:200].replace("\n", " ") + "..." if len(content) > 200 else content.replace("\n", " ")
    await update_document_metadata(
        document_id=document_id,
        workspace_id=workspace_id,
        content_path=f"full_docs/{workspace_id}/{document_id}/{document_id}.md",
        content_hash=content_hash,
        content_length=len(content),
        content_summary=content_summary,
    )

    # Move to chunking
    next_event = event.next_event(StageName.CHUNKING)
    return StageResult(next_events=[next_event], message="Extraction completed successfully")
