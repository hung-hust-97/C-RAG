"""
This module contains all document-related routes for the LightRAG API.
"""

import asyncio
import base64
import contextvars
import inspect
import os
import textwrap
from functools import lru_cache, wraps
from lightrag.utils import logger, get_pinyin_sort_key
import aiofiles
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal, Callable, Awaitable
from io import BytesIO
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lightrag import LightRAG
from lightrag.base import DeletionResult, DocProcessingStatus, DocStatus
from lightrag.utils import (
    compute_mdhash_id,
    sanitize_text_for_encoding,
    get_content_summary,
)
from lightrag.api.reembed import reembed_workspace_vectors
from lightrag.api.utils_api import get_combined_auth_dependency
from ..config import global_args
from lightrag.ocr import OCRConfig, process_document_with_ocr
from lightrag.ocr.markdown import MarkdownPreprocessor


@lru_cache(maxsize=1)
def _is_docling_available() -> bool:
    """Check if docling is available (cached check).

    This function uses lru_cache to avoid repeated import attempts.
    The result is cached after the first call.

    Returns:
        bool: True if docling is available, False otherwise
    """
    try:
        import docling  # noqa: F401  # type: ignore[import-not-found]

        return True
    except ImportError:
        return False

async def generate_title_from_text(text: str, rag_instance: LightRAG, max_length: int = 100) -> str:
    """Generate a concise title from text content using LLM.

    Args:
        text: The text content to generate title from
        rag_instance: LightRAG instance to use for LLM calls
        max_length: Maximum length of the generated title

    Returns:
        str: Generated title, sanitized for use as filename
    """
    try:
        # Truncate text if too long (use first 1000 chars for title generation)
        text_preview = text[:1000] if len(text) > 1000 else text
        
        # Create prompt for title generation
        system_prompt = "You are a helpful assistant that generates concise, descriptive titles for documents."
        user_prompt = f"""Based on the following text content, generate a short, descriptive title (maximum {max_length} characters).
The title should capture the main topic or theme of the text.
Return ONLY the title text, without quotes or additional formatting.

Text content:
{text_preview}"""

        # Call LLM to generate title with timeout
        response = await asyncio.wait_for(
            rag_instance.llm_model_func(
                user_prompt,
                system_prompt=system_prompt,
                max_tokens=50,
            ),
            timeout=10.0  # 10 second timeout for title generation
        )
        
        # Clean up the response
        title = response.strip().strip('"').strip("'")
        
        # Limit length
        if len(title) > max_length:
            title = title[:max_length].rsplit(' ', 1)[0]  # Cut at last word boundary
        
        # Sanitize for filename use
        # Remove invalid filename characters
        invalid_chars = '<>:"/\\|?*\n\r\t'
        for char in invalid_chars:
            title = title.replace(char, ' ')
        
        # Replace multiple spaces with single space
        title = ' '.join(title.split())
        
        # Ensure title is not empty
        if not title:
            title = "Untitled Document"
        
        return title
        
    except asyncio.TimeoutError:
        logger.warning("Timeout generating title from text after 10s, using fallback")
        # Fallback: use first few words of text
        words = text.split()[:10]
        fallback_title = ' '.join(words)
        if len(fallback_title) > max_length:
            fallback_title = fallback_title[:max_length].rsplit(' ', 1)[0]
        return fallback_title if fallback_title else "Untitled Document"
    except Exception as e:
        logger.warning(f"Failed to generate title from text: {str(e)}")
        # Fallback: use first few words of text
        words = text.split()[:10]
        fallback_title = ' '.join(words)
        if len(fallback_title) > max_length:
            fallback_title = fallback_title[:max_length].rsplit(' ', 1)[0]
        return fallback_title if fallback_title else "Untitled Document"


# Function to format datetime to ISO format string with timezone information
def format_datetime(dt: Any) -> Optional[str]:
    """Format datetime to ISO format string with timezone information

    Args:
        dt: Datetime object, string, or None

    Returns:
        ISO format string with timezone information, or None if input is None
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        return dt

    # Check if datetime object has timezone information
    if isinstance(dt, datetime):
        # If datetime object has no timezone info (naive datetime), add UTC timezone
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

    # Return ISO format string with timezone information
    return dt.isoformat()


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)

# Temporary file prefix
temp_prefix = "__tmp__"
UNKNOWN_FILE_SOURCE = "unknown_source"
LEGACY_EMPTY_FILE_PATH_SENTINELS = {"", "no-file-path"}


def normalize_file_path(file_path: str | None) -> str:
    """Normalize missing document sources to a single non-null sentinel."""
    if file_path is None:
        return UNKNOWN_FILE_SOURCE

    normalized = file_path.strip()
    if normalized in LEGACY_EMPTY_FILE_PATH_SENTINELS:
        return UNKNOWN_FILE_SOURCE

    return normalized


def sanitize_filename(filename: str, input_dir: Path) -> str:
    """
    Sanitize uploaded filename to prevent Path Traversal attacks.

    Args:
        filename: The original filename from the upload
        input_dir: The target input directory

    Returns:
        str: Sanitized filename that is safe to use

    Raises:
        HTTPException: If the filename is unsafe or invalid
    """
    # Basic validation
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    # Remove path separators and traversal sequences
    clean_name = filename.replace("/", "").replace("\\", "")
    clean_name = clean_name.replace("..", "")

    # Remove control characters and null bytes
    clean_name = "".join(c for c in clean_name if ord(c) >= 32 and c != "\x7f")

    # Remove leading/trailing whitespace and dots
    clean_name = clean_name.strip().strip(".")

    # Check if anything is left after sanitization
    if not clean_name:
        raise HTTPException(
            status_code=400, detail="Invalid filename after sanitization"
        )

    # Verify the final path stays within the input directory
    try:
        final_path = (input_dir / clean_name).resolve()
        if not final_path.is_relative_to(input_dir.resolve()):
            raise HTTPException(status_code=400, detail="Unsafe filename detected")
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid filename")

    return clean_name


class ScanResponse(BaseModel):
    """Response model for document scanning operation

    Attributes:
        status: Status of the scanning operation
        message: Optional message with additional details
        doc_ids: Array of document IDs for monitoring scanning progress
        total_files: Total number of files found to process
        queued_files: Number of files successfully queued for processing
        failed_files: Number of files that failed to queue
    """

    status: Literal["scanning_started"] = Field(
        description="Status of the scanning operation"
    )
    message: Optional[str] = Field(
        default=None, description="Additional details about the scanning operation"
    )
    workspace_id: str = Field(description="Workspace identifier for this scan request")
    doc_ids: List[str] = Field(description="Array of document IDs for monitoring scanning progress")
    total_files: int = Field(description="Total number of files found to process")
    queued_files: int = Field(description="Number of files successfully queued for processing")
    failed_files: int = Field(description="Number of files that failed to queue")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "scanning_started",
                "message": "Scanning process has been initiated in the background",
                "workspace_id": "default",
                "doc_ids": ["doc-550e8400-e29b-41d4-a716-446655440000", "doc-660e8400-e29b-41d4-a716-446655440001"],
                "total_files": 2,
                "queued_files": 2,
                "failed_files": 0,
            }
        }
    )


class ReprocessResponse(BaseModel):
    """Response model for reprocessing failed documents operation

    Attributes:
        status: Status of the reprocessing operation
        message: Message describing the operation result
        documents_count: Number of documents that will be reprocessed
    """

    status: Literal["reprocessing_started"] = Field(
        description="Status of the reprocessing operation"
    )
    message: str = Field(description="Human-readable message describing the operation")
    documents_count: int = Field(
        description="Number of documents that will be reprocessed",
        default=0
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "reprocessing_started",
                "message": "Reprocessing of 5 failed documents has been initiated in background",
                "documents_count": 5
            }
        }
    )


class CancelPipelineResponse(BaseModel):
    """Response model for pipeline cancellation operation

    Attributes:
        status: Status of the cancellation request
        message: Message describing the operation result
    """

    status: Literal["cancellation_requested", "not_busy"] = Field(
        description="Status of the cancellation request"
    )
    message: str = Field(description="Human-readable message describing the operation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "cancellation_requested",
                "message": "Pipeline cancellation has been requested. Documents will be marked as FAILED.",
            }
        }
    )


class InsertTextRequest(BaseModel):
    """Request model for inserting a single text document

    Attributes:
        text: The text content to be inserted into the RAG system
        workspace_id: Workspace identifier (required)
        file_source: Source of the text (optional)
        doc_type: Document type classification ("legal" or "general", default: "general")
    """

    text: str = Field(
        min_length=1,
        description="The text to insert",
    )
    workspace_id: str = Field(
        description="Workspace identifier for this request (required)"
    )
    file_source: Optional[str] = Field(
        default=None, min_length=0, description="File Source"
    )
    doc_type: Literal["legal", "general"] = Field(
        default="general", description="Document type classification"
    )

    @field_validator("text", mode="after")
    @classmethod
    def strip_text_after(cls, text: str) -> str:
        return text.strip()

    @field_validator("file_source", mode="before")
    @classmethod
    def normalize_source_before(cls, file_source: Optional[str]) -> str:
        return normalize_file_path(file_source)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "This is a sample text to be inserted into the RAG system.",
                "workspace_id": "my-workspace-id",
                "file_source": "Source of the text (optional)",
                "doc_type": "general",
            }
        }
    )


class InsertTextsRequest(BaseModel):
    """Request model for inserting multiple text documents

    Attributes:
        texts: List of text contents to be inserted into the RAG system
        workspace_id: Workspace identifier (required)
        file_sources: Sources of the texts (optional)
        doc_type: Document type classification ("legal" or "general", default: "general")
    """

    texts: list[str] = Field(
        min_length=1,
        description="The texts to insert",
    )
    workspace_id: str = Field(
        description="Workspace identifier for this request (required)"
    )
    file_sources: Optional[list[str]] = Field(
        default=None, min_length=0, description="Sources of the texts"
    )
    doc_type: Literal["legal", "general"] = Field(
        default="general", description="Document type classification"
    )

    @field_validator("texts", mode="after")
    @classmethod
    def strip_texts_after(cls, texts: list[str]) -> list[str]:
        return [text.strip() for text in texts]

    @field_validator("file_sources", mode="before")
    @classmethod
    def normalize_sources_before(
        cls, file_sources: Optional[list[str]]
    ) -> Optional[list[str]]:
        if file_sources is None:
            return None

        return [normalize_file_path(file_source) for file_source in file_sources]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "This is the first text to be inserted.",
                    "This is the second text to be inserted.",
                ],
                "workspace_id": "my-workspace-id",
                "file_sources": [
                    "First file source (optional)",
                ],
                "doc_type": "general",
            }
        }
    )


class InsertResponse(BaseModel):
    """Response model for document insertion operations

    Attributes:
        status: Status of the operation (success, duplicated, partial_success, failure)
        message: Detailed message describing the operation result
        doc_id: Document ID for monitoring processing status
    """

    status: Literal["success", "duplicated", "partial_success", "failure"] = Field(
        description="Status of the operation"
    )
    message: str = Field(description="Message describing the operation result")
    workspace_id: str = Field(description="Workspace identifier for this operation")
    doc_id: str = Field(description="Document ID for monitoring processing status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "File 'document.pdf' uploaded successfully. Processing will continue in background.",
                "doc_id": "doc-550e8400-e29b-41d4-a716-446655440000",
            }
        }
    )


class BatchInsertResponse(BaseModel):
    """Response model for batch document insertion operations

    Attributes:
        status: Status of the operation (success, partial_success, failure)
        message: Detailed message describing the operation result
        workspace_id: Workspace identifier for this operation
        doc_ids: Array of document IDs for monitoring processing status
        total_files: Total number of files/texts in the batch
        queued_files: Number of files/texts successfully queued for processing
        failed_files: Number of files/texts that failed to queue
    """

    status: Literal["success", "partial_success", "failure"] = Field(
        description="Status of the operation"
    )
    message: str = Field(description="Message describing the operation result")
    workspace_id: str = Field(description="Workspace identifier for this operation")
    doc_ids: List[str] = Field(description="Array of document IDs for monitoring processing status")
    total_files: int = Field(description="Total number of files/texts in the batch")
    queued_files: int = Field(description="Number of files/texts successfully queued for processing")
    failed_files: int = Field(description="Number of files/texts that failed to queue")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "3 texts successfully queued for processing.",
                "workspace_id": "default",
                "doc_ids": ["doc-550e8400-e29b-41d4-a716-446655440000", "doc-660e8400-e29b-41d4-a716-446655440001", "doc-770e8400-e29b-41d4-a716-446655440002"],
                "total_files": 3,
                "queued_files": 3,
                "failed_files": 0,
            }
        }
    )


class PreviewDocumentResponse(BaseModel):
    """Response model for document preview."""

    workspace_id: str = Field(description="Workspace identifier for this request")
    file_path: str = Field(description="Original file path")
    file_name: str = Field(description="Rendered file name")
    mime_type: str = Field(description="MIME type (e.g. application/pdf, text/markdown)")
    data_base64: Optional[str] = Field(default=None, description="Binary data encoded as base64")
    data_text: Optional[str] = Field(default=None, description="Plain text data (for markdown/text)")


class ClearDocumentsResponse(BaseModel):
    """Response model for document clearing operation

    Attributes:
        status: Status of the clear operation
        message: Detailed message describing the operation result
    """

    status: Literal["success", "partial_success", "busy", "fail"] = Field(
        description="Status of the clear operation"
    )
    workspace_id: str = Field(description="Workspace identifier for this operation")
    message: str = Field(description="Message describing the operation result")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "All documents cleared successfully. Deleted 15 files.",
            }
        }
    )


class ClearCacheRequest(BaseModel):
    """Request model for clearing cache

    This model is kept for API compatibility but no longer accepts any parameters.
    All cache will be cleared regardless of the request content.
    """

    model_config = ConfigDict(json_schema_extra={"example": {}})


class ClearCacheResponse(BaseModel):
    """Response model for cache clearing operation

    Attributes:
        status: Status of the clear operation
        message: Detailed message describing the operation result
    """

    status: Literal["success", "fail"] = Field(
        description="Status of the clear operation"
    )
    message: str = Field(description="Message describing the operation result")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "success",
                "message": "Successfully cleared cache for modes: ['default', 'naive']",
            }
        }
    )


"""Response model for document status

Attributes:
    id: Document identifier
    content_summary: Summary of document content
    content_length: Length of document content
    status: Current processing status
    created_at: Creation timestamp (ISO format string)
    updated_at: Last update timestamp (ISO format string)
    chunks_count: Number of chunks (optional)
    error: Error message if any (optional)
    metadata: Additional metadata (optional)
    file_path: Path to the document file
"""


class DeleteDocRequest(BaseModel):
    workspace_id: str | None = Field(
        default=None,
        description="Workspace ID that owns the documents. Optional for backward compatibility.",
    )
    doc_ids: List[str] = Field(..., description="The IDs of the documents to delete.")
    delete_file: bool = Field(
        default=False,
        description="Whether to delete the corresponding file in the upload directory.",
    )
    delete_llm_cache: bool = Field(
        default=False,
        description="Whether to delete cached LLM extraction results for the documents.",
    )

    @field_validator("doc_ids", mode="after")
    @classmethod
    def validate_doc_ids(cls, doc_ids: List[str]) -> List[str]:
        if not doc_ids:
            raise ValueError("Document IDs list cannot be empty")

        validated_ids = []
        for doc_id in doc_ids:
            if not doc_id or not doc_id.strip():
                raise ValueError("Document ID cannot be empty")
            validated_ids.append(doc_id.strip())

        # Check for duplicates
        if len(validated_ids) != len(set(validated_ids)):
            raise ValueError("Document IDs must be unique")

        return validated_ids

    @field_validator("workspace_id", mode="after")
    @classmethod
    def validate_workspace_id(cls, workspace_id: str | None) -> str | None:
        if workspace_id is None:
            return None
        normalized_workspace_id = workspace_id.strip()
        return normalized_workspace_id or None

    class Config:
        json_schema_extra = {
            "example": {
                "workspace_id": "default",
                "doc_ids": ["doc_123", "doc_456"],
                "delete_file": False,
                "delete_llm_cache": False,
            }
        }


class DeleteEntityRequest(BaseModel):
    entity_name: str = Field(..., description="The name of the entity to delete.")

    @field_validator("entity_name", mode="after")
    @classmethod
    def validate_entity_name(cls, entity_name: str) -> str:
        if not entity_name or not entity_name.strip():
            raise ValueError("Entity name cannot be empty")
        return entity_name.strip()


class DeleteRelationRequest(BaseModel):
    source_entity: str = Field(..., description="The name of the source entity.")
    target_entity: str = Field(..., description="The name of the target entity.")

    @field_validator("source_entity", "target_entity", mode="after")
    @classmethod
    def validate_entity_names(cls, entity_name: str) -> str:
        if not entity_name or not entity_name.strip():
            raise ValueError("Entity name cannot be empty")
        return entity_name.strip()


class ReembedRequest(BaseModel):
    workspace_id: str = Field(
        ..., description="Workspace ID for re-embedding existing vectors."
    )
    batch_size: int = Field(
        default=256,
        ge=1,
        le=5000,
        description="Batch size for re-embedding vectors.",
    )

    @field_validator("workspace_id", mode="after")
    @classmethod
    def validate_workspace_id(cls, workspace_id: str) -> str:
        normalized_workspace_id = workspace_id.strip() if workspace_id else ""
        if not normalized_workspace_id:
            raise ValueError("workspace_id cannot be empty")
        return normalized_workspace_id


class ReembedResponse(BaseModel):
    status: Literal["success", "busy", "fail"] = Field(
        description="Status of re-embedding operation"
    )
    message: str = Field(description="Operation result message")
    workspace_id: str = Field(description="Workspace where re-embedding was executed")
    documents: int = Field(default=0, description="Number of processed documents")
    chunks: int = Field(default=0, description="Number of re-embedded chunks")
    entities: int = Field(default=0, description="Number of re-embedded entities")
    relations: int = Field(default=0, description="Number of re-embedded relations")


class DocStatusResponse(BaseModel):
    id: str = Field(description="Document identifier")
    content_summary: Optional[str] = Field(default="", description="Summary of document content")
    content_length: Optional[int] = Field(default=0, description="Length of document content in characters")
    status: DocStatus = Field(description="Current processing status")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp (ISO format string)")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp (ISO format string)")
    workspace_id: str = Field(description="Workspace identifier for this document")
    chunks_count: Optional[int] = Field(
        default=None, description="Number of chunks the document was split into"
    )
    error_msg: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Additional metadata about the document"
    )
    file_path: str = Field(description="Path to the document file")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "doc-550e8400-e29b-41d4-a716-446655440000",
                "content_summary": "Research paper on machine learning",
                "content_length": 15240,
                "status": "processed",
                "created_at": "2025-03-31T12:34:56",
                "updated_at": "2025-03-31T12:35:30",
                "chunks_count": 12,
                "error": None,
                "metadata": {"author": "John Doe", "year": 2025},
                "file_path": "research_paper.pdf",
            }
        }
    )


class DocsStatusesResponse(BaseModel):
    """Response model for document statuses

    Attributes:
        statuses: Dictionary mapping document status to lists of document status responses
    """

    statuses: Dict[DocStatus, List[DocStatusResponse]] = Field(
        default_factory=dict,
        description="Dictionary mapping document status to lists of document status responses",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "statuses": {
                    "EXTRACTED": [
                        {
                            "id": "doc-550e8400-e29b-41d4-a716-446655440000",
                            "content_summary": "Extracted document",
                            "content_length": 5000,
                            "status": "extracted",
                            "created_at": "2025-03-31T10:00:00",
                            "updated_at": "2025-03-31T10:00:00",
                            "chunks_count": None,
                            "error": None,
                            "metadata": None,
                            "file_path": "extracted_doc.pdf",
                        }
                    ],

                    "PROCESSED": [
                        {
                            "id": "doc-770e8400-e29b-41d4-a716-446655440002",
                            "content_summary": "Processed document",
                            "content_length": 8000,
                            "status": "processed",
                            "created_at": "2025-03-31T09:00:00",
                            "updated_at": "2025-03-31T09:05:00",
                            "chunks_count": 8,
                            "error": None,
                            "metadata": {"author": "John Doe"},
                            "file_path": "processed_doc.pdf",
                        }
                    ],
                }
            }
        }
    )


# TrackStatusResponse removed - track_id functionality has been deprecated
# Use individual document queries with doc_id instead


class DocumentsRequest(BaseModel):
    """Request model for paginated document queries

    Attributes:
        status_filter: Filter by document status, None for all statuses
        page: Page number (1-based)
        page_size: Number of documents per page (10-200)
        sort_field: Field to sort by ('created_at', 'updated_at', 'id', 'file_path')
        sort_direction: Sort direction ('asc' or 'desc')
    """

    status_filter: Optional[DocStatus] = Field(
        default=None, description="Filter by document status, None for all statuses"
    )
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(
        default=50, ge=10, le=200, description="Number of documents per page (10-200)"
    )
    sort_field: Literal["created_at", "updated_at", "id", "file_path"] = Field(
        default="updated_at", description="Field to sort by"
    )
    sort_direction: Literal["asc", "desc"] = Field(
        default="desc", description="Sort direction"
    )
    workspace_id: Optional[str] = Field(
        default=None, description="Workspace identifier for this request"
    )

    @field_validator("status_filter", mode="before")
    @classmethod
    def validate_status_filter(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            v_upper = v.upper().strip()
            # Special handling for common UI values if needed
            return v_upper
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status_filter": "PROCESSED",
                "page": 1,
                "page_size": 50,
                "sort_field": "updated_at",
                "sort_direction": "desc",
            }
        }
    )


class DocumentSearchByNameRequest(BaseModel):
    """Request model for searching documents by name substring."""

    workspace_id: Optional[str] = Field(
        default=None, description="Workspace identifier for this request"
    )
    keyword: str = Field(
        ...,
        min_length=1,
        description="Substring to search within document name (stored as file_path in doc status).",
    )
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    page_size: int = Field(
        default=50, ge=10, le=200, description="Number of results per page (10-200)"
    )


class PaginationInfo(BaseModel):
    """Pagination information

    Attributes:
        page: Current page number
        page_size: Number of items per page
        total_count: Total number of items
        total_pages: Total number of pages
        has_next: Whether there is a next page
        has_prev: Whether there is a previous page
    """

    page: int = Field(description="Current page number")
    page_size: int = Field(description="Number of items per page")
    total_count: int = Field(description="Total number of items")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there is a next page")
    has_prev: bool = Field(description="Whether there is a previous page")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "page": 1,
                "page_size": 50,
                "total_count": 150,
                "total_pages": 3,
                "has_next": True,
                "has_prev": False,
            }
        }
    )


class DocumentSearchByNameResponse(BaseModel):
    """Response model for searching documents by name substring."""

    workspace_id: str = Field(description="Resolved workspace identifier")
    keyword: str = Field(description="Searched keyword (trimmed)")
    total_matches: int = Field(description="Total number of matched documents")
    pagination: PaginationInfo = Field(description="Pagination info over matches")
    documents: List[DocStatusResponse] = Field(
        description="List of matched documents for the requested page"
    )


class PaginatedDocsResponse(BaseModel):
    """Response model for paginated document queries

    Attributes:
        documents: List of documents for the current page
        pagination: Pagination information
        status_counts: Count of documents by status for all documents
    """

    documents: List[DocStatusResponse] = Field(
        description="List of documents for the current page"
    )
    workspace_id: str = Field(description="Workspace identifier for this request")
    pagination: PaginationInfo = Field(description="Pagination information")
    status_counts: Dict[str, int] = Field(
        description="Count of documents by status for all documents"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": [
                    {
                        "id": "doc_123456",
                        "content_summary": "Research paper on machine learning",
                        "content_length": 15240,
                        "status": "PROCESSED",
                        "created_at": "2025-03-31T12:34:56",
                        "updated_at": "2025-03-31T12:35:30",
                        "chunks_count": 12,
                        "error_msg": None,
                        "metadata": {"author": "John Doe", "year": 2025},
                        "file_path": "research_paper.pdf",
                    }
                ],
                "pagination": {
                    "page": 1,
                    "page_size": 50,
                    "total_count": 150,
                    "total_pages": 3,
                    "has_next": True,
                    "has_prev": False,
                },
                "status_counts": {
                    "EXTRACTED": 10,
                    "PROCESSED": 135,
                    "FAILED": 5,
                },
            }
        }
    )


class StatusCountsResponse(BaseModel):
    """Response model for document status counts

    Attributes:
        workspace_id: Workspace identifier for this request
        status_counts: Count of documents by status
        workspaces: Aggregated stats for all workspaces (only if workspace_id was not specified)
    """

    workspace_id: str = Field(description="Workspace identifier for this request")
    status_counts: Dict[str, int] = Field(description="Count of documents by status")
    workspaces: Optional[Dict[str, Dict[str, int]]] = Field(
        default=None, description="Aggregated stats for all workspaces"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status_counts": {
                    "PROCESSED": 135,
                    "FAILED": 5,
                }
            }
        }
    )


class ReprocessRequest(BaseModel):
    """Request model for reprocessing specific documents"""

    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document IDs to reprocess. If not provided, all failed/stuck documents in the workspace are reprocessed."
    )
    doc_type: Optional[str] = Field(
        default=None,
        description="Optional doc_type filter ('legal' or 'general') to reprocess only failed documents of this type."
    )


class PipelineStageRunsRequest(BaseModel):
    """Request model for retrieving pipeline stage runs"""

    workspace_id: Optional[str] = Field(default=None, description="Workspace ID")
    stage: Optional[str] = Field(default=None, description="Filter by stage name")
    status: Optional[list[str]] = Field(default=None, description="Filter by status list (e.g. ['running', 'failed'])")
    search: Optional[str] = Field(default=None, description="Search term for document title or ID")
    doc_type: Optional[Literal["all", "legal", "general"]] = Field(
        default=None,
        description="Filter by document type ('all', 'legal', 'general')"
    )
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=500, description="Items per page")


class PipelineStageRunResponse(BaseModel):
    """Model representing a single pipeline stage run"""

    run_id: str = Field(description="Run ID")
    stage: str = Field(description="Stage name")
    shard_index: int = Field(description="Shard index")
    shard_count: int = Field(description="Shard count")
    workspace_id: str = Field(description="Workspace ID")
    doc_id: str = Field(description="Document ID")
    document_name: Optional[str] = Field(default=None, description="Document name / title")
    doc_type: Optional[str] = Field(default="general", description="Document type ('legal' or 'general')")
    status: str = Field(description="Status of the stage run")
    attempt: int = Field(description="Attempt number")
    error_json: Optional[Any] = Field(default=None, description="Error JSON structure if failed")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


class PaginatedPipelineStageRunsResponse(BaseModel):
    """Paginated list of pipeline stage runs with counts matrix"""

    runs: List[PipelineStageRunResponse] = Field(description="List of stage runs")
    total: int = Field(description="Total number of runs matching filters")
    page: int = Field(description="Current page")
    page_size: int = Field(description="Page size")
    status_matrix: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Nested dictionary of counts grouped by stage and status"
    )
    doc_type_counts: Optional[Dict[str, int]] = Field(
        default_factory=dict,
        description="Counts by doc_type: {'all': X, 'legal': Y, 'general': Z}"
    )


class ApproveStageRequest(BaseModel):
    """Request model for approving a paused pipeline stage"""

    run_id: str = Field(description="Run ID")
    stage: str = Field(description="Stage name")
    shard_index: int = Field(default=0, description="Shard index")
    workspace_id: Optional[str] = Field(default=None, description="Workspace ID")


class ApproveStageResponse(BaseModel):
    """Response model for stage approval"""

    status: str = Field(description="Response status")
    message: str = Field(description="Message detailing the result of the action")


class ReplayStageRequest(BaseModel):
    """Request model for an audited replay of a failed pipeline stage."""

    run_id: str = Field(min_length=1, description="Pipeline run ID")
    stage: str = Field(min_length=1, description="Failed stage name")
    shard_index: int = Field(default=0, ge=0, description="Failed shard index")
    workspace_id: Optional[str] = Field(default=None, description="Workspace ID")
    reason: str = Field(
        min_length=3,
        max_length=500,
        description="Operator reason for replay",
    )

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 3:
            raise ValueError("reason must contain at least three non-space characters")
        return normalized


class ReplayStageResponse(BaseModel):
    status: str = Field(description="Replay status")
    message: str = Field(description="Replay result")
    event_id: str = Field(description="Queued replay event ID")
    attempt: int = Field(description="Attempt number for the replay event")


class GenerationStartRequest(BaseModel):
    workspace_id: Optional[str] = Field(default=None, description="Workspace ID")
    generation_id: Optional[str] = Field(
        default=None, description="Optional explicit generation ID"
    )
    reason: str = Field(
        min_length=3,
        max_length=500,
        description="Operator reason for starting shadow generation rebuild",
    )


class GenerationActionRequest(BaseModel):
    workspace_id: Optional[str] = Field(default=None, description="Workspace ID")
    reason: str = Field(
        min_length=3,
        max_length=500,
        description="Operator reason for cutover or rollback",
    )


class GenerationStatusResponse(BaseModel):
    workspace_id: str
    generation_id: str
    storage_workspace_id: str
    status: str
    snapshot_complete: Optional[bool] = None
    expected_documents: Optional[int] = None
    pending_documents: Optional[int] = None
    running_documents: Optional[int] = None
    completed_documents: Optional[int] = None
    failed_documents: Optional[int] = None



class BulkApproveStageRequest(BaseModel):
    """Request model for bulk-approving all paused runs of a specific stage"""

    stage: str = Field(description="Stage name to approve all paused runs for")
    workspace_id: Optional[str] = Field(default=None, description="Workspace ID")


class StageSpeedEntry(BaseModel):
    """Speed statistics for a single stage"""

    stage: str
    docs_completed_1m: int = Field(default=0, description="Documents completed in the last 1 minute")
    docs_completed_5m: int = Field(default=0, description="Documents completed in the last 5 minutes")
    avg_duration_seconds: Optional[float] = Field(default=None, description="Average processing duration in seconds")


class PipelineConfigResponse(BaseModel):
    """Current pipeline configuration"""

    entity_extraction_model: str = Field(default="", description="Entity extraction LLM model name")
    entity_extraction_host: str = Field(default="", description="Entity extraction LLM host(s)")
    entity_extraction_api_key: str = Field(default="", description="Entity extraction LLM API key(s)")
    paused_stages: str = Field(default="", description="Comma-separated paused stage names")


class PipelineConfigUpdateRequest(BaseModel):
    """Request to update pipeline configuration"""

    entity_extraction_model: Optional[str] = Field(default=None, description="Entity extraction model")
    entity_extraction_host: Optional[str] = Field(default=None, description="Entity extraction host(s)")
    entity_extraction_api_key: Optional[str] = Field(default=None, description="Entity extraction API key(s)")
    paused_stages: Optional[str] = Field(default=None, description="Comma-separated paused stages")
    restart_workers: bool = Field(default=True, description="Auto-restart workers after update")


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status

    Attributes:
        autoscanned: Whether auto-scan has started
        busy: Whether the pipeline is currently busy
        job_name: Current job name (e.g., indexing files/indexing texts)
        job_start: Job start time as ISO format string with timezone (optional)
        docs: Total number of documents to be indexed
        batchs: Number of batches for processing documents
        cur_batch: Current processing batch
        request_pending: Flag for pending request for processing
        latest_message: Latest message from pipeline processing
        history_messages: List of history messages
        update_status: Status of update flags for all namespaces
    """

    autoscanned: bool = False
    busy: bool = False
    job_name: str = "Default Job"
    job_start: Optional[str] = None
    docs: int = 0
    batchs: int = 0
    cur_batch: int = 0
    request_pending: bool = False
    latest_message: str = ""
    workspace_id: str = ""
    history_messages: Optional[List[str]] = None
    update_status: Optional[dict] = None

    @field_validator("job_start", mode="before")
    @classmethod
    def parse_job_start(cls, value):
        """Process datetime and return as ISO format string with timezone"""
        return format_datetime(value)

    model_config = ConfigDict(extra="allow")


class DocumentManager:
    def __init__(
        self,
        input_dir: str,
        workspace: str = "",  # New parameter for workspace isolation
        supported_extensions: tuple = (
            ".txt",
            ".md",
            ".mdx",  # MDX (Markdown + JSX)
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".rtf",  # Rich Text Format
            ".odt",  # OpenDocument Text
            ".tex",  # LaTeX
            ".epub",  # Electronic Publication
            ".html",  # HyperText Markup Language
            ".htm",  # HyperText Markup Language
            ".csv",  # Comma-Separated Values
            ".json",  # JavaScript Object Notation
            ".xml",  # eXtensible Markup Language
            ".yaml",  # YAML Ain't Markup Language
            ".yml",  # YAML
            ".log",  # Log files
            ".conf",  # Configuration files
            ".ini",  # Initialization files
            ".properties",  # Java properties files
            ".sql",  # SQL scripts
            ".bat",  # Batch files
            ".sh",  # Shell scripts
            ".c",  # C source code
            ".h",  # C header
            ".cpp",  # C++ source code
            ".hpp",  # C++ header
            ".py",  # Python source code
            ".java",  # Java source code
            ".js",  # JavaScript source code
            ".ts",  # TypeScript source code
            ".swift",  # Swift source code
            ".go",  # Go source code
            ".rb",  # Ruby source code
            ".php",  # PHP source code
            ".css",  # Cascading Style Sheets
            ".scss",  # Sassy CSS
            ".less",  # LESS CSS
            ".jpg",   # JPEG Image
            ".jpeg",  # JPEG Image
            ".png",   # PNG Image
            ".webp",  # WebP Image
            ".tiff",  # TIFF Image
            ".bmp",   # Bitmap Image
        ),
    ):
        # Store the base input directory and workspace
        self.base_input_dir = Path(input_dir)
        self.workspace = workspace
        self.supported_extensions = supported_extensions
        self.indexed_files = set()

        # Create workspace-specific input directory
        # If workspace is provided, create a subdirectory for data isolation
        if workspace:
            self.input_dir = self.base_input_dir / workspace
        else:
            self.input_dir = self.base_input_dir

        # Create input directory if it doesn't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def scan_directory_for_new_files(self) -> List[Path]:
        """Scan input directory for new files"""
        new_files = []
        for ext in self.supported_extensions:
            logger.debug(f"Scanning for {ext} files in {self.input_dir}")
            for file_path in self.input_dir.glob(f"*{ext}"):
                if file_path not in self.indexed_files:
                    new_files.append(file_path)
        return new_files

    def mark_as_indexed(self, file_path: Path):
        self.indexed_files.add(file_path)

    def is_supported_file(self, filename: str) -> bool:
        return any(filename.lower().endswith(ext) for ext in self.supported_extensions)


def validate_file_path_security(file_path_str: str, base_dir: Path) -> Optional[Path]:
    """
    Validate file path security to prevent Path Traversal attacks.

    Args:
        file_path_str: The file path string to validate
        base_dir: The base directory that the file must be within

    Returns:
        Path: Safe file path if valid, None if unsafe or invalid
    """
    if not file_path_str or not file_path_str.strip():
        return None

    try:
        # Clean the file path string
        clean_path_str = file_path_str.strip()

        # Check for obvious path traversal patterns before processing
        # This catches both Unix (..) and Windows (..\) style traversals
        if ".." in clean_path_str:
            # Additional check for Windows-style backslash traversal
            if (
                "\\..\\" in clean_path_str
                or clean_path_str.startswith("..\\")
                or clean_path_str.endswith("\\..")
            ):
                # logger.warning(
                #     f"Security violation: Windows path traversal attempt detected - {file_path_str}"
                # )
                return None

        # Normalize path separators (convert backslashes to forward slashes)
        # This helps handle Windows-style paths on Unix systems
        normalized_path = clean_path_str.replace("\\", "/")

        # Create path object and resolve it (handles symlinks and relative paths)
        candidate_path = (base_dir / normalized_path).resolve()
        base_dir_resolved = base_dir.resolve()

        # Check if the resolved path is within the base directory
        if not candidate_path.is_relative_to(base_dir_resolved):
            # logger.warning(
            #     f"Security violation: Path traversal attempt detected - {file_path_str}"
            # )
            return None

        return candidate_path

    except (OSError, ValueError, Exception) as e:
        logger.warning(f"Invalid file path detected: {file_path_str} - {str(e)}")
        return None


def get_unique_filename_in_enqueued(target_dir: Path, original_name: str) -> str:
    """Generate a unique filename in the target directory by adding numeric suffixes if needed

    Args:
        target_dir: Target directory path
        original_name: Original filename

    Returns:
        str: Unique filename (may have numeric suffix added)
    """
    import time

    original_path = Path(original_name)
    base_name = original_path.stem
    extension = original_path.suffix

    # Try original name first
    if not (target_dir / original_name).exists():
        return original_name

    # Try with numeric suffixes 001-999
    for i in range(1, 1000):
        suffix = f"{i:03d}"
        new_name = f"{base_name}_{suffix}{extension}"
        if not (target_dir / new_name).exists():
            return new_name

    # Fallback with timestamp if all 999 slots are taken
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}{extension}"


def _create_markdown_content(content: str, file_path: str, doc_id: str, content_hash: str, extraction_timestamp: str) -> str:
    """Create markdown content with metadata header for MinIO storage.
    
    Args:
        content: The extracted text content
        file_path: Original file path/name
        doc_id: Document ID
        content_hash: Content hash for duplicate detection
        extraction_timestamp: When the extraction was performed
        
    Returns:
        Formatted markdown content with YAML frontmatter
    """
    # Create YAML frontmatter with metadata
    frontmatter = f"""---
doc_id: {doc_id}
original_file: {file_path}
content_hash: {content_hash}
extraction_timestamp: {extraction_timestamp}
format: markdown
source: lightrag_extraction
---

"""
    
    # Clean and format the content
    cleaned_content = content.strip()
    
    # If content doesn't start with a header, add the filename as title
    if not cleaned_content.startswith('#'):
        title = Path(file_path).stem.replace('_', ' ').replace('-', ' ').title()
        cleaned_content = f"# {title}\n\n{cleaned_content}"
    
    # Combine frontmatter and content
    markdown_content = frontmatter + cleaned_content
    
    return markdown_content


# Document processing helper functions (synchronous)
# These functions run in thread pool via asyncio.to_thread() to avoid blocking the event loop


def _convert_with_docling(file_path: Path) -> str:
    """Convert document using docling (synchronous).

    Args:
        file_path: Path to the document file

    Returns:
        str: Extracted markdown content
    """
    from docling.document_converter import DocumentConverter  # type: ignore

    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()



def _extract_docx(file_bytes: bytes) -> str:
    """Extract DOCX content including tables in document order (synchronous).

    Args:
        file_bytes: DOCX file content as bytes

    Returns:
        str: Extracted text content with tables in their original positions.
             Tables are separated from paragraphs with blank lines for clarity.
    """
    from docx import Document  # type: ignore
    from docx.table import Table  # type: ignore
    from docx.text.paragraph import Paragraph  # type: ignore

    docx_file = BytesIO(file_bytes)
    doc = Document(docx_file)

    def escape_cell(cell_value: str | None) -> str:
        """Escape characters that would break tab-delimited layout.

        Escape order is critical: backslashes first, then tabs/newlines.
        This prevents double-escaping issues.

        Args:
            cell_value: The cell value to escape (can be None or str)

        Returns:
            str: Escaped cell value safe for tab-delimited format
        """
        if cell_value is None:
            return ""
        text = str(cell_value)
        # CRITICAL: Escape backslash first to avoid double-escaping
        return (
            text.replace("\\", "\\\\")  # Must be first: \ -> \\
            .replace("\t", "&emsp;&emsp;")  # Tab -> \t (visible)
            .replace("\r\n", "<br>")  # Windows newline -> \n
            .replace("\r", "<br>")  # Mac newline -> \n
            .replace("\n", "<br>")  # Unix newline -> \n
        )

    content_parts = []
    in_table = False  # Track if we're currently processing a table

    # Iterate through all body elements in document order
    for element in doc.element.body:
        # Check if element is a paragraph
        if element.tag.endswith("p"):
            # If coming out of a table, add blank line after table
            if in_table:
                content_parts.append("")  # Blank line after table
                in_table = False

            paragraph = Paragraph(element, doc)
            text = paragraph.text
            # Always append to preserve document spacing (including blank paragraphs)
            content_parts.append(text)

        # Check if element is a table
        elif element.tag.endswith("tbl"):
            # Add blank line before table (if content exists)
            if content_parts and not in_table:
                content_parts.append("")  # Blank line before table

            in_table = True
            table = Table(element, doc)
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    cell_text = cell.text
                    # Escape special characters to preserve tab-delimited structure
                    row_text.append(escape_cell(cell_text))
                # Only add row if at least one cell has content
                if any(cell for cell in row_text):
                    content_parts.append("\t".join(row_text))

    return "\n".join(content_parts)


def _extract_pptx(file_bytes: bytes) -> str:
    """Extract PPTX content (synchronous).

    Args:
        file_bytes: PPTX file content as bytes

    Returns:
        str: Extracted text content
    """
    from pptx import Presentation  # type: ignore

    pptx_file = BytesIO(file_bytes)
    prs = Presentation(pptx_file)
    content = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                content += shape.text + "\n"
    return content


def _extract_xlsx(file_bytes: bytes) -> str:
    """Extract XLSX content in tab-delimited format with clear sheet separation.

    This function processes Excel workbooks and converts them to a structured text format
    suitable for LLM prompts and RAG systems. Each sheet is clearly delimited with
    separator lines, and special characters are escaped to preserve the tab-delimited structure.

    Features:
    - Each sheet is wrapped with '====================' separators for visual distinction
    - Special characters (tabs, newlines, backslashes) are escaped to prevent structure corruption
    - Column alignment is preserved across all rows to maintain tabular structure
    - Empty rows are preserved as blank lines to maintain row structure
    - Uses sheet.max_column to determine column width efficiently

    Args:
        file_bytes: XLSX file content as bytes

    Returns:
        str: Extracted text content with all sheets in tab-delimited format.
             Format: Sheet separators, sheet name, then tab-delimited rows.

    Example output:
        ==================== Sheet: Data ====================
        Name\tAge\tCity
        Alice\t30\tNew York
        Bob\t25\tLondon

        ==================== Sheet: Summary ====================
        Total\t2
        ====================
    """
    from openpyxl import load_workbook  # type: ignore

    xlsx_file = BytesIO(file_bytes)
    wb = load_workbook(xlsx_file)

    def escape_cell(cell_value: str | int | float | None) -> str:
        """Escape characters that would break tab-delimited layout.

        Escape order is critical: backslashes first, then tabs/newlines.
        This prevents double-escaping issues.

        Args:
            cell_value: The cell value to escape (can be None, str, int, or float)

        Returns:
            str: Escaped cell value safe for tab-delimited format
        """
        if cell_value is None:
            return ""
        text = str(cell_value)
        # CRITICAL: Escape backslash first to avoid double-escaping
        return (
            text.replace("\\", "\\\\")  # Must be first: \ -> \\
            .replace("\t", "\\t")  # Tab -> \t (visible)
            .replace("\r\n", "\\n")  # Windows newline -> \n
            .replace("\r", "\\n")  # Mac newline -> \n
            .replace("\n", "\\n")  # Unix newline -> \n
        )

    def escape_sheet_title(title: str) -> str:
        """Escape sheet title to prevent formatting issues in separators.

        Args:
            title: Original sheet title

        Returns:
            str: Sanitized sheet title with tabs/newlines replaced
        """
        return str(title).replace("\n", " ").replace("\t", " ").replace("\r", " ")

    content_parts: list[str] = []
    sheet_separator = "=" * 20

    for idx, sheet in enumerate(wb):
        if idx > 0:
            content_parts.append("")  # Blank line between sheets for readability

        # Escape sheet title to handle edge cases with special characters
        safe_title = escape_sheet_title(sheet.title)
        content_parts.append(f"{sheet_separator} Sheet: {safe_title} {sheet_separator}")

        # Use sheet.max_column to get the maximum column width directly
        max_columns = sheet.max_column if sheet.max_column else 0

        # Extract rows with consistent width to preserve column alignment
        for row in sheet.iter_rows(values_only=True):
            row_parts = []

            # Build row up to max_columns width
            for idx in range(max_columns):
                if idx < len(row):
                    row_parts.append(escape_cell(row[idx]))
                else:
                    row_parts.append("")  # Pad short rows

            # Check if row is completely empty
            if all(part == "" for part in row_parts):
                # Preserve empty rows as blank lines (maintains row structure)
                content_parts.append("")
            else:
                # Join all columns to maintain consistent column count
                content_parts.append("\t".join(row_parts))

    # Final separator for symmetry (makes parsing easier)
    content_parts.append(sheet_separator)
    return "\n".join(content_parts)


def _to_pdf_hex_text(value: str) -> str:
    """Encode text for PDF text operator using UTF-16BE hex string."""
    return ("feff" + value.encode("utf-16-be").hex()).upper()


def _render_text_to_pdf_bytes(content: str) -> bytes:
    """Render plain text content to a simple multi-page PDF."""
    page_width = 612
    page_height = 792
    margin = 48
    font_size = 11
    line_height = 14
    chars_per_line = 92

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    wrapped_lines: list[str] = []
    for raw_line in normalized.split("\n"):
        if not raw_line:
            wrapped_lines.append(" ")
            continue
        wrapped_lines.extend(textwrap.wrap(raw_line, width=chars_per_line) or [" "])

    lines_per_page = max(1, (page_height - margin * 2) // line_height)
    pages = [
        wrapped_lines[i : i + lines_per_page]
        for i in range(0, len(wrapped_lines), lines_per_page)
    ] or [[" "]]

    objects: list[bytes] = []

    # 1: Catalog, 2: Pages, 3: Font
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    page_object_ids: list[int] = []
    content_object_ids: list[int] = []
    next_id = 4
    for _ in pages:
        page_object_ids.append(next_id)
        content_object_ids.append(next_id + 1)
        next_id += 2

    kids = " ".join(f"{pid} 0 R" for pid in page_object_ids)
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>".encode(
            "ascii"
        )
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_id, content_id, page_lines in zip(page_object_ids, content_object_ids, pages):
        y = page_height - margin - font_size
        stream_cmds = ["BT", f"/F1 {font_size} Tf", f"{margin} {y} Td"]
        first = True
        for line in page_lines:
            text_hex = _to_pdf_hex_text(line)
            if not first:
                stream_cmds.append(f"0 -{line_height} Td")
            stream_cmds.append(f"<{text_hex}> Tj")
            first = False
        stream_cmds.append("ET")

        stream_bytes = "\n".join(stream_cmds).encode("ascii")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        objects.append(
            (
                f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
                + stream_bytes
                + b"\nendstream"
            )
        )

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]

    for obj_id, obj_data in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        output.extend(obj_data)
        output.extend(b"\nendobj\n")

    xref_pos = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        output.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


async def _extract_text_for_preview(file_path: Path) -> str:
    """Extract text content from a supported file for PDF preview rendering."""
    ext = file_path.suffix.lower()
    async with aiofiles.open(file_path, "rb") as f:
        file_bytes = await f.read()

    match ext:
        case (
            ".txt"
            | ".md"
            | ".mdx"
            | ".html"
            | ".htm"
            | ".tex"
            | ".json"
            | ".xml"
            | ".yaml"
            | ".yml"
            | ".rtf"
            | ".odt"
            | ".epub"
            | ".csv"
            | ".log"
            | ".conf"
            | ".ini"
            | ".properties"
            | ".sql"
            | ".bat"
            | ".sh"
            | ".c"
            | ".h"
            | ".cpp"
            | ".hpp"
            | ".py"
            | ".java"
            | ".js"
            | ".ts"
            | ".swift"
            | ".go"
            | ".rb"
            | ".php"
            | ".css"
            | ".scss"
            | ".less"
        ):
            return file_bytes.decode("utf-8", errors="replace")
        case ".pdf":
            # Disabled direct PDF extraction without full processing framework
            return "PDF previews are supported through MinIO markdown conversions only."
        case ".docx":
            return await asyncio.to_thread(_extract_docx, file_bytes)
        case ".pptx":
            return await asyncio.to_thread(_extract_pptx, file_bytes)
        case ".xlsx":
            return await asyncio.to_thread(_extract_xlsx, file_bytes)
        case _:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for preview conversion: {ext}",
            )


async def pipeline_enqueue_file(
    rag: LightRAG, file_path: Path, doc_id: str = None, task_id: str = None, doc_type: str = "general"
) -> tuple[bool, str]:
    """Add a file to the queue for processing

    Args:
        rag: LightRAG instance
        file_path: Path to the saved file
        doc_id: Optional document ID (for reprocessing existing documents)
        task_id: Optional task ID for tracking batch operations
        doc_type: Document type classification ("legal" or "general", default: "general")
    Returns:
        tuple: (success: bool, doc_id: str)
    """
    from lightrag.utils import generate_doc_id

    # Generate UUID-based doc_id if not provided (before extraction)
    # Note: UUID4 collisions are extremely rare (~1 in 2^122). If a collision occurs,
    # the database primary key constraint will fail during upsert, and we retry with
    # a new UUID up to 3 times before failing.
    max_uuid_retries = 3
    for uuid_attempt in range(max_uuid_retries):
        if doc_id is None:
            doc_id = generate_doc_id("doc-")
        
        # Create document record with status=EXTRACTING and content_hash=NULL
        try:
            # Build metadata with doc_type
            metadata = {"file_name": file_path.name, "doc_type": doc_type}
            if task_id:
                metadata["task_id"] = task_id
            
            initial_doc_data = {
                doc_id: {
                    "status": DocStatus.EXTRACTING,
                    "content_summary": "",
                    "content_length": 0,
                    "chunks_count": 0,
                    "file_path": file_path.name,
                    "chunks_list": [],
                    "content_hash": None,  # Will be set after extraction
                    "metadata": metadata,
                    "error_msg": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            }
            await rag.doc_status.upsert(initial_doc_data)
            logger.info(f"[Document] Created doc_id {doc_id} with EXTRACTING status and doc_type={doc_type} for file: {file_path.name}")
            break  # Success - exit retry loop
            
        except Exception as e:
            # Check if this is a UUID collision (primary key violation)
            error_str = str(e).lower()
            is_collision = any(keyword in error_str for keyword in [
                'duplicate key', 'unique constraint', 'primary key', 'already exists'
            ])
            
            if is_collision and uuid_attempt < max_uuid_retries - 1:
                # UUID collision detected - retry with new UUID
                logger.warning(
                    f"[Document] UUID collision detected for doc_id {doc_id} "
                    f"(attempt {uuid_attempt + 1}/{max_uuid_retries}). Retrying with new UUID."
                )
                doc_id = None  # Force generation of new UUID on next iteration
                continue
            else:
                # Non-collision error or max retries exhausted
                if is_collision:
                    logger.error(
                        f"[Document] Failed to generate unique doc_id after {max_uuid_retries} attempts"
                    )
                else:
                    logger.error(f"[Document] Failed to create initial document record: {e}")
                return False, doc_id if doc_id else generate_doc_id("doc-")

    try:
        content = ""
        ext = file_path.suffix.lower()
        file_size = 0

        # Get file size for error reporting
        try:
            file_size = file_path.stat().st_size
        except Exception:
            file_size = 0

        file = None
        try:
            async with aiofiles.open(file_path, "rb") as f:
                file = await f.read()
        except PermissionError as e:
            error_msg = f"[File Extraction]Permission denied - cannot read file: {str(e)}"
            await rag.doc_status.upsert({
                doc_id: {
                    "status": DocStatus.FAILED,
                    "error_msg": error_msg,
                    "content_length": file_size,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            })
            logger.error(f"[File Extraction]Permission denied reading file: {file_path.name}")
            return False, doc_id
        except FileNotFoundError as e:
            error_msg = f"[File Extraction]File not found: {str(e)}"
            await rag.doc_status.upsert({
                doc_id: {
                    "status": DocStatus.FAILED,
                    "error_msg": error_msg,
                    "content_length": file_size,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            })
            logger.error(f"[File Extraction]File not found: {file_path.name}")
            return False, doc_id
        except Exception as e:
            error_msg = f"[File Extraction]File reading error: {str(e)}"
            await rag.doc_status.upsert({
                doc_id: {
                    "status": DocStatus.FAILED,
                    "error_msg": error_msg,
                    "content_length": file_size,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            })
            logger.error(f"[File Extraction]Error reading file {file_path.name}: {str(e)}")
            return False, doc_id

        # Process based on file type
        try:
            match ext:
                case (
                    ".txt"
                    | ".md"
                    | ".mdx"
                    | ".html"
                    | ".htm"
                    | ".tex"
                    | ".json"
                    | ".xml"
                    | ".yaml"
                    | ".yml"
                    | ".log"
                    | ".conf"
                    | ".ini"
                    | ".properties"
                    | ".sql"
                    | ".bat"
                    | ".sh"
                    | ".c"
                    | ".h"
                    | ".cpp"
                    | ".hpp"
                    | ".py"
                    | ".java"
                    | ".js"
                    | ".ts"
                    | ".swift"
                    | ".go"
                    | ".rb"
                    | ".php"
                    | ".css"
                    | ".scss"
                    | ".less"
                ):
                    # Text-based files: read directly without OCR
                    try:
                        content = file.decode('utf-8')
                        logger.info(f"[File Extraction]Text file read directly: {file_path.name} ({len(content)} chars)")
                    except UnicodeDecodeError:
                        # Try other encodings
                        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                            try:
                                content = file.decode(encoding)
                                logger.info(f"[File Extraction]Text file read with {encoding}: {file_path.name} ({len(content)} chars)")
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            error_msg = "[File Extraction]Text file encoding error: Failed to decode text file with utf-8, latin-1, cp1252, or iso-8859-1"
                            await rag.doc_status.upsert({
                                doc_id: {
                                    "status": DocStatus.FAILED,
                                    "error_msg": error_msg,
                                    "content_length": file_size,
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                }
                            })
                            logger.error(f"[File Extraction]Failed to decode text file: {file_path.name}")
                            return False, doc_id
                    
                    if not content.strip():
                        error_msg = "[File Extraction]Empty text file: Text file is empty or contains only whitespace"
                        await rag.doc_status.upsert({
                            doc_id: {
                                "status": DocStatus.FAILED,
                                "error_msg": error_msg,
                                "content_length": file_size,
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                            }
                        })
                        logger.warning(f"[File Extraction]Empty text file: {file_path.name}")
                        return False, doc_id

                case (
                    ".rtf"
                    | ".odt"
                    | ".epub"
                    | ".csv"
                    | ".pdf"
                    | ".jpg"
                    | ".jpeg"
                    | ".png"
                    | ".webp"
                    | ".tiff"
                    | ".bmp"
                ):
                    # Binary files: use OCR/extraction pipeline
                    try:
                        # Create OCR config from global_args
                        ocr_config = OCRConfig.from_global_args(global_args)
                        
                        # Call process_document_with_ocr() for document extraction
                        ocr_result = await process_document_with_ocr(
                            file_bytes=file,
                            password=global_args.pdf_decrypt_password,
                            ocr_config=ocr_config,
                            file_extension=ext.lower()
                        )
                        
                        # Log OCR engine used and processing time
                        logger.info(
                            f"[File Extraction]Document processed with {ocr_result.engine_used} engine: "
                            f"{file_path.name} ({ocr_result.page_count} pages, "
                            f"{ocr_result.processing_time:.2f}s)"
                        )
                        
                        # Handle OCRResult with markdown format
                        if ocr_result.format == "markdown":
                            # Use MarkdownPreprocessor for markdown output
                            preprocessor = MarkdownPreprocessor(
                                preserve_structure=ocr_config.preserve_markdown_structure
                            )
                            content = preprocessor.preprocess(ocr_result.text)
                            
                            # Extract structure hints to pass to RAG pipeline
                            structure_hints = preprocessor.extract_structure_hints(ocr_result.text)
                            
                            logger.debug(
                                f"[File Extraction]Markdown preprocessing complete: "
                                f"{len(structure_hints['headers'])} headers, "
                                f"{len(structure_hints['tables'])} table rows, "
                                f"{len(structure_hints['lists'])} list items"
                            )
                            
                            # Note: structure_hints will be passed to RAG pipeline via metadata
                            # This is handled in the enqueue step below
                        else:
                            # Handle plain text format directly
                            content = ocr_result.text
                        
                        # Check if extraction failed
                        if not content.strip():
                            # Strictly avoid fallback to Docling for PDFs and Images to prevent CPU/Memory exhaustion and low quality results
                            # as per AGENTS.md guidelines.
                            is_binary_doc = ext.lower() in [".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"]
                            
                            if is_binary_doc:
                                logger.warning(
                                    f"[File Extraction] OCR returned no content for {file_path.name}. "
                                    "Skipping fallback for binary document."
                                )
                            elif _is_docling_available():
                                logger.info(f"[File Extraction] Fallback to Docling for {file_path.name}")
                                try:
                                    content = await asyncio.to_thread(_convert_with_docling, file_path)
                                except Exception as e:
                                    logger.error(f"[File Extraction] Docling fallback failed: {e}")
                                    
                        if not content.strip():
                            if ocr_result and ocr_result.error:
                                error_msg = f"[File Extraction]Document extraction failed: {ocr_result.error}"
                            else:
                                error_msg = "[File Extraction]Document extraction returned no content (OCR)"
                            
                            await rag.doc_status.upsert({
                                doc_id: {
                                    "status": DocStatus.FAILED,
                                    "error_msg": error_msg,
                                    "content_length": file_size,
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                }
                            })
                            logger.error(f"[File Extraction]{error_msg} for {file_path.name}")
                            return False, doc_id

                            
                    except Exception as e:
                        error_msg = f"[File Extraction]Document processing error: Failed to extract text from Document: {str(e)}"
                        await rag.doc_status.upsert({
                            doc_id: {
                                "status": DocStatus.FAILED,
                                "error_msg": error_msg,
                                "content_length": file_size,
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                            }
                        })
                        logger.error(f"[File Extraction]Error processing Document {file_path.name}: {str(e)}")
                        return False, doc_id

                case ".docx" | ".pptx" | ".xlsx":
                    try:
                        content = ""
                        # Try DOCLING first if available
                        if _is_docling_available():
                            logger.info(f"[File Extraction] Attempting Docling for {file_path.name}")
                            content = await asyncio.to_thread(
                                _convert_with_docling, file_path
                            )
                        else:
                            # High-speed native fallback (python-docx, python-pptx, openpyxl)
                            logger.info(f"[File Extraction] Docling not available, using native extractor for {file_path.name}")
                            if ext == ".docx":
                                content = await asyncio.to_thread(_extract_docx, file)
                            elif ext == ".pptx":
                                content = await asyncio.to_thread(_extract_pptx, file)
                            elif ext == ".xlsx":
                                content = await asyncio.to_thread(_extract_xlsx, file)
                        
                        if not content or not content.strip():
                            # Fallback to OCR as requested
                            logger.info(f"[File Extraction] Primary native extraction failed/empty for {file_path.name}, falling back to OCR")
                            ocr_config = OCRConfig.from_global_args(global_args)
                            ocr_result = await process_document_with_ocr(
                                file_bytes=file,
                                ocr_config=ocr_config,
                                file_extension=ext.lower()
                            )
                            content = ocr_result.text
                            
                    except Exception as e:
                        logger.warning(f"[File Extraction] Primary extraction for {file_path.name} failed: {e}. Trying OCR fallback.")
                        try:
                            ocr_config = OCRConfig.from_global_args(global_args)
                            ocr_result = await process_document_with_ocr(
                                file_bytes=file,
                                ocr_config=ocr_config,
                                file_extension=ext.lower()
                            )
                            content = ocr_result.text
                        except Exception as ocr_e:
                            logger.error(f"[File Extraction] OCR fallback also failed for {file_path.name}: {ocr_e}")
                            content = ""

                    if not content or not content.strip():
                        error_msg = f"[File Extraction] All extraction methods failed for {file_path.name}"
                        await rag.doc_status.upsert({
                            doc_id: {
                                "status": DocStatus.FAILED,
                                "error_msg": error_msg,
                                "content_length": file_size,
                                "updated_at": datetime.now(timezone.utc).isoformat(),
                            }
                        })
                        logger.error(f"[File Extraction]{error_msg}")
                        return False, doc_id

                case _:
                    error_msg = f"[File Extraction]Unsupported file type: {ext} - File extension {ext} is not supported"
                    await rag.doc_status.upsert({
                        doc_id: {
                            "status": DocStatus.FAILED,
                            "error_msg": error_msg,
                            "content_length": file_size,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    })
                    logger.error(f"[File Extraction]Unsupported file type: {file_path.name} (extension {ext})")
                    return False, doc_id

        except Exception as e:
            error_msg = f"[File Extraction]File format processing error: Unexpected error during file extracting: {str(e)}"
            await rag.doc_status.upsert({
                doc_id: {
                    "status": DocStatus.FAILED,
                    "error_msg": error_msg,
                    "content_length": file_size,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            })
            logger.error(f"[File Extraction]Unexpected error during {file_path.name} extracting: {str(e)}")
            return False, doc_id

        # Insert into the RAG queue
        if content:
            # Check if content contains only whitespace characters
            if not content.strip():
                error_msg = "[File Extraction]File contains only whitespace: File content contains only whitespace characters"
                await rag.doc_status.upsert({
                    doc_id: {
                        "status": DocStatus.FAILED,
                        "error_msg": error_msg,
                        "content_length": file_size,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                })
                logger.warning(f"[File Extraction]File contains only whitespace characters: {file_path.name}")
                return False, doc_id

            try:
                # Compute content_hash after successful extraction
                from lightrag.utils import compute_content_hash, sanitize_text_for_encoding
                
                sanitized_content = sanitize_text_for_encoding(content)
                content_hash = compute_content_hash(sanitized_content, rag.workspace)
                
                logger.debug(f"[Document] Computed content_hash for doc_id {doc_id}: {content_hash}")
                
                # Check for duplicate content using content_hash
                is_duplicate, existing_doc_id = await rag.doc_status.check_duplicate_by_content_hash(
                    content_hash=content_hash,
                    exclude_doc_id=doc_id
                )
                
                if is_duplicate:
                    # Mark as duplicate and store reference to original
                    logger.info(f"[Document] Duplicate detected: doc_id {doc_id} is duplicate of {existing_doc_id}")
                    
                    await rag.doc_status.upsert({
                        doc_id: {
                            "status": DocStatus.DUPLICATED,
                            "content_hash": content_hash,
                            "content_summary": get_content_summary(content),
                            "content_length": len(content),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "metadata": {
                                "file_name": file_path.name,
                                "duplicate_of": existing_doc_id,
                                "original_file_path": file_path.name
                            }
                        }
                    })
                    
                    logger.info(f"[Document] Marked doc_id {doc_id} as DUPLICATED (original: {existing_doc_id})")
                    
                    # Move file to __enqueued__ directory even for duplicates
                    try:
                        enqueued_dir = file_path.parent / "__enqueued__"
                        enqueued_dir.mkdir(exist_ok=True)
                        unique_filename = get_unique_filename_in_enqueued(
                            enqueued_dir, file_path.name
                        )
                        target_path = enqueued_dir / unique_filename
                        file_path.rename(target_path)
                        logger.debug(
                            f"Moved duplicate file to enqueued directory: {file_path.name} -> {unique_filename}"
                        )
                    except Exception as move_error:
                        logger.error(
                            f"Failed to move duplicate file {file_path.name} to __enqueued__ directory: {move_error}"
                        )
                    
                    return True, doc_id
                
                # Not a duplicate - proceed with normal processing
                logger.debug(f"[Document] No duplicate found for doc_id {doc_id}, proceeding to chunking")
                
                # Update document with content_hash and status EXTRACTED
                # This allows the document to be picked up by the chunking pipeline
                await rag.doc_status.upsert({
                    doc_id: {
                        "status": DocStatus.EXTRACTED,
                        "content_hash": content_hash,
                        "content_summary": get_content_summary(content),
                        "content_length": len(content),
                        "file_path": file_path.name,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                })
                
                # Store full document content as markdown file in MinIO
                # Convert content to markdown format with metadata header
                markdown_content = _create_markdown_content(
                    content=content,
                    file_path=file_path.name,
                    doc_id=doc_id,
                    content_hash=content_hash,
                    extraction_timestamp=datetime.now(timezone.utc).isoformat()
                )
                
                await rag.full_docs.upsert({
                    doc_id: {
                        "content": markdown_content,
                        "file_path": file_path.name,
                        "format": "markdown",
                        "original_filename": file_path.name,
                        "content_hash": content_hash,
                        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                })
                
                logger.info(f"Successfully extracted and stored markdown content in MinIO for: {file_path.name} with doc_id: {doc_id}")

                # Move file to __enqueued__ directory after enqueuing
                try:
                    enqueued_dir = file_path.parent / "__enqueued__"
                    enqueued_dir.mkdir(exist_ok=True)

                    # Generate unique filename to avoid conflicts
                    unique_filename = get_unique_filename_in_enqueued(
                        enqueued_dir, file_path.name
                    )
                    target_path = enqueued_dir / unique_filename

                    # Move the file
                    file_path.rename(target_path)
                    logger.debug(
                        f"Moved file to enqueued directory: {file_path.name} -> {unique_filename}"
                    )

                except Exception as move_error:
                    logger.error(
                        f"Failed to move file {file_path.name} to __enqueued__ directory: {move_error}"
                    )
                    # Don't affect the main function's success status

                return True, doc_id

            except Exception as e:
                error_msg = f"Document enqueue error: Failed to enqueue document: {str(e)}"
                await rag.doc_status.upsert({
                    doc_id: {
                        "status": DocStatus.FAILED,
                        "error_msg": error_msg,
                        "content_length": file_size,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                })
                logger.error(f"Error enqueueing document {file_path.name}: {str(e)}")
                return False, doc_id
        else:
            error_msg = "No content extracted: No content could be extracted from file"
            await rag.doc_status.upsert({
                doc_id: {
                    "status": DocStatus.FAILED,
                    "error_msg": error_msg,
                    "content_length": file_size,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            })
            logger.error(f"No content extracted from file: {file_path.name}")
            return False, doc_id

    except Exception as e:
        # Catch-all for any unexpected errors
        try:
            file_size = file_path.stat().st_size if file_path.exists() else 0
        except Exception:
            file_size = 0

        error_msg = f"Unexpected processing error: {str(e)}"
        await rag.doc_status.upsert({
            doc_id: {
                "status": DocStatus.FAILED,
                "error_msg": error_msg,
                "content_length": file_size,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        })
        logger.error(f"Enqueuing file {file_path.name} error: {str(e)}")
        logger.error(traceback.format_exc())
        return False, doc_id
    finally:
        if file_path.name.startswith(temp_prefix):
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")


async def pipeline_index_file(rag: LightRAG, file_path: Path, doc_id: str = None, resolved_workspace_id: str = None, doc_type: str = "general") -> str:
    """Index a file with optional doc_id for reprocessing

    Args:
        rag: LightRAG instance
        file_path: Path to the saved file
        doc_id: Optional document ID (for reprocessing existing documents)
        resolved_workspace_id: Resolved workspace ID (for Celery task, defaults to rag.workspace or "default")
        doc_type: Document type classification ("legal" or "general", default: "general")
        
    Returns:
        str: The document ID (UUID-based)
    """
    from lightrag.utils import generate_doc_id
    
    # Generate UUID doc_id if not provided
    if doc_id is None:
        doc_id = generate_doc_id("doc-")
    
    # Resolve workspace_id for Celery/Kafka task
    if resolved_workspace_id is None:
        resolved_workspace_id = rag.workspace if rag.workspace else "default"

    from lightrag.pipeline.ingest_flags import (
        build_ingest_stage_event,
        enqueue_kafka_stage_events,
        resolve_ingest_backend,
    )
    from lightrag.pipeline.models import StageName

    ingest_backend = resolve_ingest_backend()
    if ingest_backend == "kafka":
        event = build_ingest_stage_event(
            workspace_id=resolved_workspace_id,
            doc_id=doc_id,
            stage=StageName.EXTRACTION,
            doc_type=doc_type,
            file_path=str(file_path),
        )
        await enqueue_kafka_stage_events([event])
        return doc_id

    # Check if Celery is available and enabled
    use_celery = ingest_backend == "celery"
    
    try:
        from celery_worker.tasks import task_extract_and_enqueue
        celery_available = True
    except ImportError:
        celery_available = False
        logger.warning("Celery worker module not available, falling back to synchronous processing")
    
    if celery_available and use_celery:
        try:
            # Use resolved_workspace_id to ensure consistency with database
            # Empty workspace ("") is normalized to "default" for Celery tasks
            task_extract_and_enqueue.delay(workspace_id=resolved_workspace_id, file_path_str=str(file_path), doc_id=doc_id, doc_type=doc_type)
            return doc_id
        except Exception as e:
            logger.error(f"Error enqueueing file via Celery {file_path.name}, falling back to synchronous: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Fallback to synchronous processing
    try:
        await pipeline_enqueue_file(rag, file_path, doc_id, doc_type=doc_type)
        return doc_id
    except Exception as e:
        logger.error(f"Error indexing file {file_path.name}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


async def pipeline_index_files(
    rag: LightRAG, file_paths: List[Path], resolved_workspace_id: str = None
) -> List[str]:
    """Index multiple files sequentially to avoid high CPU load

    Args:
        rag: LightRAG instance
        file_paths: Paths to the files to index
        resolved_workspace_id: Resolved workspace ID (for Celery task, defaults to rag.workspace or "default")

    Returns:
        List[str]: List of generated doc_ids (one per file)
    """
    if not file_paths:
        return []
    
    from lightrag.utils import generate_doc_id
    
    # Resolve workspace_id for Celery task
    if resolved_workspace_id is None:
        resolved_workspace_id = rag.workspace if rag.workspace else "default"
    
    doc_ids = []

    from lightrag.pipeline.ingest_flags import (
        build_ingest_stage_event,
        enqueue_kafka_stage_events,
        resolve_ingest_backend,
    )
    from lightrag.pipeline.models import StageName

    ingest_backend = resolve_ingest_backend()
    if ingest_backend == "kafka":
        sorted_file_paths = sorted(
            file_paths, key=lambda p: get_pinyin_sort_key(str(p))
        )
        events = []
        for file_path in sorted_file_paths:
            doc_id = generate_doc_id("doc-")
            doc_ids.append(doc_id)
            events.append(
                build_ingest_stage_event(
                    workspace_id=resolved_workspace_id,
                    doc_id=doc_id,
                    stage=StageName.EXTRACTION,
                    file_path=str(file_path),
                )
            )
        await enqueue_kafka_stage_events(events)
        return doc_ids

    # Check if Celery is available and enabled
    use_celery = ingest_backend == "celery"
    
    try:
        from celery_worker.tasks import task_extract_and_enqueue
        celery_available = True
    except ImportError:
        celery_available = False
        logger.warning("Celery worker module not available, falling back to synchronous processing")
    
    if celery_available and use_celery:
        try:
            # Use get_pinyin_sort_key for Chinese pinyin sorting
            sorted_file_paths = sorted(
                file_paths, key=lambda p: get_pinyin_sort_key(str(p))
            )

            # Process files sequentially, generating unique doc_id for each
            for file_path in sorted_file_paths:
                doc_id = generate_doc_id("doc-")
                doc_ids.append(doc_id)
                # Use resolved_workspace_id to ensure consistency with database
                task_extract_and_enqueue.delay(workspace_id=resolved_workspace_id, file_path_str=str(file_path), doc_id=doc_id)
            
            return doc_ids
        except Exception as e:
            logger.error(f"Error enqueueing files via Celery, falling back to synchronous: {str(e)}")
            logger.error(traceback.format_exc())
            # Clear doc_ids on error and fall through to synchronous processing
            doc_ids = []
    
    # Fallback to synchronous processing
    try:
        # Use get_pinyin_sort_key for Chinese pinyin sorting
        sorted_file_paths = sorted(
            file_paths, key=lambda p: get_pinyin_sort_key(str(p))
        )
        
        for file_path in sorted_file_paths:
            doc_id = generate_doc_id("doc-")
            success, returned_doc_id = await pipeline_enqueue_file(rag, file_path, doc_id)
            if success:
                doc_ids.append(returned_doc_id)
            else:
                # Still add the doc_id even on failure so client can track the failed document
                doc_ids.append(returned_doc_id)
        
        return doc_ids
    except Exception as e:
        logger.error(f"Error indexing files: {str(e)}")
        logger.error(traceback.format_exc())
        return doc_ids


async def pipeline_index_texts(
    rag: LightRAG,
    texts: List[str],
    file_sources: List[str] = None,
    doc_ids: List[str] = None,
    doc_type: str = "general",
) -> List[str]:
    """Index a list of texts, generating unique doc_id for each if not provided

    Args:
        rag: LightRAG instance
        texts: The texts to index
        file_sources: Sources of the texts
        doc_ids: Optional list of doc_ids (one per text). If not provided, UUIDs will be generated.
        doc_type: Document type classification ("legal" or "general", default: "general")

    Returns:
        List[str]: List of doc_ids (one per text)
    """
    if not texts:
        return []
    
    from lightrag.utils import generate_doc_id

    normalized_file_sources: list[str] | None = None
    if file_sources:
        normalized_file_sources = [
            normalize_file_path(source) for source in file_sources
        ]
        if len(normalized_file_sources) > len(texts):
            raise ValueError("Number of file sources must not exceed number of texts")
        if len(normalized_file_sources) < len(texts):
            normalized_file_sources.extend(
                [UNKNOWN_FILE_SOURCE] * (len(texts) - len(normalized_file_sources))
            )
    else:
        normalized_file_sources = [UNKNOWN_FILE_SOURCE] * len(texts)

    # Generate unique UUID doc_id for each text if not provided
    if doc_ids is None:
        doc_ids = [generate_doc_id("doc-") for _ in texts]
    elif len(doc_ids) != len(texts):
        raise ValueError("Number of doc_ids must match number of texts")

    await rag.apipeline_enqueue_documents(
        input=texts, ids=doc_ids, file_paths=normalized_file_sources, doc_type=doc_type
    )

    from lightrag.pipeline.ingest_flags import (
        build_ingest_stage_event,
        enqueue_kafka_stage_events,
        resolve_ingest_backend,
    )
    from lightrag.pipeline.models import StageName

    ingest_backend = resolve_ingest_backend()
    if ingest_backend == "kafka":
        events = [
            build_ingest_stage_event(
                workspace_id=rag.workspace or "default",
                doc_id=doc_id,
                stage=StageName.CHUNKING,
                doc_type=doc_type,
                file_path=src,
            )
            for doc_id, src in zip(doc_ids, normalized_file_sources)
        ]
        await enqueue_kafka_stage_events(events)
        return doc_ids

    # Check if Celery is available and enabled
    use_celery = ingest_backend == "celery"
    
    try:
        from celery_worker.tasks import task_chunk_and_graph
        celery_available = True
    except ImportError:
        celery_available = False
        logger.warning("Celery worker module not available, processing will continue synchronously")
    
    if celery_available and use_celery:
        try:
            task_chunk_and_graph.delay(rag.workspace)
        except Exception as e:
            logger.error(f"Error enqueueing chunk_and_graph task via Celery: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        # When Celery is not available, the processing happens synchronously
        # through apipeline_enqueue_documents above
        logger.info("Processing texts synchronously (Celery not available or disabled)")
    
    return doc_ids


async def run_scanning_process(
    rag: LightRAG, doc_manager: DocumentManager
):
    """Background task to scan and index documents

    Args:
        rag: LightRAG instance
        doc_manager: DocumentManager instance
        
    Returns:
        List[str]: List of generated doc_ids for the scanned files
    """
    try:
        new_files = doc_manager.scan_directory_for_new_files()
        total_files = len(new_files)
        logger.info(f"Found {total_files} files to index.")

        doc_ids = []
        
        if new_files:
            # Check for files with PROCESSED status and filter them out
            valid_files = []
            processed_files = []

            for file_path in new_files:
                filename = file_path.name
                existing_doc_data = await rag.doc_status.get_doc_by_file_path(filename)

                existing_status = str(
                    existing_doc_data.get("status", "") if existing_doc_data else ""
                ).strip().upper()
                if existing_doc_data and existing_status in {"PROCESSED", "COMPLETED"}:
                    # File is already PROCESSED, skip it with warning
                    processed_files.append(filename)
                    logger.warning(f"Skipping already processed file: {filename}")
                else:
                    # File is new or in non-PROCESSED status, add to processing list
                    valid_files.append(file_path)

            # Process valid files (new files + non-PROCESSED status files)
            if valid_files:
                # Use rag.workspace as the workspace_id
                doc_ids = await pipeline_index_files(rag, valid_files, rag.workspace)
                if processed_files:
                    logger.info(
                        f"Scanning process completed: {len(valid_files)} files Processed {len(processed_files)} skipped."
                    )
                else:
                    logger.info(
                        f"Scanning process completed: {len(valid_files)} files Processed."
                    )
            else:
                logger.info(
                    "No files to process after filtering already processed files."
                )
        else:
            # No new files to index, check if there are any documents in the queue
            logger.info(
                "No upload file found, check if there are any documents in the queue..."
            )
            await rag.apipeline_process_enqueue_documents()
        
        return doc_ids

    except Exception as e:
        logger.error(f"Error during scanning process: {str(e)}")
        logger.error(traceback.format_exc())
        return []


async def background_delete_documents(
    rag: LightRAG,
    doc_manager: DocumentManager,
    doc_ids: List[str],
    delete_file: bool = False,
    delete_llm_cache: bool = False,
):
    """Background task to delete multiple documents"""
    from lightrag.kg.shared_storage import (
        get_namespace_data,
        get_namespace_lock,
    )

    pipeline_status = await get_namespace_data(
        "pipeline_status", workspace=rag.workspace
    )
    pipeline_status_lock = get_namespace_lock(
        "pipeline_status", workspace=rag.workspace
    )

    total_docs = len(doc_ids)
    successful_deletions = []
    failed_deletions = []

    # Double-check pipeline status before proceeding
    async with pipeline_status_lock:
        if pipeline_status.get("busy", False):
            logger.warning("Error: Unexpected pipeline busy state, aborting deletion.")
            return  # Abort deletion operation

        # Set pipeline status to busy for deletion
        pipeline_status.update(
            {
                "busy": True,
                # Job name can not be changed, it's verified in adelete_by_doc_id()
                "job_name": f"Deleting {total_docs} Documents",
                "job_start": datetime.now().isoformat(),
                "docs": total_docs,
                "batchs": total_docs,
                "cur_batch": 0,
                "latest_message": "Starting document deletion process",
            }
        )
        # Use slice assignment to clear the list in place
        pipeline_status["history_messages"][:] = ["Starting document deletion process"]
        if delete_llm_cache:
            pipeline_status["history_messages"].append(
                "LLM cache cleanup requested for this deletion job"
            )

    try:
        # Loop through each document ID and delete them one by one
        for i, doc_id in enumerate(doc_ids, 1):
            # Check for cancellation at the start of each document deletion
            async with pipeline_status_lock:
                if pipeline_status.get("cancellation_requested", False):
                    cancel_msg = f"Deletion cancelled by user at document {i}/{total_docs}. {len(successful_deletions)} deleted, {total_docs - i + 1} remaining."
                    logger.info(cancel_msg)
                    pipeline_status["latest_message"] = cancel_msg
                    pipeline_status["history_messages"].append(cancel_msg)
                    # Add remaining documents to failed list with cancellation reason
                    failed_deletions.extend(
                        doc_ids[i - 1 :]
                    )  # i-1 because enumerate starts at 1
                    break  # Exit the loop, remaining documents unchanged

                start_msg = f"Deleting document {i}/{total_docs}: {doc_id}"
                logger.info(start_msg)
                pipeline_status["cur_batch"] = i
                pipeline_status["latest_message"] = start_msg
                pipeline_status["history_messages"].append(start_msg)

            file_path = "#"
            try:
                result = await rag.adelete_by_doc_id(
                    doc_id, delete_llm_cache=delete_llm_cache
                )
                file_path = (
                    getattr(result, "file_path", "-") if "result" in locals() else "-"
                )
                if result.status == "success":
                    successful_deletions.append(doc_id)
                    success_msg = (
                        f"Document deleted {i}/{total_docs}: {doc_id}[{file_path}]"
                    )
                    logger.info(success_msg)
                    async with pipeline_status_lock:
                        pipeline_status["history_messages"].append(success_msg)

                    # Handle file deletion if requested and file_path is available
                    if (
                        delete_file
                        and result.file_path
                        and result.file_path != "unknown_source"
                    ):
                        try:
                            deleted_files = []
                            # SECURITY FIX: Use secure path validation to prevent arbitrary file deletion
                            safe_file_path = validate_file_path_security(
                                result.file_path, doc_manager.input_dir
                            )

                            if safe_file_path is None:
                                # Security violation detected - log and skip file deletion
                                security_msg = f"Security violation: Unsafe file path detected for deletion - {result.file_path}"
                                logger.warning(security_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = security_msg
                                    pipeline_status["history_messages"].append(
                                        security_msg
                                    )
                            else:
                                # check and delete files from input_dir directory
                                if safe_file_path.exists():
                                    try:
                                        safe_file_path.unlink()
                                        deleted_files.append(safe_file_path.name)
                                        file_delete_msg = f"Successfully deleted input_dir file: {result.file_path}"
                                        logger.info(file_delete_msg)
                                        async with pipeline_status_lock:
                                            pipeline_status["latest_message"] = (
                                                file_delete_msg
                                            )
                                            pipeline_status["history_messages"].append(
                                                file_delete_msg
                                            )
                                    except Exception as file_error:
                                        file_error_msg = f"Failed to delete input_dir file {result.file_path}: {str(file_error)}"
                                        logger.debug(file_error_msg)
                                        async with pipeline_status_lock:
                                            pipeline_status["latest_message"] = (
                                                file_error_msg
                                            )
                                            pipeline_status["history_messages"].append(
                                                file_error_msg
                                            )

                                # Also check and delete files from __enqueued__ directory
                                enqueued_dir = doc_manager.input_dir / "__enqueued__"
                                if enqueued_dir.exists():
                                    # SECURITY FIX: Validate that the file path is safe before processing
                                    # Only proceed if the original path validation passed
                                    base_name = Path(result.file_path).stem
                                    extension = Path(result.file_path).suffix

                                    # Search for exact match and files with numeric suffixes
                                    for enqueued_file in enqueued_dir.glob(
                                        f"{base_name}*{extension}"
                                    ):
                                        # Additional security check: ensure enqueued file is within enqueued directory
                                        safe_enqueued_path = (
                                            validate_file_path_security(
                                                enqueued_file.name, enqueued_dir
                                            )
                                        )
                                        if safe_enqueued_path is not None:
                                            try:
                                                enqueued_file.unlink()
                                                deleted_files.append(enqueued_file.name)
                                                logger.info(
                                                    f"Successfully deleted enqueued file: {enqueued_file.name}"
                                                )
                                            except Exception as enqueued_error:
                                                file_error_msg = f"Failed to delete enqueued file {enqueued_file.name}: {str(enqueued_error)}"
                                                logger.debug(file_error_msg)
                                                async with pipeline_status_lock:
                                                    pipeline_status[
                                                        "latest_message"
                                                    ] = file_error_msg
                                                    pipeline_status[
                                                        "history_messages"
                                                    ].append(file_error_msg)
                                        else:
                                            security_msg = f"Security violation: Unsafe enqueued file path detected - {enqueued_file.name}"
                                            logger.warning(security_msg)

                            if deleted_files == []:
                                file_error_msg = f"File deletion skipped, missing or unsafe file: {result.file_path}"
                                logger.warning(file_error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = file_error_msg
                                    pipeline_status["history_messages"].append(
                                        file_error_msg
                                    )

                        except Exception as file_error:
                            file_error_msg = f"Failed to delete file {result.file_path}: {str(file_error)}"
                            logger.error(file_error_msg)
                            async with pipeline_status_lock:
                                pipeline_status["latest_message"] = file_error_msg
                                pipeline_status["history_messages"].append(
                                    file_error_msg
                                )
                    elif delete_file:
                        no_file_msg = (
                            f"File deletion skipped, missing file path: {doc_id}"
                        )
                        logger.warning(no_file_msg)
                        async with pipeline_status_lock:
                            pipeline_status["latest_message"] = no_file_msg
                            pipeline_status["history_messages"].append(no_file_msg)
                else:
                    failed_deletions.append(doc_id)
                    error_msg = f"Failed to delete {i}/{total_docs}: {doc_id}[{file_path}] - {result.message}"
                    logger.error(error_msg)
                    async with pipeline_status_lock:
                        pipeline_status["latest_message"] = error_msg
                        pipeline_status["history_messages"].append(error_msg)

            except Exception as e:
                failed_deletions.append(doc_id)
                error_msg = f"Error deleting document {i}/{total_docs}: {doc_id}[{file_path}] - {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = error_msg
                    pipeline_status["history_messages"].append(error_msg)

    except Exception as e:
        error_msg = f"Critical error during batch deletion: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        async with pipeline_status_lock:
            pipeline_status["history_messages"].append(error_msg)
    finally:
        # Final summary and check for pending requests
        async with pipeline_status_lock:
            pipeline_status["busy"] = False
            pipeline_status["pending_requests"] = False  # Reset pending requests flag
            pipeline_status["cancellation_requested"] = (
                False  # Always reset cancellation flag
            )
            completion_msg = f"Deletion completed: {len(successful_deletions)} successful, {len(failed_deletions)} failed"
            pipeline_status["latest_message"] = completion_msg
            pipeline_status["history_messages"].append(completion_msg)

            # Check if there are pending document indexing requests
            has_pending_request = pipeline_status.get("request_pending", False)

        # If there are pending requests, start document processing pipeline
        if has_pending_request:
            try:
                logger.info(
                    "Processing pending document indexing requests after deletion"
                )
                await rag.apipeline_process_enqueue_documents()
            except Exception as e:
                logger.error(f"Error processing pending documents after deletion: {e}")


_STAGE_RUNS_WS_CACHE: dict[str, Any] = {"data": None, "ts": 0.0}
_STAGE_RUNS_METRICS_CACHE: dict[str, tuple[float, Any]] = {}
_STAGE_RUNS_METRICS_TTL = 30.0  # 30 seconds TTL for expensive aggregate metrics


def _get_stage_metrics_cache(key: str) -> Optional[Any]:
    import time
    entry = _STAGE_RUNS_METRICS_CACHE.get(key)
    if entry:
        ts, data = entry
        if time.time() - ts < _STAGE_RUNS_METRICS_TTL:
            return data
    return None


def _set_stage_metrics_cache(key: str, data: Any):
    import time
    if len(_STAGE_RUNS_METRICS_CACHE) > 500:
        _STAGE_RUNS_METRICS_CACHE.clear()
    _STAGE_RUNS_METRICS_CACHE[key] = (time.time(), data)


async def _get_stage_runs_workspace_map(db) -> tuple[list[str], list[Any]]:
    from lightrag.kg.postgres_impl import to_uuid
    import time
    now = time.monotonic()
    if _STAGE_RUNS_WS_CACHE["data"] is not None and (now - _STAGE_RUNS_WS_CACHE["ts"]) < 60.0:
        return _STAGE_RUNS_WS_CACHE["data"]

    ws_ids = set()
    try:
        ws_rows = await db.query("SELECT id, name FROM tbl_workspace", [], multirows=True)
        for r in (ws_rows or []):
            if r.get("id"):
                ws_ids.add(str(r["id"]).strip())
            if r.get("name"):
                ws_ids.add(str(r["name"]).strip())
    except Exception:
        pass
    ws_ids.add("default")

    ws_id_list = sorted([ws for ws in ws_ids if ws])
    ws_uuid_list = [to_uuid(ws) for ws in ws_id_list]
    res = (ws_id_list, ws_uuid_list)
    _STAGE_RUNS_WS_CACHE["data"] = res
    _STAGE_RUNS_WS_CACHE["ts"] = now
    return res


def create_document_routes(
    rag: LightRAG,
    doc_manager: DocumentManager,
    api_key: Optional[str] = None,
    get_rag_for_workspace: Optional[Callable[[str], Awaitable[LightRAG]]] = None,
    get_doc_manager_for_workspace: Optional[Callable[[str], DocumentManager]] = None,
):
    # Create combined auth dependency for document routes
    combined_auth = get_combined_auth_dependency(api_key)

    def normalize_workspace_id(workspace_id: Optional[str]) -> str:
        if workspace_id is None:
            return rag.workspace

        normalized_workspace_id = workspace_id.strip()
        if not normalized_workspace_id:
            return rag.workspace

        return normalized_workspace_id

    def generation_workspace_id(workspace_id: str) -> str:
        return workspace_id or "default"

    async def ensure_workspace_generation_writable(workspace_id: str) -> None:
        from lightrag.pipeline.coordinator import create_coordinator_from_env

        workspace_id = generation_workspace_id(workspace_id)
        coordinator = create_coordinator_from_env()
        try:
            if await coordinator.is_workspace_write_locked(workspace_id):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Workspace '{workspace_id}' is locked for a generation "
                        "build or cutover. Retry this write after the operation completes."
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                "Failed to verify generation write lock for workspace_id=%s",
                workspace_id,
            )
            raise HTTPException(
                status_code=503,
                detail="Unable to verify workspace generation state",
            ) from exc
        finally:
            try:
                await coordinator.finalize()
            except Exception:
                logger.exception(
                    "Failed to release generation coordinator for workspace_id=%s",
                    workspace_id,
                )

    class WorkspaceWriteLeaseManager:
        def __init__(self, coordinator):
            self.coordinator = coordinator
            self.leases = {}
            self.renewal_task = asyncio.create_task(self._renew_leases())

        async def acquire(self, workspace_ids: list[str]) -> None:
            from lightrag.pipeline.coordinator import WorkspaceGenerationConflictError

            for workspace_id in sorted(
                {generation_workspace_id(item) for item in workspace_ids}
            ):
                if workspace_id in self.leases:
                    continue
                try:
                    lease = await self.coordinator.acquire_workspace_write_lease(
                        workspace_id
                    )
                except WorkspaceGenerationConflictError as exc:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Workspace '{workspace_id}' is locked for a generation "
                            "build or cutover. Retry this write after the operation completes."
                        ),
                    ) from exc
                except Exception as exc:
                    logger.exception(
                        "Failed to acquire generation write lease workspace_id=%s",
                        workspace_id,
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Unable to acquire workspace write lease",
                    ) from exc
                self.leases[workspace_id] = lease

        async def _renew_leases(self) -> None:
            while True:
                await asyncio.sleep(60)
                for workspace_id, lease in list(self.leases.items()):
                    self.leases[workspace_id] = (
                        await self.coordinator.renew_workspace_write_lease(lease)
                    )

        async def run_background_tasks(self, tasks) -> None:
            try:
                for task in tasks:
                    task_run = asyncio.create_task(task())
                    done, _ = await asyncio.wait(
                        {task_run, self.renewal_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if self.renewal_task in done:
                        try:
                            self.renewal_task.result()
                        except Exception:
                            task_run.cancel()
                            await asyncio.gather(task_run, return_exceptions=True)
                            logger.exception(
                                "Workspace write lease renewal failed during background work"
                            )
                            raise
                        raise RuntimeError("Workspace write lease renewal stopped")
                    await task_run
            finally:
                await self.close()

        async def close(self) -> None:
            cleanup_errors = []
            self.renewal_task.cancel()
            await asyncio.gather(self.renewal_task, return_exceptions=True)
            for workspace_id, lease in self.leases.items():
                try:
                    await self.coordinator.release_workspace_write_lease(lease)
                except Exception as exc:
                    cleanup_errors.append(exc)
                    logger.exception(
                        "Failed to release workspace write lease workspace_id=%s",
                        workspace_id,
                    )
            self.leases.clear()
            try:
                await self.coordinator.finalize()
            except Exception as exc:
                cleanup_errors.append(exc)
                logger.exception("Failed to finalize generation coordinator")
            if cleanup_errors:
                raise RuntimeError("Workspace write lease cleanup failed") from cleanup_errors[0]

    lease_context = contextvars.ContextVar("workspace_write_lease_context", default=None)

    async def acquire_workspace_write_leases(workspace_ids: list[str]) -> None:
        manager = lease_context.get()
        if manager is None:
            raise RuntimeError("Workspace write lease context is unavailable")
        await manager.acquire(workspace_ids)

    def with_workspace_write_lease(
        endpoint: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        @wraps(endpoint)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound = inspect.signature(endpoint).bind_partial(*args, **kwargs)
            request = next(
                (
                    value
                    for value in bound.arguments.values()
                    if isinstance(value, Request)
                ),
                None,
            )
            requested_workspace = bound.arguments.get("workspace_id")
            if not requested_workspace:
                requested_workspace = next(
                    (
                        getattr(value, "workspace_id", None)
                        for value in bound.arguments.values()
                        if getattr(value, "workspace_id", None)
                    ),
                    None,
                )
            dynamic_workspace_selection = requested_workspace == "all" or (
                endpoint.__name__ == "reprocess_failed_documents"
                and not requested_workspace
            )
            background_tasks = bound.arguments.get("background_tasks")
            initial_task_count = (
                len(background_tasks.tasks) if background_tasks is not None else 0
            )

            from lightrag.pipeline.coordinator import create_coordinator_from_env

            coordinator = create_coordinator_from_env()
            manager = WorkspaceWriteLeaseManager(coordinator)
            token = lease_context.set(manager)
            endpoint_succeeded = False
            transferred_to_background = False
            try:
                if not dynamic_workspace_selection:
                    if not requested_workspace and request is not None:
                        requested_workspace = (
                            request.headers.get("LIGHTRAG-WORKSPACE-ID")
                            or request.headers.get("LIGHTRAG-WORKSPACE")
                        )
                    resolved_workspace = normalize_workspace_id(requested_workspace)
                    current_rag = (
                        await get_rag_for_workspace(resolved_workspace)
                        if get_rag_for_workspace is not None
                        else rag
                    )
                    await manager.acquire(
                        [generation_workspace_id(current_rag.workspace)]
                    )

                endpoint_task = asyncio.create_task(endpoint(*args, **kwargs))
                try:
                    done, _ = await asyncio.wait(
                        {endpoint_task, manager.renewal_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if manager.renewal_task in done:
                        try:
                            manager.renewal_task.result()
                        except Exception as exc:
                            endpoint_task.cancel()
                            await asyncio.gather(
                                endpoint_task, return_exceptions=True
                            )
                            logger.exception(
                                "Workspace write lease renewal failed during request"
                            )
                            raise HTTPException(
                                status_code=503,
                                detail="Workspace write lease could not be maintained",
                            ) from exc
                        raise RuntimeError(
                            "Workspace write lease renewal stopped"
                        )
                    result = await endpoint_task
                except asyncio.CancelledError:
                    endpoint_task.cancel()
                    await asyncio.gather(endpoint_task, return_exceptions=True)
                    raise

                endpoint_succeeded = True
                if background_tasks is not None and len(background_tasks.tasks) > initial_task_count:
                    pending_tasks = list(background_tasks.tasks)
                    background_tasks.tasks.clear()
                    background_tasks.add_task(
                        manager.run_background_tasks, pending_tasks
                    )
                    transferred_to_background = True
                return result
            finally:
                lease_context.reset(token)
                if not transferred_to_background:
                    try:
                        await manager.close()
                    except Exception as exc:
                        if endpoint_succeeded:
                            raise HTTPException(
                                status_code=503,
                                detail="Workspace write lease cleanup failed",
                            ) from exc

        return wrapped

    async def resolve_request_context(
        request: Optional[Request] = None,
        workspace_id: Optional[str] = None,
    ) -> tuple[str, LightRAG, DocumentManager]:
        header_workspace_id = ""
        if request is not None:
            header_workspace_id = request.headers.get(
                "LIGHTRAG-WORKSPACE-ID", ""
            ).strip()
            if not header_workspace_id:
                header_workspace_id = request.headers.get(
                    "LIGHTRAG-WORKSPACE", ""
                ).strip()

        resolved_workspace_id = normalize_workspace_id(workspace_id or header_workspace_id)

        current_rag = (
            await get_rag_for_workspace(resolved_workspace_id)
            if get_rag_for_workspace is not None
            else rag
        )

        # Use canonical workspace_id resolved by backend context.
        # If it's empty string (default workspace), return "default" for API responses
        backend_workspace = current_rag.workspace
        resolved_workspace_id = backend_workspace if backend_workspace else "default"

        if get_doc_manager_for_workspace is not None:
            current_doc_manager = get_doc_manager_for_workspace(backend_workspace)
        elif backend_workspace == doc_manager.workspace:
            current_doc_manager = doc_manager
        else:
            current_doc_manager = DocumentManager(
                str(doc_manager.base_input_dir), workspace=backend_workspace
            )

        return resolved_workspace_id, current_rag, current_doc_manager

    @router.post(
        "/scan", response_model=ScanResponse, dependencies=[Depends(combined_auth)]
    )
    @with_workspace_write_lease
    async def scan_for_new_documents(
        background_tasks: BackgroundTasks,
        request: Request,
        workspace_id: Optional[str] = None,
    ):
        """
        Trigger the scanning process for new documents.

        This endpoint initiates a background task that scans the input directory for new documents
        and processes them. Returns an array of doc_ids for the files that will be processed.

        Returns:
            ScanResponse: A response object containing the scanning status and doc_ids array
        """
        resolved_workspace_id, current_rag, current_doc_manager = (
            await resolve_request_context(request, workspace_id)
        )
        await ensure_workspace_generation_writable(current_rag.workspace)

        # Scan for new files to get count
        new_files = current_doc_manager.scan_directory_for_new_files()
        
        # Generate doc_ids for the files
        from lightrag.utils import generate_doc_id
        doc_ids = [generate_doc_id("doc-") for _ in new_files]
        
        # Start the scanning process in the background
        background_tasks.add_task(
            run_scanning_process, current_rag, current_doc_manager
        )
        
        return ScanResponse(
            status="scanning_started",
            message=f"Scanning process has been initiated in the background. Found {len(new_files)} files to process.",
            workspace_id=resolved_workspace_id,
            doc_ids=doc_ids,
            total_files=len(new_files),
            queued_files=len(doc_ids),
            failed_files=0,
        )

    @router.post(
        "/upload", response_model=InsertResponse, dependencies=[Depends(combined_auth)]
    )
    @with_workspace_write_lease
    async def upload_to_input_dir(
        background_tasks: BackgroundTasks,
        request: Request,
        file: UploadFile = File(...),
        workspace_id: Optional[str] = Form(None),
        doc_type: Literal["legal", "general"] = Form("general"),
    ):
        """
        Upload a file to the input directory and index it.

        This API endpoint accepts a file through an HTTP POST request, checks if the
        uploaded file is of a supported type, saves it in the specified input directory,
        indexes it for retrieval, and returns a success status with relevant details.

        **File Size Limit:**
        - Configurable via `MAX_UPLOAD_SIZE` environment variable (default: 100MB)
        - Set to `None` or `0` for unlimited upload size
        - Returns HTTP 413 (Request Entity Too Large) if file exceeds limit

        **Document Type Classification:**
        - `doc_type` parameter classifies documents as "legal" or "general" (default: "general")
        - Legal documents use article-based chunking with Vietnamese legal metadata
        - General documents use semantic chunking with breadcrumb context
        - This classification affects chunking strategy and citation formatting

        **Duplicate Detection Behavior:**

        This endpoint handles two types of duplicate scenarios differently:

        1. **Filename Duplicate (Synchronous Detection)**:
           - Detected immediately before file processing
           - Returns `status="duplicated"` with the existing document's doc_id
           - Two cases:
             - If filename exists in document storage: returns existing doc_id
             - If filename exists in file system only: returns empty doc_id ("")

        2. **Content Duplicate (Asynchronous Detection)**:
           - Detected during background processing after content extraction
           - Returns `status="success"` with a new doc_id immediately
           - The duplicate is detected later when processing the file content
           - Use `/documents/{doc_id}` to check the final result:
             - Document will have `status="FAILED"`
             - `error_msg` contains "Content already exists. Original doc_id: xxx"
             - `metadata.is_duplicate=true` with reference to original document
             - `metadata.original_doc_id` points to the existing document

        **Why Different Behavior?**
        - Filename check is fast (simple lookup), done synchronously
        - Content extraction is expensive (PDF/DOCX parsing), done asynchronously
        - This design prevents blocking the client during expensive operations

        Args:
            background_tasks: FastAPI BackgroundTasks for async processing
            file (UploadFile): The file to be uploaded. It must have an allowed extension.
            workspace_id: Optional workspace identifier for multi-tenant support
            doc_type: Document type classification ("legal" or "general", default: "general")

        Returns:
            InsertResponse: A response object containing the upload status and a message.
                - status="success": File accepted and queued for processing
                - status="duplicated": Filename already exists (check doc_id for existing document)

        Raises:
            HTTPException: If the file type is not supported (400), file too large (413), or other errors occur (500).
        """
        try:
            resolved_workspace_id, current_rag, current_doc_manager = (
                await resolve_request_context(request, workspace_id)
            )
            await ensure_workspace_generation_writable(current_rag.workspace)

            # Sanitize filename to prevent Path Traversal attacks
            safe_filename = sanitize_filename(
                file.filename, current_doc_manager.input_dir
            )

            if not current_doc_manager.is_supported_file(safe_filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Supported types: {current_doc_manager.supported_extensions}",
                )

            # Check file size limit (if configured)
            if (
                global_args.max_upload_size is not None
                and global_args.max_upload_size > 0
            ):
                # Safe access to file size (not available in older Starlette versions)
                file_size = getattr(file, "size", None)

                # Pre-flight size check (only if size is available)
                if file_size is not None:
                    if file_size > global_args.max_upload_size:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large. Maximum size: {global_args.max_upload_size / 1024 / 1024:.1f}MB, uploaded: {file_size / 1024 / 1024:.1f}MB",
                        )
                else:
                    # If size not available, we'll check during streaming
                    logger.debug(
                        f"File size not available in UploadFile for {safe_filename}, will check during streaming"
                    )

            # Check if filename already exists in doc_status storage
            existing_doc_data = await current_rag.doc_status.get_doc_by_file_path(
                safe_filename
            )
            retry_failed_file = False
            if existing_doc_data:
                # Get document status and doc_id from existing document
                status = existing_doc_data.get("status", "unknown")
                existing_doc_id = existing_doc_data.get("id", "")
                normalized_status = str(status).strip().lower()

                # Allow retry when previous extraction failed for the same filename.
                if normalized_status == DocStatus.FAILED.value:
                    retry_failed_file = True
                    failed_docs = await current_rag.doc_status.get_docs_by_status(
                        DocStatus.FAILED
                    )
                    failed_doc_ids = [
                        doc_id
                        for doc_id, doc_status in failed_docs.items()
                        if getattr(doc_status, "file_path", "") == safe_filename
                    ]
                    if failed_doc_ids:
                        await current_rag.doc_status.delete(failed_doc_ids)
                        logger.info(
                            f"[File Extraction]Removed {len(failed_doc_ids)} FAILED record(s) for retry upload: {safe_filename}"
                        )
                else:
                    return InsertResponse(
                        status="duplicated",
                        message=f"File '{safe_filename}' already exists in document storage (Status: {status}).",
                        workspace_id=resolved_workspace_id,
                        doc_id=existing_doc_id,
                    )

            file_path = current_doc_manager.input_dir / safe_filename
            # Check if file already exists in file system
            if file_path.exists():
                if retry_failed_file:
                    try:
                        file_path.unlink()
                        logger.info(
                            f"[File Extraction]Removed stale failed file before retry upload: {safe_filename}"
                        )
                    except Exception as cleanup_error:
                        logger.error(
                            f"[File Extraction]Failed to remove stale failed file {safe_filename}: {cleanup_error}"
                        )
                        raise HTTPException(
                            status_code=500,
                            detail=(
                                f"Failed to clean up stale failed file '{safe_filename}' before retry upload"
                            ),
                        )
                else:
                    # If no doc_status record exists but file remains in input dir,
                    # treat it as orphan cache and auto-clean when pipeline is idle.
                    if existing_doc_data is None:
                        from lightrag.kg.shared_storage import (
                            get_namespace_data,
                            get_namespace_lock,
                        )

                        pipeline_status = await get_namespace_data(
                            "pipeline_status", workspace=current_rag.workspace
                        )
                        pipeline_status_lock = get_namespace_lock(
                            "pipeline_status", workspace=current_rag.workspace
                        )

                        async with pipeline_status_lock:
                            pipeline_busy = pipeline_status.get("busy", False)

                        if not pipeline_busy:
                            try:
                                file_path.unlink()
                                logger.warning(
                                    f"[File Extraction]Removed orphan input file without doc_status record: {safe_filename}"
                                )
                            except Exception as cleanup_error:
                                logger.error(
                                    f"[File Extraction]Failed to remove orphan file {safe_filename}: {cleanup_error}"
                                )
                                raise HTTPException(
                                    status_code=500,
                                    detail=(
                                        f"Failed to clean up orphan input file '{safe_filename}' before retry upload"
                                    ),
                                )
                        else:
                            return InsertResponse(
                                status="duplicated",
                                message=(
                                    f"File '{safe_filename}' is already cached in input directory and pipeline is busy. "
                                    "Please wait for processing to finish or cancel the pipeline."
                                ),
                                workspace_id=resolved_workspace_id,
                                doc_id="",
                            )
                    else:
                        return InsertResponse(
                            status="duplicated",
                            message=f"File '{safe_filename}' already exists in the input directory.",
                            workspace_id=resolved_workspace_id,
                            doc_id="",
                        )

            # Async streaming write with size check
            bytes_written = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            needs_cleanup = False

            async with aiofiles.open(file_path, "wb") as out_file:
                while True:
                    # Read chunk from upload stream
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break

                    # Check size limit during streaming (if not checked before)
                    if (
                        global_args.max_upload_size is not None
                        and global_args.max_upload_size > 0
                    ):
                        bytes_written += len(chunk)
                        if bytes_written > global_args.max_upload_size:
                            needs_cleanup = True
                            break

                    # Write chunk to file
                    await out_file.write(chunk)

            # Cleanup after file is closed
            if needs_cleanup:
                try:
                    file_path.unlink()
                except Exception as cleanup_error:
                    logger.error(
                        f"Error cleaning up oversized file {safe_filename}: {cleanup_error}"
                    )

                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {global_args.max_upload_size / 1024 / 1024:.1f}MB, uploaded: {bytes_written / 1024 / 1024:.1f}MB",
                )

            # Generate doc_id and add to background tasks
            from lightrag.utils import generate_doc_id
            doc_id = generate_doc_id("doc-")
            
            background_tasks.add_task(
                pipeline_index_file, current_rag, file_path, doc_id, doc_type=doc_type
            )

            return InsertResponse(
                status="success",
                message=f"File '{safe_filename}' uploaded successfully. Processing will continue in background.",
                workspace_id=resolved_workspace_id,
                doc_id=doc_id,
            )

        except HTTPException:
            # Re-raise HTTP exceptions (400, 413, etc.)
            raise
        except Exception as e:
            logger.error(f"Error /documents/upload: {file.filename}: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/preview",
        dependencies=[Depends(combined_auth)],
        response_model=PreviewDocumentResponse,
        summary="Preview an uploaded document file",
    )
    async def preview_document(
        request: Request,
        file_path: str,
        workspace_id: Optional[str] = None,
        download: bool = False,
    ):
        """Preview or download an uploaded source document.

        Args:
            file_path: Path to the document file
            workspace_id: Optional workspace identifier
            download: If True, returns base64-encoded original file for download.
                     If False (default), returns markdown/text preview only.

        Returns:
            PreviewDocumentResponse with either:
            - download=True: data_base64 populated, data_text is None
            - download=False: data_text populated (if available), data_base64 is None

        Security notes:
        - Path traversal is blocked by `validate_file_path_security`.
        - File access is scoped to the resolved workspace input directory.
        """
        try:
            resolved_workspace_id, current_rag, current_doc_manager = await resolve_request_context(
                request, workspace_id
            )

            normalized_file_path = normalize_file_path(file_path)
            if normalized_file_path == UNKNOWN_FILE_SOURCE:
                raise HTTPException(
                    status_code=400, detail="Invalid file_path for preview"
                )

            # Files are moved to __enqueued__ after successful enqueue; search both locations.
            search_dirs = [
                current_doc_manager.input_dir,
                current_doc_manager.input_dir / "__enqueued__",
            ]

            safe_file_path = None
            for base_dir in search_dirs:
                candidate = validate_file_path_security(normalized_file_path, base_dir)
                if candidate is not None and candidate.exists() and candidate.is_file():
                    safe_file_path = candidate
                    break

            if safe_file_path is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"File '{normalized_file_path}' not found in workspace "
                        f"'{resolved_workspace_id or current_rag.workspace or 'default'}'"
                    ),
                )

            # Default to original file info
            output_name = safe_file_path.name
            
            # Use mimetypes library for better coverage of original file formats
            import mimetypes
            mime_type, _ = mimetypes.guess_type(safe_file_path.name)
            if mime_type is None:
                # Manual fallbacks for common types if guess_type fails (common in some Linux/Docker envs)
                extension = safe_file_path.suffix.lower()
                mapping = {
                    ".pdf": "application/pdf",
                    ".md": "text/markdown",
                    ".txt": "text/plain",
                    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    ".xls": "application/vnd.ms-excel",
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ".doc": "application/msword",
                    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    ".ppt": "application/vnd.ms-powerpoint",
                    ".json": "application/json",
                    ".csv": "text/csv",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg"
                }
                mime_type = mapping.get(extension, "application/octet-stream")
            
            # Initialize response fields
            data_base64: str | None = None
            data_text: str | None = None
            
            if download:
                # --- DOWNLOAD MODE: Return base64 only ---
                async with aiofiles.open(safe_file_path, "rb") as f:
                    original_content_bytes = await f.read()
                
                if original_content_bytes is None:
                    raise HTTPException(status_code=500, detail="Failed to load original document content")
                
                data_base64 = base64.b64encode(original_content_bytes).decode("ascii")
            else:
                # --- PREVIEW MODE: Return markdown/text only ---
                # 1. Try DB-first for markdown (S3/MINIO FULL_DOCS SUPPORT)
                try:
                    existing_doc_data = await current_rag.doc_status.get_doc_by_file_path(
                        normalized_file_path
                    )
                    if existing_doc_data:
                        raw_full_doc = await current_rag.full_docs.get_by_id(existing_doc_data["id"])
                        if raw_full_doc and "content" in raw_full_doc:
                            data_text = raw_full_doc["content"]
                            logger.info(f"Loaded Markdown preview from full_docs (S3/DB) for doc_id: {existing_doc_data['id']}")
                except Exception as e:
                    logger.warning(f"Could not lookup doc status for file_path '{normalized_file_path}': {e}")

                # 2. Try sibling .md file if still no text and it's a PDF
                if data_text is None and safe_file_path.suffix.lower() == ".pdf":
                    md_candidate = safe_file_path.with_suffix(".md")
                    if md_candidate.exists() and md_candidate.is_file():
                        async with aiofiles.open(md_candidate, "r", encoding="utf-8") as f:
                            data_text = await f.read()
                        logger.info(f"Loaded Markdown preview from sibling file: {md_candidate.name}")

                # 3. If the original file itself is a text-based file
                if data_text is None and safe_file_path.suffix.lower() in [".md", ".txt", ".json", ".csv"]:
                    # Load original content for text extraction
                    async with aiofiles.open(safe_file_path, "rb") as f:
                        original_content_bytes = await f.read()
                    try:
                        data_text = original_content_bytes.decode("utf-8")
                    except Exception:
                        data_text = original_content_bytes.decode("latin-1", errors="replace")


            return PreviewDocumentResponse(
                workspace_id=resolved_workspace_id,
                file_path=normalized_file_path,
                file_name=output_name,
                mime_type=mime_type,
                data_base64=data_base64,
                data_text=data_text
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error previewing document '{file_path}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/text", response_model=InsertResponse, dependencies=[Depends(combined_auth)]
    )
    @with_workspace_write_lease
    async def insert_text(
        request: InsertTextRequest,
        background_tasks: BackgroundTasks,
        http_request: Request,
    ):
        """
        Insert text into the RAG system.

        This endpoint allows you to insert text data into the RAG system for later retrieval
        and use in generating responses.
        
        If file_source is not provided or is empty, a title will be automatically generated
        from the text content using LLM and used as the file_source.

        Args:
            request (InsertTextRequest): The request body containing the text to be inserted.
            background_tasks: FastAPI BackgroundTasks for async processing

        Returns:
            InsertResponse: A response object containing the status of the operation.

        Raises:
            HTTPException: If an error occurs during text processing (500).
        """
        try:
            logger.info(f"[insert_text] Starting text insertion, file_source={request.file_source}")
            
            resolved_workspace_id, current_rag, _ = await resolve_request_context(
                http_request, request.workspace_id
            )
            await ensure_workspace_generation_writable(current_rag.workspace)
            logger.info(f"[insert_text] Resolved workspace: {resolved_workspace_id}")

            # Auto-generate title if file_source is not provided or is unknown_source
            if not request.file_source or request.file_source == "unknown_source":
                logger.info("[insert_text] Generating title from text content...")
                try:
                    generated_title = await generate_title_from_text(request.text, current_rag)
                    request.file_source = f"{generated_title}.txt"
                    logger.info(f"[insert_text] Generated title: {request.file_source}")
                except Exception as e:
                    logger.error(f"[insert_text] Error generating title: {str(e)}")
                    # Use fallback title
                    request.file_source = "Untitled Document.txt"

            logger.info(f"[insert_text] Checking for existing file_source: {request.file_source}")
            # Check if file_source already exists in doc_status storage
            if (
                request.file_source
                and request.file_source.strip()
                and request.file_source != "unknown_source"
            ):
                existing_doc_data = await current_rag.doc_status.get_doc_by_file_path(
                    request.file_source
                )
                if existing_doc_data:
                    # Get document status and doc_id from existing document
                    status = existing_doc_data.get("status", "unknown")
                    existing_doc_id = existing_doc_data.get("id", "")
                    logger.info("[insert_text] File source already exists, returning duplicated")
                    return InsertResponse(
                        status="duplicated",
                        message=f"File source '{request.file_source}' already exists in document storage (Status: {status}).",
                        workspace_id=resolved_workspace_id,
                        doc_id=existing_doc_id,
                    )

            logger.info("[insert_text] Checking for duplicate content...")
            # Check if content already exists by computing content hash (doc_id)
            sanitized_text = sanitize_text_for_encoding(request.text)
            content_doc_id = compute_mdhash_id(sanitized_text, prefix="doc-")
            existing_doc = await current_rag.doc_status.get_by_id(content_doc_id)
            if existing_doc:
                # Content already exists, return duplicated with existing doc_id
                status = existing_doc.get("status", "unknown")
                existing_doc_id = existing_doc.get("id", content_doc_id)
                logger.info("[insert_text] Content already exists, returning duplicated")
                return InsertResponse(
                    status="duplicated",
                    message=f"Identical content already exists in document storage (doc_id: {content_doc_id}, Status: {status}).",
                    workspace_id=resolved_workspace_id,
                    doc_id=existing_doc_id,
                )

            # Generate doc_id for text insertion
            from lightrag.utils import generate_doc_id
            doc_id = generate_doc_id("doc-")
            logger.info(f"[insert_text] Adding background task for text insertion with doc_id: {doc_id}")

            background_tasks.add_task(
                pipeline_index_texts,
                current_rag,
                [request.text],
                file_sources=[request.file_source],
                doc_ids=[doc_id],
                doc_type=request.doc_type,
            )

            logger.info("[insert_text] Returning success response")
            return InsertResponse(
                status="success",
                message=f"Text successfully received with title '{request.file_source}'. Processing will continue in background.",
                workspace_id=resolved_workspace_id,
                doc_id=doc_id,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error /documents/text: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/texts",
        response_model=BatchInsertResponse,
        dependencies=[Depends(combined_auth)],
    )
    @with_workspace_write_lease
    async def insert_texts(
        request: InsertTextsRequest,
        background_tasks: BackgroundTasks,
        http_request: Request,
    ):
        """
        Insert multiple texts into the RAG system.

        This endpoint allows you to insert multiple text entries into the RAG system
        in a single request.
        
        If file_sources is not provided or has fewer entries than texts, titles will be
        automatically generated from the text content using LLM for the missing entries.

        Args:
            request (InsertTextsRequest): The request body containing the list of texts.
            background_tasks: FastAPI BackgroundTasks for async processing

        Returns:
            BatchInsertResponse: A response object containing the status and doc_ids array.

        Raises:
            HTTPException: If an error occurs during text processing (500).
        """
        try:
            resolved_workspace_id, current_rag, _ = await resolve_request_context(
                http_request, request.workspace_id
            )
            await ensure_workspace_generation_writable(current_rag.workspace)

            # Auto-generate titles for texts without file_sources
            if not request.file_sources:
                request.file_sources = []
            
            # Ensure file_sources list has same length as texts
            while len(request.file_sources) < len(request.texts):
                request.file_sources.append("unknown_source")
            
            # Generate titles for unknown_source entries
            for i, file_source in enumerate(request.file_sources):
                if not file_source or file_source == "unknown_source":
                    logger.info(f"Generating title for text {i+1}/{len(request.texts)}...")
                    generated_title = await generate_title_from_text(request.texts[i], current_rag)
                    request.file_sources[i] = f"{generated_title}.txt"
                    logger.info(f"Generated title: {request.file_sources[i]}")

            # Check if any file_sources already exist in doc_status storage
            if request.file_sources:
                for file_source in request.file_sources:
                    if (
                        file_source
                        and file_source.strip()
                        and file_source != "unknown_source"
                    ):
                        existing_doc_data = (
                            await current_rag.doc_status.get_doc_by_file_path(
                                file_source
                            )
                        )
                        if existing_doc_data:
                            # Get document status and doc_id from existing document
                            status = existing_doc_data.get("status", "unknown")
                            existing_doc_id = existing_doc_data.get("id", "")
                            # For batch operations, return failure for any duplicate
                            return BatchInsertResponse(
                                status="failure",
                                message=f"File source '{file_source}' already exists in document storage (Status: {status}).",
                                workspace_id=resolved_workspace_id,
                                doc_ids=[existing_doc_id] if existing_doc_id else [],
                                total_files=len(request.texts),
                                queued_files=0,
                                failed_files=len(request.texts),
                            )

            # Check if any content already exists by computing content hash (doc_id)
            for text in request.texts:
                sanitized_text = sanitize_text_for_encoding(text)
                content_doc_id = compute_mdhash_id(sanitized_text, prefix="doc-")
                existing_doc = await current_rag.doc_status.get_by_id(content_doc_id)
                if existing_doc:
                    # Content already exists, return failure with existing doc_id
                    status = existing_doc.get("status", "unknown")
                    existing_doc_id = existing_doc.get("id", content_doc_id)
                    return BatchInsertResponse(
                        status="failure",
                        message=f"Identical content already exists in document storage (doc_id: {content_doc_id}, Status: {status}).",
                        workspace_id=resolved_workspace_id,
                        doc_ids=[existing_doc_id],
                        total_files=len(request.texts),
                        queued_files=0,
                        failed_files=len(request.texts),
                    )

            # Generate doc_ids for texts insertion
            from lightrag.utils import generate_doc_id
            doc_ids = [generate_doc_id("doc-") for _ in request.texts]

            background_tasks.add_task(
                pipeline_index_texts,
                current_rag,
                request.texts,
                file_sources=request.file_sources,
                doc_ids=doc_ids,
                doc_type=request.doc_type,
            )

            return BatchInsertResponse(
                status="success",
                message=f"{len(request.texts)} texts successfully received with auto-generated titles. Processing will continue in background.",
                workspace_id=resolved_workspace_id,
                doc_ids=doc_ids,
                total_files=len(request.texts),
                queued_files=len(doc_ids),
                failed_files=0,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error /documents/texts: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete(
        "", response_model=ClearDocumentsResponse, dependencies=[Depends(combined_auth)]
    )
    @with_workspace_write_lease
    async def clear_documents(
        request: Request, workspace_id: Optional[str] = None
    ):
        """
        Clear all documents from the RAG system.

        This endpoint deletes all documents, entities, relationships, and files from the system.
        It uses the storage drop methods to properly clean up all data and removes all files
        from the input directory.

        Returns:
            ClearDocumentsResponse: A response object containing the status and message.
                - status="success":           All documents and files were successfully cleared.
                - status="partial_success":   Document clear job exit with some errors.
                - status="busy":              Operation could not be completed because the pipeline is busy.
                - status="fail":              All storage drop operations failed, with message
                - message: Detailed information about the operation results, including counts
                  of deleted files and any errors encountered.

        Raises:
            HTTPException: Raised when a serious error occurs during the clearing process,
                          with status code 500 and error details in the detail field.
        """
        resolved_workspace_id, current_rag, current_doc_manager = (
            await resolve_request_context(request, workspace_id)
        )
        await ensure_workspace_generation_writable(current_rag.workspace)

        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_namespace_lock,
        )

        # Get pipeline status and lock
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=current_rag.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=current_rag.workspace
        )

        # Check and set status with lock
        async with pipeline_status_lock:
            if pipeline_status.get("busy", False):
                return ClearDocumentsResponse(
                    status="busy",
                    workspace_id=resolved_workspace_id,
                    message="Cannot clear documents while pipeline is busy",
                )
            # Set busy to true
            pipeline_status.update(
                {
                    "busy": True,
                    "job_name": "Clearing Documents",
                    "job_start": datetime.now().isoformat(),
                    "docs": 0,
                    "batchs": 0,
                    "cur_batch": 0,
                    "request_pending": False,  # Clear any previous request
                    "latest_message": "Starting document clearing process",
                }
            )
            # Cleaning history_messages without breaking it as a shared list object
            del pipeline_status["history_messages"][:]
            pipeline_status["history_messages"].append(
                "Starting document clearing process"
            )

        try:
            # Use drop method to clear all data
            drop_tasks = []
            storages = [
                current_rag.text_chunks,
                current_rag.full_docs,
                current_rag.full_entities,
                current_rag.full_relations,
                current_rag.entity_chunks,
                current_rag.relation_chunks,
                current_rag.entities_vdb,
                current_rag.relationships_vdb,
                current_rag.chunks_vdb,
                current_rag.chunk_entity_relation_graph,
                current_rag.doc_status,
            ]

            # Log storage drop start
            if "history_messages" in pipeline_status:
                pipeline_status["history_messages"].append(
                    "Starting to drop storage components"
                )

            for storage in storages:
                if storage is not None:
                    drop_tasks.append(storage.drop())

            # Wait for all drop tasks to complete
            drop_results = await asyncio.gather(*drop_tasks, return_exceptions=True)

            # Check for errors and log results
            errors = []
            storage_success_count = 0
            storage_error_count = 0

            for i, result in enumerate(drop_results):
                storage_name = storages[i].__class__.__name__
                if isinstance(result, Exception):
                    error_msg = f"Error dropping {storage_name}: {str(result)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    storage_error_count += 1
                else:
                    namespace = storages[i].namespace
                    workspace = storages[i].workspace
                    logger.info(
                        f"Successfully dropped {storage_name}: {workspace}/{namespace}"
                    )
                    storage_success_count += 1

            # Log storage drop results
            if "history_messages" in pipeline_status:
                if storage_error_count > 0:
                    pipeline_status["history_messages"].append(
                        f"Dropped {storage_success_count} storage components with {storage_error_count} errors"
                    )
                else:
                    pipeline_status["history_messages"].append(
                        f"Successfully dropped all {storage_success_count} storage components"
                    )

            # If all storage operations failed, return error status and don't proceed with file deletion
            if storage_success_count == 0 and storage_error_count > 0:
                error_message = "All storage drop operations failed. Aborting document clearing process."
                logger.error(error_message)
                if "history_messages" in pipeline_status:
                    pipeline_status["history_messages"].append(error_message)
                return ClearDocumentsResponse(
                    status="fail",
                    workspace_id=resolved_workspace_id,
                    message=error_message,
                )

            # Log file deletion start
            if "history_messages" in pipeline_status:
                pipeline_status["history_messages"].append(
                    "Starting to delete files in input directory"
                )

            # Delete only files in the current directory, preserve files in subdirectories
            deleted_files_count = 0
            file_errors_count = 0

            for file_path in current_doc_manager.input_dir.glob("*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_files_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting file {file_path}: {str(e)}")
                        file_errors_count += 1

            # Log file deletion results
            if "history_messages" in pipeline_status:
                if file_errors_count > 0:
                    pipeline_status["history_messages"].append(
                        f"Deleted {deleted_files_count} files with {file_errors_count} errors"
                    )
                    errors.append(f"Failed to delete {file_errors_count} files")
                else:
                    pipeline_status["history_messages"].append(
                        f"Successfully deleted {deleted_files_count} files"
                    )

            # Prepare final result message
            final_message = ""
            if errors:
                final_message = f"Cleared documents with some errors. Deleted {deleted_files_count} files."
                status = "partial_success"
            else:
                final_message = f"All documents cleared successfully. Deleted {deleted_files_count} files."
                status = "success"

            # Log final result
            if "history_messages" in pipeline_status:
                pipeline_status["history_messages"].append(final_message)

            # Return response based on results
            return ClearDocumentsResponse(
                status=status,
                workspace_id=resolved_workspace_id,
                message=final_message,
            )
        except Exception as e:
            error_msg = f"Error clearing documents: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            if "history_messages" in pipeline_status:
                pipeline_status["history_messages"].append(error_msg)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Reset busy status after completion
            async with pipeline_status_lock:
                pipeline_status["busy"] = False
                completion_msg = "Document clearing process completed"
                pipeline_status["latest_message"] = completion_msg
                if "history_messages" in pipeline_status:
                    pipeline_status["history_messages"].append(completion_msg)

    @router.get(
        "/pipeline_status",
        dependencies=[Depends(combined_auth)],
        response_model=PipelineStatusResponse,
    )
    async def get_pipeline_status(
        request: Request, workspace_id: Optional[str] = None
    ) -> PipelineStatusResponse:
        """
        Get the current status of the document indexing pipeline.

        This endpoint returns information about the current state of the document processing pipeline,
        including the processing status, progress information, and history messages.

        Returns:
            PipelineStatusResponse: A response object containing:
                - autoscanned (bool): Whether auto-scan has started
                - busy (bool): Whether the pipeline is currently busy
                - job_name (str): Current job name (e.g., indexing files/indexing texts)
                - job_start (str, optional): Job start time as ISO format string
                - docs (int): Total number of documents to be indexed
                - batchs (int): Number of batches for processing documents
                - cur_batch (int): Current processing batch
                - request_pending (bool): Flag for pending request for processing
                - latest_message (str): Latest message from pipeline processing
                - history_messages (List[str], optional): List of history messages (limited to latest 1000 entries,
                  with truncation message if more than 1000 messages exist)

        Raises:
            HTTPException: If an error occurs while retrieving pipeline status (500)
        """
        try:
            resolved_workspace_id, current_rag, _ = await resolve_request_context(
                request, workspace_id
            )

            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_namespace_lock,
                get_all_update_flags_status,
            )

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=current_rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=current_rag.workspace
            )

            # Get update flags status for all namespaces
            update_status = await get_all_update_flags_status(
                workspace=current_rag.workspace
            )

            # Convert MutableBoolean objects to regular boolean values
            processed_update_status = {}
            for namespace, flags in update_status.items():
                processed_flags = []
                for flag in flags:
                    # Handle both multiprocess and single process cases
                    if hasattr(flag, "value"):
                        processed_flags.append(bool(flag.value))
                    else:
                        processed_flags.append(bool(flag))
                processed_update_status[namespace] = processed_flags

            async with pipeline_status_lock:
                # Convert to regular dict if it's a Manager.dict
                status_dict = dict(pipeline_status)

            # Add processed update_status to the status dictionary
            status_dict["update_status"] = processed_update_status
            status_dict["workspace_id"] = resolved_workspace_id

            # Convert history_messages to a regular list if it's a Manager.list
            # and limit to latest 1000 entries with truncation message if needed
            if "history_messages" in status_dict:
                history_list = list(status_dict["history_messages"])
                total_count = len(history_list)

                if total_count > 1000:
                    # Calculate truncated message count
                    truncated_count = total_count - 1000

                    # Take only the latest 1000 messages
                    latest_messages = history_list[-1000:]

                    # Add truncation message at the beginning
                    truncation_message = (
                        f"[Truncated history messages: {truncated_count}/{total_count}]"
                    )
                    status_dict["history_messages"] = [
                        truncation_message
                    ] + latest_messages
                else:
                    # No truncation needed, return all messages
                    status_dict["history_messages"] = history_list

            # Ensure job_start is properly formatted as a string with timezone information
            if "job_start" in status_dict and status_dict["job_start"]:
                # Use format_datetime to ensure consistent formatting
                status_dict["job_start"] = format_datetime(status_dict["job_start"])

            return PipelineStatusResponse(**status_dict)
        except Exception as e:
            logger.error(f"Error getting pipeline status: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    # TODO: Deprecated, use /documents/paginated instead
    @router.get(
        "", response_model=DocsStatusesResponse, dependencies=[Depends(combined_auth)]
    )
    async def documents(
        request: Request, workspace_id: Optional[str] = None
    ) -> DocsStatusesResponse:
        """
        Get the status of all documents in the system. This endpoint is deprecated; use /documents/paginated instead.
        To prevent excessive resource consumption, a maximum of 1,000 records is returned.

        This endpoint retrieves the current status of all documents, grouped by their
        processing status (PENDING, PROCESSING, PREPROCESSED, PROCESSED, FAILED). The results are
        limited to 1000 total documents with fair distribution across all statuses.

        Returns:
            DocsStatusesResponse: A response object containing a dictionary where keys are
                                DocStatus values and values are lists of DocStatusResponse
                                objects representing documents in each status category.
                                Maximum 1000 documents total will be returned.

        Raises:
            HTTPException: If an error occurs while retrieving document statuses (500).
        """
        try:
            _, current_rag, _ = await resolve_request_context(request, workspace_id)

            statuses = (
                DocStatus.UPLOADING,
                DocStatus.EXTRACTING,
                DocStatus.EXTRACTED,
                DocStatus.CHUNKING,
                DocStatus.PROCESSED,
                DocStatus.FAILED,
            )

            tasks = [current_rag.get_docs_by_status(status) for status in statuses]
            results: List[Dict[str, DocProcessingStatus]] = await asyncio.gather(*tasks)

            response = DocsStatusesResponse()
            total_documents = 0
            max_documents = 1000

            # Convert results to lists for easier processing
            status_documents = []
            for idx, result in enumerate(results):
                status = statuses[idx]
                docs_list = []
                for doc_id, doc_status in result.items():
                    docs_list.append((doc_id, doc_status))
                status_documents.append((status, docs_list))

            # Fair distribution: round-robin across statuses
            status_indices = [0] * len(
                status_documents
            )  # Track current index for each status
            current_status_idx = 0

            while total_documents < max_documents:
                # Check if we have any documents left to process
                has_remaining = False
                for status_idx, (status, docs_list) in enumerate(status_documents):
                    if status_indices[status_idx] < len(docs_list):
                        has_remaining = True
                        break

                if not has_remaining:
                    break

                # Try to get a document from the current status
                status, docs_list = status_documents[current_status_idx]
                current_index = status_indices[current_status_idx]

                if current_index < len(docs_list):
                    doc_id, doc_status = docs_list[current_index]
                    
                    # Skip track documents (id starts with "track-")
                    if doc_id.startswith("track-"):
                        status_indices[current_status_idx] += 1
                        current_status_idx = (current_status_idx + 1) % len(status_documents)
                        continue

                    if status not in response.statuses:
                        response.statuses[status] = []

                    response.statuses[status].append(
                        DocStatusResponse(
                            id=doc_id,
                            content_summary=doc_status.content_summary,
                            content_length=doc_status.content_length,
                            status=doc_status.status,
                            created_at=format_datetime(doc_status.created_at),
                            updated_at=format_datetime(doc_status.updated_at),
                            workspace_id=current_rag.workspace,
                            chunks_count=doc_status.chunks_count,
                            error_msg=doc_status.error_msg,
                            metadata=doc_status.metadata,
                            file_path=normalize_file_path(doc_status.file_path),
                        )
                    )

                    status_indices[current_status_idx] += 1
                    total_documents += 1

                # Move to next status (round-robin)
                current_status_idx = (current_status_idx + 1) % len(status_documents)

            return response
        except Exception as e:
            logger.error(f"Error GET /documents: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    class DeleteDocByIdResponse(BaseModel):
        """Response model for single document deletion operation."""

        status: Literal["deletion_started", "busy", "not_allowed"] = Field(
            description="Status of the deletion operation"
        )
        message: str = Field(description="Message describing the operation result")
        doc_id: str = Field(description="The ID of the document to delete")

    @router.delete(
        "/delete_document",
        response_model=DeleteDocByIdResponse,
        dependencies=[Depends(combined_auth)],
        summary="Delete documents by IDs within a workspace.",
    )
    @with_workspace_write_lease
    async def delete_document(
        delete_request: DeleteDocRequest,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> DeleteDocByIdResponse:
        """
        Delete documents and all their associated data by their IDs using background processing.

        Deletes specific documents and all their associated data, including their status,
        text chunks, vector embeddings, and any related graph data. When requested,
        cached LLM extraction responses are removed after graph deletion/rebuild completes.
        The deletion process runs in the background to avoid blocking the client connection.

        This operation is irreversible and will interact with the pipeline status.

        Args:
            delete_request (DeleteDocRequest): The request containing workspace_id, document IDs, and deletion options.
            background_tasks: FastAPI BackgroundTasks for async processing

        Returns:
            DeleteDocByIdResponse: The result of the deletion operation.
                - status="deletion_started": The document deletion has been initiated in the background.
                - status="busy": The pipeline is busy with another operation.

        Raises:
            HTTPException:
              - 500: If an unexpected internal error occurs during initialization.
        """
        doc_ids = delete_request.doc_ids
        workspace_id = (
            delete_request.workspace_id
            or request.query_params.get("workspace_id")
            or "default"
        )

        try:
            _, current_rag, current_doc_manager = await resolve_request_context(
                request, workspace_id
            )
            await ensure_workspace_generation_writable(current_rag.workspace)

            # Validate all documents belong to the resolved workspace before enqueueing deletion.
            existing_docs = await asyncio.gather(
                *(current_rag.doc_status.get_by_id(doc_id) for doc_id in doc_ids)
            )
            missing_doc_ids = [
                doc_id for doc_id, doc_data in zip(doc_ids, existing_docs) if not doc_data
            ]
            if missing_doc_ids:
                return DeleteDocByIdResponse(
                    status="not_allowed",
                    message=(
                        "The following document IDs do not belong to workspace "
                        f"'{current_rag.workspace}': {', '.join(missing_doc_ids)}"
                    ),
                    doc_id=", ".join(missing_doc_ids),
                )

            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_namespace_lock,
            )

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=current_rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=current_rag.workspace
            )

            # Check if pipeline is busy with proper lock
            async with pipeline_status_lock:
                if pipeline_status.get("busy", False):
                    return DeleteDocByIdResponse(
                        status="busy",
                        message="Cannot delete documents while pipeline is busy",
                        doc_id=", ".join(doc_ids),
                    )

            # Add deletion task to background tasks
            background_tasks.add_task(
                background_delete_documents,
                current_rag,
                current_doc_manager,
                doc_ids,
                delete_request.delete_file,
                delete_request.delete_llm_cache,
            )

            return DeleteDocByIdResponse(
                status="deletion_started",
                message=f"Document deletion for {len(doc_ids)} documents has been initiated. Processing will continue in background.",
                doc_id=", ".join(doc_ids),
            )

        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Error initiating document deletion for {delete_request.doc_ids}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)

    @router.post(
        "/reembed",
        response_model=ReembedResponse,
        dependencies=[Depends(combined_auth)],
        summary="Rebuild chunk/entity/relation vectors for a workspace.",
    )
    @with_workspace_write_lease
    async def reembed_documents(
        reembed_request: ReembedRequest,
        request: Request,
    ) -> ReembedResponse:
        """Rebuild vector storages from existing documents and KG data in workspace."""
        try:
            _, current_rag, _ = await resolve_request_context(
                request, reembed_request.workspace_id
            )
            await ensure_workspace_generation_writable(current_rag.workspace)

            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_namespace_lock,
            )

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=current_rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=current_rag.workspace
            )

            async with pipeline_status_lock:
                if pipeline_status.get("busy", False):
                    return ReembedResponse(
                        status="busy",
                        message="Cannot re-embed while pipeline is busy",
                        workspace_id=current_rag.workspace,
                    )

                pipeline_status["busy"] = True
                pipeline_status["job_name"] = "reembed_vectors"
                pipeline_status["latest_message"] = "Re-embedding vectors started"
                pipeline_status.setdefault("history_messages", []).append(
                    "Re-embedding vectors started"
                )

            try:
                stats = await reembed_workspace_vectors(
                    current_rag, batch_size=reembed_request.batch_size
                )
            finally:
                async with pipeline_status_lock:
                    pipeline_status["busy"] = False
                    pipeline_status["job_name"] = ""

            success_message = (
                "Re-embedding completed: "
                f"documents={stats['documents']}, chunks={stats['chunks']}, "
                f"entities={stats['entities']}, relations={stats['relations']}"
            )

            async with pipeline_status_lock:
                pipeline_status["latest_message"] = success_message
                pipeline_status.setdefault("history_messages", []).append(success_message)

            return ReembedResponse(
                status="success",
                message=success_message,
                workspace_id=current_rag.workspace,
                documents=stats["documents"],
                chunks=stats["chunks"],
                entities=stats["entities"],
                relations=stats["relations"],
            )
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Error re-embedding workspace vectors: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return ReembedResponse(
                status="fail",
                message=error_msg,
                workspace_id=reembed_request.workspace_id,
            )

    @router.post(
        "/clear_cache",
        response_model=ClearCacheResponse,
        dependencies=[Depends(combined_auth)],
    )
    @with_workspace_write_lease
    async def clear_cache(request: ClearCacheRequest):
        """
        Clear all cache data from the LLM response cache storage.

        This endpoint clears all cached LLM responses regardless of mode.
        The request body is accepted for API compatibility but is ignored.

        Args:
            request (ClearCacheRequest): The request body (ignored for compatibility).

        Returns:
            ClearCacheResponse: A response object containing the status and message.

        Raises:
            HTTPException: If an error occurs during cache clearing (500).
        """
        try:
            # Call the aclear_cache method (no modes parameter)
            await rag.aclear_cache()

            # Prepare success message
            message = "Successfully cleared all cache"

            return ClearCacheResponse(status="success", message=message)
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete(
        "/delete_entity",
        response_model=DeletionResult,
        dependencies=[Depends(combined_auth)],
    )
    @with_workspace_write_lease
    async def delete_entity(request: DeleteEntityRequest):
        """
        Delete an entity and all its relationships from the knowledge graph.

        Args:
            request (DeleteEntityRequest): The request body containing the entity name.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.

        Raises:
            HTTPException: If the entity is not found (404) or an error occurs (500).
        """
        try:
            await ensure_workspace_generation_writable(rag.workspace)
            result = await rag.adelete_by_entity(entity_name=request.entity_name)
            if result.status == "not_found":
                raise HTTPException(status_code=404, detail=result.message)
            if result.status == "fail":
                raise HTTPException(status_code=500, detail=result.message)
            # Set doc_id to empty string since this is an entity operation, not document
            result.doc_id = ""
            return result
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Error deleting entity '{request.entity_name}': {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)

    @router.delete(
        "/delete_relation",
        response_model=DeletionResult,
        dependencies=[Depends(combined_auth)],
    )
    @with_workspace_write_lease
    async def delete_relation(request: DeleteRelationRequest):
        """
        Delete a relationship between two entities from the knowledge graph.

        Args:
            request (DeleteRelationRequest): The request body containing the source and target entity names.

        Returns:
            DeletionResult: An object containing the outcome of the deletion process.

        Raises:
            HTTPException: If the relation is not found (404) or an error occurs (500).
        """
        try:
            await ensure_workspace_generation_writable(rag.workspace)
            result = await rag.adelete_by_relation(
                source_entity=request.source_entity,
                target_entity=request.target_entity,
            )
            if result.status == "not_found":
                raise HTTPException(status_code=404, detail=result.message)
            if result.status == "fail":
                raise HTTPException(status_code=500, detail=result.message)
            # Set doc_id to empty string since this is a relation operation, not document
            result.doc_id = ""
            return result
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Error deleting relation from '{request.source_entity}' to '{request.target_entity}': {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)

    @router.get(
        "/track_status/{track_id}",
        dependencies=[Depends(combined_auth)],
        deprecated=True,
    )
    async def get_track_status(
        track_id: str,
        request: Request,
        workspace_id: Optional[str] = None,
    ):
        """
        [DEPRECATED] Get the processing status of documents by tracking ID.
        
        This endpoint is deprecated and has been removed.
        Use individual document status queries with doc_id instead.
        
        Migration Guide:
        - Instead of tracking by track_id, track individual documents by their doc_id
        - Use GET /documents/{doc_id} to query individual document status
        - Use POST /documents/paginated to query multiple documents

        Args:
            track_id (str): The tracking ID (no longer supported)

        Returns:
            TrackStatusResponse: Empty response with deprecation notice

        Raises:
            HTTPException: 410 Gone - This endpoint has been removed
        """
        raise HTTPException(
            status_code=410,
            detail="This endpoint has been removed. Track documents individually using their doc_id instead. "
                   "Use GET /documents/{doc_id} for individual status or POST /documents/paginated for multiple documents."
        )

    @router.post(
        "/paginated",
        response_model=PaginatedDocsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_documents_paginated(
        request: DocumentsRequest,
        http_request: Request,
    ) -> PaginatedDocsResponse:
        """
        Get documents with pagination support.

        This endpoint retrieves documents with pagination, filtering, and sorting capabilities.
        It provides better performance for large document collections by loading only the
        requested page of data.

        Args:
            request (DocumentsRequest): The request body containing pagination parameters

        Returns:
            PaginatedDocsResponse: A response object containing:
                - documents: List of documents for the current page
                - pagination: Pagination information (page, total_count, etc.)
                - status_counts: Count of documents by status for all documents

        Raises:
            HTTPException: If an error occurs while retrieving documents (500).
        """
        try:
            resolved_workspace_id, current_rag, _ = await resolve_request_context(
                http_request, request.workspace_id
            )

            # Get paginated documents and status counts in parallel
            docs_task = current_rag.doc_status.get_docs_paginated(
                status_filter=request.status_filter,
                page=request.page,
                page_size=request.page_size,
                sort_field=request.sort_field,
                sort_direction=request.sort_direction,
            )
            status_counts_task = current_rag.doc_status.get_all_status_counts()

            # Execute both queries in parallel
            (documents_with_ids, total_count), status_counts = await asyncio.gather(
                docs_task, status_counts_task
            )

            # Convert documents to response format
            doc_responses = []
            for doc_id, doc in documents_with_ids:
                doc_responses.append(
                    DocStatusResponse(
                        id=doc_id,
                        content_summary=doc.content_summary or "",
                        content_length=doc.content_length or 0,
                        status=doc.status,
                        created_at=format_datetime(doc.created_at),
                        updated_at=format_datetime(doc.updated_at),
                        workspace_id=current_rag.workspace,
                        chunks_count=doc.chunks_count,
                        error_msg=doc.error_msg,
                        metadata=doc.metadata,
                        file_path=normalize_file_path(doc.file_path),
                    )
                )

            # Calculate pagination info
            total_pages = (total_count + request.page_size - 1) // request.page_size
            has_next = request.page < total_pages
            has_prev = request.page > 1

            pagination = PaginationInfo(
                page=request.page,
                page_size=request.page_size,
                total_count=total_count,
                total_pages=total_pages,
                has_next=has_next,
                has_prev=has_prev,
            )

            return PaginatedDocsResponse(
                documents=doc_responses,
                workspace_id=resolved_workspace_id,
                pagination=pagination,
                status_counts=status_counts,
            )

        except Exception as e:
            logger.error(f"Error getting paginated documents: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/search_by_name",
        response_model=DocumentSearchByNameResponse,
        dependencies=[Depends(combined_auth)],
        summary="Search documents by file name substring within a workspace",
    )
    async def search_documents_by_name(
        request: DocumentSearchByNameRequest,
        http_request: Request,
    ) -> DocumentSearchByNameResponse:
        """
        Search documents in a workspace where `doc_name` (stored as `file_path` in doc status)
        contains the given keyword (case-insensitive substring match).
        """
        try:
            resolved_workspace_id, current_rag, _ = await resolve_request_context(
                http_request, request.workspace_id
            )

            keyword = (request.keyword or "").strip()
            keyword_lower = keyword.lower()

            # Scan all documents in workspace using doc_status pagination, then filter in-memory.
            # NOTE: For large workspaces, this can be expensive.
            scan_page_size = 200
            scan_page = 1
            total_docs = None
            matched_docs: List[DocStatusResponse] = []

            while True:
                (docs_with_ids, total_count) = await current_rag.doc_status.get_docs_paginated(
                    status_filter=None,
                    page=scan_page,
                    page_size=scan_page_size,
                    sort_field="updated_at",
                    sort_direction="desc",
                )

                if total_docs is None:
                    total_docs = total_count

                if not docs_with_ids:
                    break

                for doc_id, doc in docs_with_ids:
                    file_path = normalize_file_path(getattr(doc, "file_path", None))
                    if keyword_lower in (file_path or "").lower():
                        matched_docs.append(
                            DocStatusResponse(
                                id=doc_id,
                                content_summary=doc.content_summary,
                                content_length=doc.content_length,
                                status=doc.status,
                                created_at=format_datetime(doc.created_at),
                                updated_at=format_datetime(doc.updated_at),
                                workspace_id=current_rag.workspace,
                                chunks_count=doc.chunks_count,
                                error_msg=doc.error_msg,
                                metadata=doc.metadata,
                                file_path=file_path,
                            )
                        )

                if total_docs is not None and scan_page * scan_page_size >= total_docs:
                    break

                scan_page += 1

            total_matches = len(matched_docs)

            # Paginate results over matches
            start_idx = (request.page - 1) * request.page_size
            end_idx = start_idx + request.page_size
            page_docs = matched_docs[start_idx:end_idx]

            total_pages = (
                (total_matches + request.page_size - 1) // request.page_size
                if request.page_size > 0
                else 0
            )
            has_next = request.page < total_pages
            has_prev = request.page > 1

            return DocumentSearchByNameResponse(
                workspace_id=resolved_workspace_id,
                keyword=keyword,
                total_matches=total_matches,
                pagination=PaginationInfo(
                    page=request.page,
                    page_size=request.page_size,
                    total_count=total_matches,
                    total_pages=total_pages,
                    has_next=has_next,
                    has_prev=has_prev,
                ),
                documents=page_docs,
            )
        except Exception as e:
            logger.error(f"Error searching documents by name: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/status_counts",
        response_model=StatusCountsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_document_status_counts(
        request: Request, 
        workspace_id: Optional[str] = None,
        count_by: str = "document"  # Only "document" is supported (track_id removed)
    ) -> StatusCountsResponse:
        """
        Get counts by status.

        This endpoint retrieves the count in each processing status.
        Counts by document_id (each document = 1 count).
        
        Args:
            count_by: "document" to count by doc_id (each document = 1 count)
                     Note: "track" option has been removed as track_id is deprecated
        
        Status categories (simplified 6-state model):
        - UPLOADING: File đang được upload (reserved for future use)
        - EXTRACTING: Đang extract full text (OCR/Docling) - handled by extraction worker
        - EXTRACTED: Đã extract xong, chờ chunking + KG - ready for chunking worker
        - CHUNKING: Đang chunking + extract entities - handled by chunking/KG worker
        - PROCESSED: Hoàn thành tất cả - final state
        - FAILED: Lỗi ở bất kỳ stage nào - can be reprocessed
        
        Legacy statuses (deprecated): PENDING, PROCESSING, PREPROCESSED, CHUNKED

        Returns:
            StatusCountsResponse: A response object containing status counts

        Raises:
            HTTPException: If an error occurs while retrieving status counts (500).
        """
        try:
            # Check if workspace was explicitly provided
            is_explicit = (workspace_id is not None) or bool(
                request.headers.get("LIGHTRAG-WORKSPACE-ID")
            ) or bool(request.headers.get("LIGHTRAG-WORKSPACE"))

            resolved_workspace_id, current_rag, _ = await resolve_request_context(
                request, workspace_id
            )

            workspaces_stats = None
            
            # Only document-based counting is supported (track_id removed)
            if count_by == "track":
                # Return error for deprecated track-based counting
                raise HTTPException(
                    status_code=400,
                    detail="count_by='track' is no longer supported. track_id has been removed. Use count_by='document' instead."
                )
            
            # Document-based counting
            if not is_explicit:
                workspaces_stats = (
                    await current_rag.doc_status.get_status_counts_across_workspaces()
                )
                total_counts = {"all": 0}
                for ws_stats in workspaces_stats.values():
                    for status, count in ws_stats.items():
                        total_counts[status] = total_counts.get(status, 0) + count
                status_counts = total_counts
            else:
                status_counts = await current_rag.doc_status.get_all_status_counts()

            return StatusCountsResponse(
                workspace_id=resolved_workspace_id,
                status_counts=status_counts,
                workspaces=workspaces_stats,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting document status counts: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/reprocess_failed",
        response_model=ReprocessResponse,
        dependencies=[Depends(combined_auth)],
    )
    @with_workspace_write_lease
    async def reprocess_failed_documents(
        background_tasks: BackgroundTasks,
        request: Request,
        reprocess_req: Optional[ReprocessRequest] = None,
        workspace_id: Optional[str] = None,
    ):
        """
        Reprocess failed and stuck documents.

        This endpoint triggers the document processing pipeline which automatically
        picks up and reprocesses documents in the following statuses:
        - FAILED: Documents that failed during previous processing attempts
        - EXTRACTING: Documents stuck in extraction phase (e.g., worker crashes)
        - EXTRACTED: Documents ready for chunking/KG processing
        - PENDING: Legacy status (mapped to EXTRACTED)
        - PROCESSING: Legacy status (mapped to CHUNKING)

        This is useful for recovering from:
        - Server crashes or worker terminations
        - Network errors or timeouts
        - LLM service outages
        - Extraction failures (OCR/Docling errors)
        - Any temporary failures during processing

        The processing happens in the background and can be monitored by checking the
        pipeline status.

        Returns:
            ReprocessResponse: Response with status and message.

        Raises:
            HTTPException: If an error occurs while initiating reprocessing (500).
        """
        try:
            _, current_rag, _ = await resolve_request_context(request, workspace_id)
            await ensure_workspace_generation_writable(current_rag.workspace)

            # Count documents that will be reprocessed
            documents_count = 0
            workspaces_to_process = []
            
            doc_ids = reprocess_req.doc_ids if reprocess_req is not None else None
            
            # If doc_type filter is specified and doc_ids not explicitly provided, find failed doc_ids matching doc_type
            if not doc_ids and reprocess_req and reprocess_req.doc_type in ("legal", "general"):
                from lightrag.kg.postgres_impl import ClientManager, to_uuid
                db = await ClientManager.get_client()
                target_dt = reprocess_req.doc_type
                if workspace_id:
                    ws_uuid = to_uuid(current_rag.workspace)
                    doc_query = """
                        SELECT DISTINCT sr.doc_id
                        FROM lightrag_pipeline_stage_runs sr
                        LEFT JOIN tbl_legal_document ld ON sr.doc_id = ld.id::text AND ld.workspace_id = $2
                        WHERE sr.workspace_id = $1 AND sr.status = 'failed'
                    """
                    if target_dt == "legal":
                        doc_query += " AND (ld.id IS NOT NULL OR sr.event_json->>'doc_type' = 'legal' OR sr.event_json->'metadata'->>'doc_type' = 'legal')"
                    else:
                        doc_query += " AND (ld.id IS NULL AND (sr.event_json->>'doc_type' IS NULL OR sr.event_json->>'doc_type' != 'legal') AND (sr.event_json->'metadata'->>'doc_type' IS NULL OR sr.event_json->'metadata'->>'doc_type' != 'legal'))"
                    doc_rows = await db.query(doc_query, [current_rag.workspace, ws_uuid], multirows=True)
                else:
                    doc_query = """
                        SELECT DISTINCT sr.doc_id
                        FROM lightrag_pipeline_stage_runs sr
                        LEFT JOIN tbl_legal_document ld ON sr.doc_id = ld.id::text
                        WHERE sr.status = 'failed'
                    """
                    if target_dt == "legal":
                        doc_query += " AND (ld.id IS NOT NULL OR sr.event_json->>'doc_type' = 'legal' OR sr.event_json->'metadata'->>'doc_type' = 'legal')"
                    else:
                        doc_query += " AND (ld.id IS NULL AND (sr.event_json->>'doc_type' IS NULL OR sr.event_json->>'doc_type' != 'legal') AND (sr.event_json->'metadata'->>'doc_type' IS NULL OR sr.event_json->'metadata'->>'doc_type' != 'legal'))"
                    doc_rows = await db.query(doc_query, [], multirows=True)
                
                doc_ids = [str(r["doc_id"]) for r in (doc_rows or []) if r.get("doc_id")]
                if not doc_ids:
                    return ReprocessResponse(
                        status="success",
                        message=f"No failed {target_dt} documents found to reprocess."
                    )
            
            if doc_ids:
                documents_count = len(doc_ids)
                workspaces_to_process = [current_rag.workspace] if workspace_id else []
            else:
                try:
                    if workspace_id:
                        # Single workspace - get counts for this workspace only
                        status_counts = await current_rag.doc_status.get_all_status_counts()
                        documents_count = (
                            status_counts.get(DocStatus.FAILED.value, 0) +
                            status_counts.get("PENDING", 0) +
                            status_counts.get("PROCESSING", 0) +
                            status_counts.get(DocStatus.CHUNKING.value, 0) +
                            status_counts.get(DocStatus.EXTRACTING.value, 0)  # Include stuck EXTRACTING docs
                        )
                        if documents_count > 0:
                            workspaces_to_process = [current_rag.workspace]
                    else:
                        # All workspaces - get counts across all workspaces
                        workspaces_stats = await current_rag.doc_status.get_status_counts_across_workspaces()
                        
                        # Find workspaces with failed/pending/processing/extracting documents
                        for ws_id, ws_stats in workspaces_stats.items():
                            ws_failed_count = (
                                ws_stats.get(DocStatus.FAILED.value, 0) +
                                ws_stats.get("PENDING", 0) +
                                ws_stats.get("PROCESSING", 0) +
                                ws_stats.get(DocStatus.CHUNKING.value, 0) +
                                ws_stats.get(DocStatus.EXTRACTING.value, 0)  # Include stuck EXTRACTING docs
                            )
                            if ws_failed_count > 0:
                                workspaces_to_process.append(ws_id)
                                documents_count += ws_failed_count
                            
                except Exception as count_error:
                    logger.warning(f"Failed to count documents for reprocessing: {str(count_error)}")
                    logger.warning(traceback.format_exc())
                    # Continue with reprocessing even if count fails
                    documents_count = 0
                    workspaces_to_process = [current_rag.workspace] if workspace_id else []

            if not workspace_id:
                for workspace_to_process in workspaces_to_process:
                    await ensure_workspace_generation_writable(workspace_to_process)

            if not workspace_id:
                await acquire_workspace_write_leases(workspaces_to_process)

            # Start the reprocessing in the background for each workspace
            if workspace_id:
                # Single workspace reprocess
                background_tasks.add_task(
                    current_rag.apipeline_process_enqueue_documents, reprocess_failed=True, doc_ids=doc_ids
                )
                workspace_msg = f"in workspace '{current_rag.workspace}'"
                if doc_ids:
                    workspace_msg += f" for {len(doc_ids)} document(s)"
            else:
                from lightrag.pipeline.ingest_flags import (
                    build_ingest_stage_event,
                    enqueue_kafka_stage_events,
                    resolve_reprocess_backend,
                )
                from lightrag.pipeline.models import StageName

                if resolve_reprocess_backend() == "kafka" and doc_ids:
                    events = [
                        build_ingest_stage_event(
                            workspace_id=ws_id,
                            doc_id=str(doc_id),
                            stage=StageName.CHUNKING,
                            metadata={"reprocess_failed": True},
                        )
                        for ws_id in workspaces_to_process
                        for doc_id in doc_ids
                    ]
                    await enqueue_kafka_stage_events(events)
                    workspace_msg = f"across {len(workspaces_to_process)} workspace(s) via Kafka"
                else:
                    # Multi-workspace reprocess - use Celery task for each workspace
                    try:
                        from celery_worker.tasks import apipeline_process_enqueue_documents_task

                        for ws_id in workspaces_to_process:
                            try:
                                # Trigger Celery task for each workspace
                                apipeline_process_enqueue_documents_task.delay(
                                    workspace_id=ws_id,
                                    reprocess_failed=True,
                                    doc_ids=doc_ids
                                )
                                logger.info(f"Triggered reprocess Celery task for workspace: {ws_id}")
                            except Exception as e:
                                logger.error(f"Failed to trigger reprocess for workspace {ws_id}: {str(e)}")
                                logger.error(traceback.format_exc())

                        workspace_msg = f"across {len(workspaces_to_process)} workspace(s)"
                    except ImportError as e:
                        logger.error(f"Failed to import Celery task: {e}")
                        workspace_msg = "failed - Celery not available"
                        documents_count = 0
            
            logger.info(f"Reprocessing of {documents_count} failed documents {workspace_msg} initiated")

            return ReprocessResponse(
                status="reprocessing_started",
                message=f"Reprocessing of {documents_count} failed documents has been initiated in background.",
                documents_count=documents_count
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error initiating reprocessing of failed documents: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/stage_runs",
        response_model=PaginatedPipelineStageRunsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_pipeline_stage_runs(
        request: Request,
        req: PipelineStageRunsRequest,
    ):
        """
        Get paginated pipeline stage runs with status matrix and document names.
        """
        try:
            from lightrag.kg.postgres_impl import ClientManager
            
            db = await ClientManager.get_client()
            
            # Resolve request context for workspace ID
            resolved_workspace_id, current_rag, _ = await resolve_request_context(request, req.workspace_id)
            
            # 1. Fetch doc_type counts for workspace with in-memory caching
            cached_dtc = _get_stage_metrics_cache(f"dtc:{resolved_workspace_id}")
            dt_task = None
            if cached_dtc is None:
                doc_type_counts_sql = """
                    SELECT 
                        COUNT(*) as all_cnt,
                        COUNT(*) FILTER (WHERE sr.event_json->>'doc_type' = 'legal' OR sr.event_json->'metadata'->>'doc_type' = 'legal' OR sr.doc_id ~ '^[0-9]+$') as legal_cnt,
                        COUNT(*) FILTER (WHERE (sr.event_json->>'doc_type' IS NULL OR sr.event_json->>'doc_type' != 'legal') AND (sr.event_json->'metadata'->>'doc_type' IS NULL OR sr.event_json->'metadata'->>'doc_type' != 'legal') AND sr.doc_id !~ '^[0-9]+$') as general_cnt
                    FROM lightrag_pipeline_stage_runs sr
                    WHERE sr.workspace_id = $1
                """
                dt_task = db.query(doc_type_counts_sql, [resolved_workspace_id])

            # 2. Fetch status matrix with in-memory caching
            cached_mat = _get_stage_metrics_cache(f"mat:{resolved_workspace_id}:{req.doc_type}")
            matrix_task = None
            if cached_mat is None:
                if req.doc_type == "legal":
                    matrix_sql = """
                        SELECT sr.stage, sr.status, COUNT(DISTINCT sr.run_id) as cnt
                        FROM lightrag_pipeline_stage_runs sr
                        WHERE sr.workspace_id = $1 AND (sr.event_json->>'doc_type' = 'legal' OR sr.event_json->'metadata'->>'doc_type' = 'legal' OR sr.doc_id ~ '^[0-9]+$')
                        GROUP BY sr.stage, sr.status
                    """
                elif req.doc_type == "general":
                    matrix_sql = """
                        SELECT sr.stage, sr.status, COUNT(DISTINCT sr.run_id) as cnt
                        FROM lightrag_pipeline_stage_runs sr
                        WHERE sr.workspace_id = $1 AND (sr.event_json->>'doc_type' IS NULL OR sr.event_json->>'doc_type' != 'legal') AND (sr.event_json->'metadata'->>'doc_type' IS NULL OR sr.event_json->'metadata'->>'doc_type' != 'legal') AND sr.doc_id !~ '^[0-9]+$'
                        GROUP BY sr.stage, sr.status
                    """
                else:
                    matrix_sql = """
                        SELECT stage, status, COUNT(DISTINCT run_id) as cnt
                        FROM lightrag_pipeline_stage_runs
                        WHERE workspace_id = $1
                        GROUP BY stage, status
                    """
                matrix_task = db.query(matrix_sql, [resolved_workspace_id], multirows=True)

            # 3. Build paginated query with filters
            query_parts = ["sr.workspace_id = $1"]
            params = [resolved_workspace_id]
            
            if req.doc_type == "legal":
                query_parts.append("(sr.event_json->>'doc_type' = 'legal' OR sr.event_json->'metadata'->>'doc_type' = 'legal' OR sr.doc_id ~ '^[0-9]+$')")
            elif req.doc_type == "general":
                query_parts.append("((sr.event_json->>'doc_type' IS NULL OR sr.event_json->>'doc_type' != 'legal') AND (sr.event_json->'metadata'->>'doc_type' IS NULL OR sr.event_json->'metadata'->>'doc_type' != 'legal') AND sr.doc_id !~ '^[0-9]+$')")

            if req.stage:
                params.append(req.stage)
                query_parts.append(f"sr.stage = ${len(params)}")
                
            if req.status:
                params.append(req.status)
                query_parts.append(f"sr.status = ANY(${len(params)})")
                
            doc_search_join = ""
            if req.search:
                params.append(f"%{req.search}%")
                query_parts.append(
                    f"(COALESCE(d.document_name, ld.title) ILIKE ${len(params)} OR sr.doc_id ILIKE ${len(params)})"
                )
                doc_search_join = (
                    "LEFT JOIN tbl_document d ON d.document_id = (CASE WHEN sr.doc_id ~ '^[0-9a-fA-F-]{36}$' THEN sr.doc_id::uuid ELSE NULL END) "
                    "LEFT JOIN tbl_legal_document ld ON ld.id = (CASE WHEN sr.doc_id ~ '^[0-9]+$' THEN sr.doc_id::integer ELSE NULL END)"
                )
                
            where_clause = " AND ".join(query_parts) if query_parts else "1=1"

            # 4. Count query (skip if no extra filters, reuse doc_type_counts directly)
            has_extra_filters = bool(req.stage or req.status or req.search)
            count_task = None
            if has_extra_filters:
                count_sql = f"""
                    SELECT COUNT(*) as total
                    FROM lightrag_pipeline_stage_runs sr
                    {doc_search_join}
                    WHERE {where_clause}
                """
                count_task = db.query(count_sql, params)

            # 5. Data query with Late Materialization (filter and paginate top rows first, join document names only for those rows)
            limit = req.page_size
            offset = (req.page - 1) * req.page_size
            
            data_params = list(params)
            data_params.append(limit)
            limit_idx = len(data_params)
            data_params.append(offset)
            offset_idx = len(data_params)
            
            data_sql = f"""
                WITH paged_sr AS (
                    SELECT 
                        sr.run_id,
                        sr.stage,
                        sr.shard_index,
                        sr.shard_count,
                        sr.workspace_id,
                        sr.doc_id,
                        sr.status,
                        sr.attempt,
                        sr.error_json,
                        sr.started_at,
                        sr.completed_at,
                        sr.updated_at,
                        sr.event_json
                    FROM lightrag_pipeline_stage_runs sr
                    {doc_search_join}
                    WHERE {where_clause}
                    ORDER BY COALESCE(sr.started_at, sr.updated_at) DESC, sr.run_id DESC, sr.stage DESC, sr.shard_index DESC
                    LIMIT ${limit_idx} OFFSET ${offset_idx}
                )
                SELECT 
                    psr.run_id,
                    psr.stage,
                    psr.shard_index,
                    psr.shard_count,
                    psr.workspace_id,
                    psr.doc_id,
                    psr.status,
                    psr.attempt,
                    psr.error_json,
                    psr.started_at,
                    psr.completed_at,
                    psr.updated_at,
                    COALESCE(d.document_name, ld.title) as document_name,
                    CASE 
                        WHEN (ld.id IS NOT NULL OR psr.event_json->>'doc_type' = 'legal' OR psr.event_json->'metadata'->>'doc_type' = 'legal') THEN 'legal'
                        ELSE 'general'
                    END as doc_type
                FROM paged_sr psr
                LEFT JOIN tbl_document d ON d.document_id = (CASE WHEN psr.doc_id ~ '^[0-9a-fA-F-]{{36}}$' THEN psr.doc_id::uuid ELSE NULL END)
                LEFT JOIN tbl_legal_document ld ON ld.id = (CASE WHEN psr.doc_id ~ '^[0-9]+$' THEN psr.doc_id::integer ELSE NULL END)
                ORDER BY COALESCE(psr.started_at, psr.updated_at) DESC, psr.run_id DESC, psr.stage DESC, psr.shard_index DESC
            """
            
            # 6. Execute all independent queries concurrently via asyncio.gather
            data_task = db.query(data_sql, data_params, multirows=True)
            task_dict: dict[str, Any] = {}
            if dt_task is not None:
                task_dict["dtc"] = dt_task
            if matrix_task is not None:
                task_dict["matrix"] = matrix_task
            task_dict["data"] = data_task
            if count_task is not None:
                task_dict["count"] = count_task

            keys = list(task_dict.keys())
            import time
            t_gather_start = time.monotonic()
            logger.info(f"[STAGE_RUNS_PERF] Running tasks: {keys}")
            results = await asyncio.gather(*task_dict.values())
            logger.info(f"[STAGE_RUNS_PERF] Tasks {keys} finished in {time.monotonic() - t_gather_start:.2f}s")
            res_map = dict(zip(keys, results))

            data_rows = res_map["data"]

            if "dtc" in res_map:
                dt_row = res_map["dtc"]
                doc_type_counts = {
                    "all": int(dt_row["all_cnt"]) if dt_row and dt_row.get("all_cnt") is not None else 0,
                    "legal": int(dt_row["legal_cnt"]) if dt_row and dt_row.get("legal_cnt") is not None else 0,
                    "general": int(dt_row["general_cnt"]) if dt_row and dt_row.get("general_cnt") is not None else 0,
                }
                _set_stage_metrics_cache(f"dtc:{resolved_workspace_id}", doc_type_counts)
            else:
                doc_type_counts = cached_dtc

            if "matrix" in res_map:
                matrix_rows = res_map["matrix"]
                status_matrix = {}
                for row in matrix_rows:
                    stage = row["stage"]
                    status = row["status"]
                    count = row["cnt"]
                    if stage not in status_matrix:
                        status_matrix[stage] = {}
                    status_matrix[stage][status] = count
                _set_stage_metrics_cache(f"mat:{resolved_workspace_id}:{req.doc_type}", status_matrix)
            else:
                status_matrix = cached_mat

            if count_task is not None:
                count_row = res_map.get("count")
                total = count_row["total"] if count_row else 0
            else:
                if req.doc_type == "legal":
                    total = doc_type_counts["legal"]
                elif req.doc_type == "general":
                    total = doc_type_counts["general"]
                else:
                    total = doc_type_counts["all"]

            runs = []
            for row in data_rows:
                err_json = row["error_json"]
                if isinstance(err_json, str):
                    try:
                        import json
                        err_json = json.loads(err_json)
                    except Exception:
                        pass
                
                runs.append(
                    PipelineStageRunResponse(
                        run_id=row["run_id"],
                        stage=row["stage"],
                        shard_index=row["shard_index"],
                        shard_count=row["shard_count"],
                        workspace_id=row["workspace_id"],
                        doc_id=row["doc_id"],
                        document_name=row["document_name"],
                        doc_type=row.get("doc_type", "general"),
                        status=row["status"],
                        attempt=row["attempt"],
                        error_json=err_json,
                        started_at=row["started_at"],
                        completed_at=row["completed_at"],
                        updated_at=row["updated_at"],
                    )
                )
                
            return PaginatedPipelineStageRunsResponse(
                runs=runs,
                total=total,
                page=req.page,
                page_size=req.page_size,
                status_matrix=status_matrix,
                doc_type_counts=doc_type_counts,
            )
            
        except Exception as e:
            logger.error(f"Error getting pipeline stage runs: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/stage_runs/all",
        response_model=PaginatedPipelineStageRunsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_all_pipeline_stage_runs(
        request: Request,
        req: PipelineStageRunsRequest,
    ):
        """
        Get paginated pipeline stage runs across all workspaces with status matrix and document names.
        """
        try:
            from lightrag.kg.postgres_impl import ClientManager
            
            db = await ClientManager.get_client()
            
            # 1. Fetch doc_type counts across all workspaces with in-memory caching
            cached_dtc = _get_stage_metrics_cache("dtc:all")
            dt_task = None
            if cached_dtc is None:
                doc_type_counts_sql = """
                    SELECT 
                        COUNT(*) as all_cnt,
                        COUNT(*) FILTER (WHERE sr.event_json->>'doc_type' = 'legal' OR sr.event_json->'metadata'->>'doc_type' = 'legal' OR sr.doc_id ~ '^[0-9]+$') as legal_cnt,
                        COUNT(*) FILTER (WHERE (sr.event_json->>'doc_type' IS NULL OR sr.event_json->>'doc_type' != 'legal') AND (sr.event_json->'metadata'->>'doc_type' IS NULL OR sr.event_json->'metadata'->>'doc_type' != 'legal') AND sr.doc_id !~ '^[0-9]+$') as general_cnt
                    FROM lightrag_pipeline_stage_runs sr
                """
                dt_task = db.query(doc_type_counts_sql, [])

            # 2. Fetch status matrix across all workspaces with in-memory caching
            cached_mat = _get_stage_metrics_cache(f"mat:all:{req.doc_type}")
            matrix_task = None
            if cached_mat is None:
                if req.doc_type == "legal":
                    matrix_sql = """
                        SELECT sr.stage, sr.status, COUNT(DISTINCT sr.run_id) as cnt
                        FROM lightrag_pipeline_stage_runs sr
                        WHERE (sr.event_json->>'doc_type' = 'legal' OR sr.event_json->'metadata'->>'doc_type' = 'legal' OR sr.doc_id ~ '^[0-9]+$')
                        GROUP BY sr.stage, sr.status
                    """
                elif req.doc_type == "general":
                    matrix_sql = """
                        SELECT sr.stage, sr.status, COUNT(DISTINCT sr.run_id) as cnt
                        FROM lightrag_pipeline_stage_runs sr
                        WHERE (sr.event_json->>'doc_type' IS NULL OR sr.event_json->>'doc_type' != 'legal') AND (sr.event_json->'metadata'->>'doc_type' IS NULL OR sr.event_json->'metadata'->>'doc_type' != 'legal') AND sr.doc_id !~ '^[0-9]+$'
                        GROUP BY sr.stage, sr.status
                    """
                else:
                    matrix_sql = """
                        SELECT stage, status, COUNT(DISTINCT run_id) as cnt
                        FROM lightrag_pipeline_stage_runs
                        GROUP BY stage, status
                    """
                matrix_task = db.query(matrix_sql, [], multirows=True)

            # 3. Build paginated query with filters
            query_parts = []
            params = []
            if req.doc_type == "legal":
                query_parts.append("(sr.event_json->>'doc_type' = 'legal' OR sr.event_json->'metadata'->>'doc_type' = 'legal' OR sr.doc_id ~ '^[0-9]+$')")
            elif req.doc_type == "general":
                query_parts.append("((sr.event_json->>'doc_type' IS NULL OR sr.event_json->>'doc_type' != 'legal') AND (sr.event_json->'metadata'->>'doc_type' IS NULL OR sr.event_json->'metadata'->>'doc_type' != 'legal') AND sr.doc_id !~ '^[0-9]+$')")

            if req.stage:
                params.append(req.stage)
                query_parts.append(f"sr.stage = ${len(params)}")
                
            if req.status:
                params.append(req.status)
                query_parts.append(f"sr.status = ANY(${len(params)})")
                
            doc_search_join = ""
            if req.search:
                params.append(f"%{req.search}%")
                query_parts.append(
                    f"(COALESCE(d.document_name, ld.title) ILIKE ${len(params)} OR sr.doc_id ILIKE ${len(params)})"
                )
                doc_search_join = (
                    "LEFT JOIN tbl_document d ON d.document_id = (CASE WHEN sr.doc_id ~ '^[0-9a-fA-F-]{36}$' THEN sr.doc_id::uuid ELSE NULL END) "
                    "LEFT JOIN tbl_legal_document ld ON ld.id = (CASE WHEN sr.doc_id ~ '^[0-9]+$' THEN sr.doc_id::integer ELSE NULL END)"
                )
                
            where_clause = " AND ".join(query_parts) if query_parts else "1=1"

            # 4. Count query (skip if no extra filters, reuse doc_type_counts directly)
            has_extra_filters = bool(req.stage or req.status or req.search)
            count_task = None
            if has_extra_filters:
                count_sql = f"""
                    SELECT COUNT(*) as total
                    FROM lightrag_pipeline_stage_runs sr
                    {doc_search_join}
                    WHERE {where_clause}
                """
                count_task = db.query(count_sql, params)

            # 5. Data query with Late Materialization (filter and paginate top rows first, join document names only for those rows)
            limit = req.page_size
            offset = (req.page - 1) * req.page_size
            
            data_params = list(params)
            data_params.append(limit)
            limit_idx = len(data_params)
            data_params.append(offset)
            offset_idx = len(data_params)
            
            data_sql = f"""
                WITH paged_sr AS (
                    SELECT 
                        sr.run_id,
                        sr.stage,
                        sr.shard_index,
                        sr.shard_count,
                        sr.workspace_id,
                        sr.doc_id,
                        sr.status,
                        sr.attempt,
                        sr.error_json,
                        sr.started_at,
                        sr.completed_at,
                        sr.updated_at,
                        sr.event_json
                    FROM lightrag_pipeline_stage_runs sr
                    {doc_search_join}
                    WHERE {where_clause}
                    ORDER BY COALESCE(sr.started_at, sr.updated_at) DESC, sr.run_id DESC, sr.stage DESC, sr.shard_index DESC
                    LIMIT ${limit_idx} OFFSET ${offset_idx}
                )
                SELECT 
                    psr.run_id,
                    psr.stage,
                    psr.shard_index,
                    psr.shard_count,
                    psr.workspace_id,
                    psr.doc_id,
                    psr.status,
                    psr.attempt,
                    psr.error_json,
                    psr.started_at,
                    psr.completed_at,
                    psr.updated_at,
                    COALESCE(d.document_name, ld.title) as document_name,
                    CASE 
                        WHEN (ld.id IS NOT NULL OR psr.event_json->>'doc_type' = 'legal' OR psr.event_json->'metadata'->>'doc_type' = 'legal') THEN 'legal'
                        ELSE 'general'
                    END as doc_type
                FROM paged_sr psr
                LEFT JOIN tbl_document d ON d.document_id = (CASE WHEN psr.doc_id ~ '^[0-9a-fA-F-]{{36}}$' THEN psr.doc_id::uuid ELSE NULL END)
                LEFT JOIN tbl_legal_document ld ON ld.id = (CASE WHEN psr.doc_id ~ '^[0-9]+$' THEN psr.doc_id::integer ELSE NULL END)
                ORDER BY COALESCE(psr.started_at, psr.updated_at) DESC, psr.run_id DESC, psr.stage DESC, psr.shard_index DESC
            """
            
            # 6. Execute all independent queries concurrently via asyncio.gather
            data_task = db.query(data_sql, data_params, multirows=True)
            task_dict: dict[str, Any] = {}
            if dt_task is not None:
                task_dict["dtc"] = dt_task
            if matrix_task is not None:
                task_dict["matrix"] = matrix_task
            task_dict["data"] = data_task
            if count_task is not None:
                task_dict["count"] = count_task

            keys = list(task_dict.keys())
            import time
            t_gather_start = time.monotonic()
            logger.info(f"[STAGE_RUNS_PERF] Running tasks: {keys}")
            results = await asyncio.gather(*task_dict.values())
            logger.info(f"[STAGE_RUNS_PERF] Tasks {keys} finished in {time.monotonic() - t_gather_start:.2f}s")
            res_map = dict(zip(keys, results))

            data_rows = res_map["data"]

            if "dtc" in res_map:
                dt_row = res_map["dtc"]
                doc_type_counts = {
                    "all": int(dt_row["all_cnt"]) if dt_row and dt_row.get("all_cnt") is not None else 0,
                    "legal": int(dt_row["legal_cnt"]) if dt_row and dt_row.get("legal_cnt") is not None else 0,
                    "general": int(dt_row["general_cnt"]) if dt_row and dt_row.get("general_cnt") is not None else 0,
                }
                _set_stage_metrics_cache("dtc:all", doc_type_counts)
            else:
                doc_type_counts = cached_dtc

            if "matrix" in res_map:
                matrix_rows = res_map["matrix"]
                status_matrix = {}
                for row in matrix_rows:
                    stage = row["stage"]
                    status = row["status"]
                    count = row["cnt"]
                    if stage not in status_matrix:
                        status_matrix[stage] = {}
                    status_matrix[stage][status] = count
                _set_stage_metrics_cache(f"mat:all:{req.doc_type}", status_matrix)
            else:
                status_matrix = cached_mat

            if count_task is not None:
                count_row = res_map.get("count")
                total = count_row["total"] if count_row else 0
            else:
                if req.doc_type == "legal":
                    total = doc_type_counts["legal"]
                elif req.doc_type == "general":
                    total = doc_type_counts["general"]
                else:
                    total = doc_type_counts["all"]



            runs = []
            for row in data_rows:
                err_json = row["error_json"]
                if isinstance(err_json, str):
                    try:
                        import json
                        err_json = json.loads(err_json)
                    except Exception:
                        pass
                
                runs.append(
                    PipelineStageRunResponse(
                        run_id=row["run_id"],
                        stage=row["stage"],
                        shard_index=row["shard_index"],
                        shard_count=row["shard_count"],
                        workspace_id=row["workspace_id"],
                        doc_id=row["doc_id"],
                        document_name=row["document_name"],
                        doc_type=row.get("doc_type", "general"),
                        status=row["status"],
                        attempt=row["attempt"],
                        error_json=err_json,
                        started_at=row["started_at"],
                        completed_at=row["completed_at"],
                        updated_at=row["updated_at"],
                    )
                )
                
            return PaginatedPipelineStageRunsResponse(
                runs=runs,
                total=total,
                page=req.page,
                page_size=req.page_size,
                status_matrix=status_matrix,
                doc_type_counts=doc_type_counts,
            )
            
        except Exception as e:
            logger.error(f"Error getting all pipeline stage runs: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    async def publish_events_to_kafka(events):
        from aiokafka import AIOKafkaProducer
        from kafka_worker.config import KAFKA_BOOTSTRAP_SERVERS
        from lightrag.pipeline.partitioning import create_partition_allocator_from_env
        from kafka_worker.pipeline_main import (
            STAGE_TOPICS,
            _canonical_doc_type,
            _normalize_event_doc_type,
        )

        if not events:
            return 0, 0

        producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
        partition_allocator = create_partition_allocator_from_env()
        partition_counts: dict[str, int] = {}
        published_count = 0
        failed_count = 0

        await producer.start()
        try:
            for event in events:
                try:
                    normalized_event = _normalize_event_doc_type(event)
                    doc_type = _canonical_doc_type(normalized_event)
                    topic = STAGE_TOPICS[normalized_event.stage_name][doc_type]
                    partition_count = partition_counts.get(topic)
                    if partition_count is None:
                        partitions = await producer.partitions_for(topic)
                        if not partitions:
                            raise RuntimeError(
                                f"Kafka topic has no available partitions: {topic}"
                            )
                        partition_count = len(partitions)
                        partition_counts[topic] = partition_count

                    partition = await partition_allocator.allocate(
                        topic, partition_count
                    )
                    await producer.send_and_wait(
                        topic,
                        normalized_event.model_dump_json(by_alias=True).encode(
                            "utf-8"
                        ),
                        key=normalized_event.partition_key().encode("utf-8"),
                        partition=partition,
                    )
                    published_count += 1
                except Exception as event_error:
                    failed_count += 1
                    logger.error(
                        "[ApproveStage] Failed to publish stage event run_id=%s stage=%s workspace_id=%s: %s",
                        getattr(event, "run_id", "unknown"),
                        getattr(event, "stage", "unknown"),
                        getattr(event, "workspace_id", "unknown"),
                        event_error,
                    )
        finally:
            await producer.stop()
            await partition_allocator.finalize()

        return published_count, failed_count

    async def publish_event_to_kafka(event):
        published_count, failed_count = await publish_events_to_kafka([event])
        if published_count != 1 or failed_count:
            raise RuntimeError("Failed to publish approved stage event")

    def build_stage_event_from_row(row, PipelineStageEvent, json):
        raw_event_json = row["event_json"]
        if isinstance(raw_event_json, str):
            event_dict = json.loads(raw_event_json)
        else:
            event_dict = raw_event_json

        event = PipelineStageEvent.model_validate(event_dict)
        normalized_event = event.model_copy(deep=True)
        normalized_event.run_id = str(row["run_id"])
        normalized_event.stage = row["stage"]
        normalized_event.shard_index = int(row["shard_index"])
        normalized_event.shard_count = int(row["shard_count"])
        normalized_event.workspace_id = str(row["workspace_id"])
        normalized_event.doc_id = str(row["doc_id"])
        return normalized_event

    @router.post(
        "/approve_stage",
        response_model=ApproveStageResponse,
        dependencies=[Depends(combined_auth)],
    )
    @with_workspace_write_lease
    async def approve_stage(
        request: Request,
        req: ApproveStageRequest,
    ):
        """
        Approve/resume a paused stage run in the pipeline.
        """
        try:
            from lightrag.kg.postgres_impl import ClientManager
            from lightrag.pipeline.models import PipelineStageEvent
            import json

            resolved_workspace_id, _, _ = await resolve_request_context(
                request, req.workspace_id
            )
            await ensure_workspace_generation_writable(resolved_workspace_id)

            db = await ClientManager.get_client()

            row = await db.query(
                """
                SELECT run_id, stage, shard_index, shard_count, workspace_id,
                       doc_id, status, event_json
                FROM lightrag_pipeline_stage_runs
                WHERE run_id = $1 AND stage = $2 AND shard_index = $3 AND workspace_id = $4
                """,
                [req.run_id, req.stage, req.shard_index, resolved_workspace_id],
            )

            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No stage run found for run_id={req.run_id}, "
                        f"stage={req.stage}, shard={req.shard_index}"
                    ),
                )

            status = row["status"]
            if status != "paused":
                raise HTTPException(
                    status_code=400,
                    detail=f"Stage run is not paused (current status: {status})",
                )

            event = build_stage_event_from_row(row, PipelineStageEvent, json)
            await publish_event_to_kafka(event)

            return ApproveStageResponse(
                status="success",
                message=(
                    f"Stage {req.stage} for run {req.run_id} approved and "
                    "resumed successfully."
                ),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error approving stage run: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/replay_stage",
        response_model=ReplayStageResponse,
        dependencies=[Depends(combined_auth)],
    )
    @with_workspace_write_lease
    async def replay_stage(
        request: Request,
        req: ReplayStageRequest,
    ):
        """Queue an audited retry for a failed pipeline stage shard."""
        from lightrag.pipeline.coordinator import (
            StageNotReplayableError,
            StageRetryBudgetExhausted,
            StageRunNotFoundError,
            create_coordinator_from_env,
        )
        from lightrag.pipeline.models import StageName

        try:
            stage = StageName(req.stage)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown pipeline stage: {req.stage}",
            ) from exc

        resolved_workspace_id, _, _ = await resolve_request_context(
            request, req.workspace_id
        )
        await ensure_workspace_generation_writable(resolved_workspace_id)
        authorization = request.headers.get("authorization", "")
        actor = "api_key" if request.headers.get("X-API-Key") else "anonymous"
        if authorization.lower().startswith("bearer "):
            from lightrag.api.auth import auth_handler

            token_info = auth_handler.validate_token(authorization[7:].strip())
            actor = str(token_info.get("username") or "authenticated_user")

        coordinator = create_coordinator_from_env()
        try:
            replay = await coordinator.replay_failed_stage(
                run_id=req.run_id,
                stage=stage,
                shard_index=req.shard_index,
                workspace_id=resolved_workspace_id,
                reason=req.reason,
                requested_by=actor,
            )
        except StageRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (StageNotReplayableError, StageRetryBudgetExhausted) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception(
                "Failed to queue pipeline stage replay run_id=%s stage=%s shard=%s workspace_id=%s actor=%s",
                req.run_id,
                stage.value,
                req.shard_index,
                resolved_workspace_id,
                actor,
            )
            raise HTTPException(status_code=500, detail="Failed to queue stage replay") from exc
        finally:
            try:
                await coordinator.finalize()
            except Exception:
                logger.exception("Failed to release pipeline coordinator after stage replay")

        logger.warning(
            "[PipelineAudit] Stage replay queued actor=%s reason=%r event_id=%s "
            "run_id=%s stage=%s shard=%s workspace_id=%s",
            actor,
            req.reason,
            replay.event.event_id,
            req.run_id,
            stage.value,
            req.shard_index,
            resolved_workspace_id,
        )
        return ReplayStageResponse(
            status=replay.status,
            message="Failed stage replay queued for delivery.",
            event_id=replay.event.event_id,
            attempt=replay.event.attempt,
        )

    @router.post(
        "/generations/start",
        response_model=GenerationStatusResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def start_generation_rebuild(
        request: Request,
        req: GenerationStartRequest,
    ):
        """Start or resume a shadow workspace generation rebuild."""
        from lightrag.pipeline.coordinator import (
            WorkspaceGenerationConflictError,
            WorkspaceGenerationTransitionError,
            create_coordinator_from_env,
        )
        from lightrag.pipeline.generation_orchestrator import (
            start_workspace_generation_rebuild,
        )

        resolved_workspace_id, _, _ = await resolve_request_context(
            request, req.workspace_id
        )
        coordinator = create_coordinator_from_env()
        try:
            gen, manifest = await start_workspace_generation_rebuild(
                coordinator,
                resolved_workspace_id,
                generation_id=req.generation_id,
            )
            logger.warning(
                "[PipelineAudit] Generation rebuild started workspace_id=%s generation_id=%s reason=%r",
                resolved_workspace_id,
                gen.generation_id,
                req.reason,
            )
            return GenerationStatusResponse(
                workspace_id=gen.workspace_id,
                generation_id=gen.generation_id,
                storage_workspace_id=gen.storage_workspace_id,
                status=gen.status.value,
                snapshot_complete=manifest.snapshot_complete,
                expected_documents=manifest.expected_documents,
                pending_documents=manifest.pending_documents,
                running_documents=manifest.running_documents,
                completed_documents=manifest.completed_documents,
                failed_documents=manifest.failed_documents,
            )
        except (WorkspaceGenerationConflictError, WorkspaceGenerationTransitionError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            await coordinator.finalize()

    @router.get(
        "/generations/{generation_id}",
        response_model=GenerationStatusResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_generation_status(
        request: Request,
        generation_id: str,
        workspace_id: Optional[str] = None,
    ):
        """Inspect status and manifest progress of a workspace generation."""
        from lightrag.pipeline.coordinator import (
            WorkspaceGenerationNotFoundError,
            create_coordinator_from_env,
        )

        resolved_workspace_id, _, _ = await resolve_request_context(
            request, workspace_id
        )
        coordinator = create_coordinator_from_env()
        try:
            gen = await coordinator.get_workspace_generation(
                resolved_workspace_id, generation_id
            )
            if gen is None:
                raise HTTPException(status_code=404, detail="Generation not found")
            manifest = await coordinator.get_workspace_generation_manifest_status(
                resolved_workspace_id, generation_id
            )
            return GenerationStatusResponse(
                workspace_id=gen.workspace_id,
                generation_id=gen.generation_id,
                storage_workspace_id=gen.storage_workspace_id,
                status=gen.status.value,
                snapshot_complete=manifest.snapshot_complete,
                expected_documents=manifest.expected_documents,
                pending_documents=manifest.pending_documents,
                running_documents=manifest.running_documents,
                completed_documents=manifest.completed_documents,
                failed_documents=manifest.failed_documents,
            )
        except WorkspaceGenerationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        finally:
            await coordinator.finalize()

    @router.post(
        "/generations/{generation_id}/cutover",
        response_model=GenerationStatusResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def cutover_generation(
        request: Request,
        generation_id: str,
        req: GenerationActionRequest,
    ):
        """Atomically promote a completed shadow generation to active."""
        from lightrag.pipeline.coordinator import (
            WorkspaceGenerationConflictError,
            WorkspaceGenerationNotFoundError,
            WorkspaceGenerationTransitionError,
            create_coordinator_from_env,
        )
        from lightrag.pipeline.generation_orchestrator import (
            cutover_workspace_generation,
        )

        resolved_workspace_id, _, _ = await resolve_request_context(
            request, req.workspace_id
        )
        coordinator = create_coordinator_from_env()
        try:
            gen = await cutover_workspace_generation(
                coordinator, resolved_workspace_id, generation_id
            )
            logger.warning(
                "[PipelineAudit] Generation cutover completed workspace_id=%s generation_id=%s reason=%r",
                resolved_workspace_id,
                generation_id,
                req.reason,
            )
            return GenerationStatusResponse(
                workspace_id=gen.workspace_id,
                generation_id=gen.generation_id,
                storage_workspace_id=gen.storage_workspace_id,
                status=gen.status.value,
            )
        except WorkspaceGenerationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (WorkspaceGenerationConflictError, WorkspaceGenerationTransitionError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            await coordinator.finalize()

    @router.post(
        "/generations/{generation_id}/rollback",
        response_model=GenerationStatusResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def rollback_generation(
        request: Request,
        generation_id: str,
        req: GenerationActionRequest,
    ):
        """Rollback active pointer to a retained generation within the 7-day window."""
        from lightrag.pipeline.coordinator import (
            WorkspaceGenerationConflictError,
            WorkspaceGenerationNotFoundError,
            WorkspaceGenerationTransitionError,
            create_coordinator_from_env,
        )
        from lightrag.pipeline.generation_orchestrator import (
            rollback_workspace_generation,
        )

        resolved_workspace_id, _, _ = await resolve_request_context(
            request, req.workspace_id
        )
        coordinator = create_coordinator_from_env()
        try:
            gen = await rollback_workspace_generation(
                coordinator, resolved_workspace_id, generation_id
            )
            logger.warning(
                "[PipelineAudit] Generation rollback completed workspace_id=%s generation_id=%s reason=%r",
                resolved_workspace_id,
                generation_id,
                req.reason,
            )
            return GenerationStatusResponse(
                workspace_id=gen.workspace_id,
                generation_id=gen.generation_id,
                storage_workspace_id=gen.storage_workspace_id,
                status=gen.status.value,
            )
        except WorkspaceGenerationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (WorkspaceGenerationConflictError, WorkspaceGenerationTransitionError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            await coordinator.finalize()

    @router.post(
        "/bulk_approve_stage",
        response_model=ApproveStageResponse,
        dependencies=[Depends(combined_auth)],
    )
    @with_workspace_write_lease
    async def bulk_approve_stage(
        request: Request,
        req: BulkApproveStageRequest,
        background_tasks: BackgroundTasks,
    ):
        """
        Approve and resume ALL paused stage runs for a specific stage.
        """
        try:
            from lightrag.kg.postgres_impl import ClientManager
            from lightrag.pipeline.models import PipelineStageEvent
            import json

            db = await ClientManager.get_client()

            query_all_workspaces = req.workspace_id == "all"
            query_all_stages = req.stage == "all"

            query_parts = ["status = 'paused'"]
            params = []

            if not query_all_stages:
                params.append(req.stage)
                query_parts.append(f"stage = ${len(params)}")

            if not query_all_workspaces:
                resolved_workspace_id, _, _ = await resolve_request_context(
                    request, req.workspace_id
                )
                params.append(resolved_workspace_id)
                query_parts.append(f"workspace_id = ${len(params)}")

            where_clause = " AND ".join(query_parts)

            rows = await db.query(
                f"""
                SELECT run_id, stage, shard_index, shard_count, workspace_id,
                       doc_id, event_json
                FROM lightrag_pipeline_stage_runs
                WHERE {where_clause}
                ORDER BY run_id
                """,
                params,
                multirows=True,
            )

            if not rows:
                stage_desc = "all stages" if query_all_stages else f"stage {req.stage}"
                workspace_desc = (
                    "all workspaces"
                    if query_all_workspaces
                    else f"workspace {req.workspace_id}"
                )
                return ApproveStageResponse(
                    status="success",
                    message=(
                        f"No paused runs found for {stage_desc} in "
                        f"{workspace_desc}"
                    ),
                )

            target_workspaces = sorted(
                {str(row["workspace_id"]) for row in rows}
            )
            await acquire_workspace_write_leases(target_workspaces)

            for workspace_id in target_workspaces:
                await ensure_workspace_generation_writable(workspace_id)

            events = []
            parse_failed_count = 0
            for row in rows:
                try:
                    events.append(
                        build_stage_event_from_row(row, PipelineStageEvent, json)
                    )
                except Exception as e:
                    parse_failed_count += 1
                    logger.error(
                        f"[BulkApprove] Failed to approve run_id={row['run_id']}: {e}"
                    )

            stage_desc = "all stages" if query_all_stages else f"stage {req.stage}"
            background_threshold = max(
                1, int(os.getenv("BULK_APPROVE_BACKGROUND_THRESHOLD", "100"))
            )
            if query_all_workspaces or len(events) > background_threshold:
                background_tasks.add_task(publish_events_to_kafka, events)
                failure_suffix = (
                    f", {parse_failed_count} failed to parse"
                    if parse_failed_count
                    else ""
                )
                return ApproveStageResponse(
                    status="success",
                    message=(
                        f"Queued {len(events)}/{len(rows)} paused runs for "
                        f"{stage_desc} for background approval{failure_suffix}"
                    ),
                )

            approved_count, publish_failed_count = await publish_events_to_kafka(events)
            failure_count = parse_failed_count + publish_failed_count
            failure_suffix = f", {failure_count} failed" if failure_count else ""
            return ApproveStageResponse(
                status="success",
                message=(
                    f"Queued {approved_count}/{len(rows)} paused runs for "
                    f"{stage_desc}{failure_suffix}"
                ),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error bulk-approving stage: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/stage_speed",
        response_model=list[StageSpeedEntry],
        dependencies=[Depends(combined_auth)],
    )
    async def get_stage_speed(
        request: Request,
        workspace_id: Optional[str] = None,
    ):
        """
        Returns processing speed stats per stage: docs completed in 1m/5m, avg duration.
        """
        try:
            from lightrag.kg.postgres_impl import ClientManager

            resolved_workspace_id, _, _ = await resolve_request_context(request, workspace_id)

            db = await ClientManager.get_client()

            rows = await db.query(
                """
                SELECT
                    stage,
                    COUNT(*) FILTER (
                        WHERE status = 'done' AND completed_at >= NOW() - INTERVAL '1 minute'
                    ) AS docs_completed_1m,
                    COUNT(*) FILTER (
                        WHERE status = 'done' AND completed_at >= NOW() - INTERVAL '5 minutes'
                    ) AS docs_completed_5m,
                    AVG(
                        EXTRACT(EPOCH FROM (completed_at - started_at))
                    ) FILTER (
                        WHERE status = 'done'
                        AND completed_at IS NOT NULL
                        AND started_at IS NOT NULL
                        AND completed_at >= NOW() - INTERVAL '10 minutes'
                    ) AS avg_duration_seconds
                FROM lightrag_pipeline_stage_runs
                WHERE workspace_id = $1
                GROUP BY stage
                ORDER BY stage
                """,
                [resolved_workspace_id],
                multirows=True,
            )

            result = []
            for row in rows:
                avg_dur = row.get("avg_duration_seconds")
                result.append(
                    StageSpeedEntry(
                        stage=row["stage"],
                        docs_completed_1m=int(row.get("docs_completed_1m", 0) or 0),
                        docs_completed_5m=int(row.get("docs_completed_5m", 0) or 0),
                        avg_duration_seconds=round(float(avg_dur), 2) if avg_dur is not None else None,
                    )
                )
            return result

        except Exception as e:
            logger.error(f"Error getting stage speed: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/pipeline_config",
        response_model=PipelineConfigResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_pipeline_config(request: Request):
        """
        Returns the current pipeline configuration from environment variables.
        """
        return PipelineConfigResponse(
            entity_extraction_model=os.getenv("ENTITY_EXTRACTION_LLM_MODEL", ""),
            entity_extraction_host=os.getenv("ENTITY_EXTRACTION_LLM_BINDING_HOST", ""),
            entity_extraction_api_key=os.getenv("ENTITY_EXTRACTION_LLM_BINDING_API_KEY", ""),
            paused_stages=os.getenv("PIPELINE_PAUSED_STAGES", ""),
        )

    @router.post(
        "/pipeline_config",
        response_model=ApproveStageResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def update_pipeline_config(
        request: Request,
        req: PipelineConfigUpdateRequest,
    ):
        """
        Update .env pipeline config and optionally restart workers.
        """
        try:
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), ".env")
            if not os.path.exists(env_path):
                raise HTTPException(status_code=404, detail=f".env file not found at {env_path}")

            # Read existing .env
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Build update map
            updates: dict[str, str] = {}
            if req.entity_extraction_model is not None:
                updates["ENTITY_EXTRACTION_LLM_MODEL"] = req.entity_extraction_model
            if req.entity_extraction_host is not None:
                updates["ENTITY_EXTRACTION_LLM_BINDING_HOST"] = req.entity_extraction_host
            if req.entity_extraction_api_key is not None:
                updates["ENTITY_EXTRACTION_LLM_BINDING_API_KEY"] = req.entity_extraction_api_key
            if req.paused_stages is not None:
                updates["PIPELINE_PAUSED_STAGES"] = req.paused_stages

            if not updates:
                return ApproveStageResponse(status="success", message="No changes to apply")

            # Replace matching lines
            updated_keys = set()
            new_lines = []
            for line in lines:
                stripped = line.strip()
                matched = False
                for key, value in updates.items():
                    if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                        new_lines.append(f"{key}={value}\n")
                        updated_keys.add(key)
                        matched = True
                        break
                if not matched:
                    new_lines.append(line)

            # Append any keys not found
            for key, value in updates.items():
                if key not in updated_keys:
                    new_lines.append(f"{key}={value}\n")

            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            # Also update current process env
            for key, value in updates.items():
                os.environ[key] = value

            msg_parts = [f"Updated: {', '.join(updates.keys())}"]

            # Auto-restart workers
            if req.restart_workers:
                import subprocess
                compose_file = os.path.join(os.path.dirname(env_path), "docker-compose.dev.yml")
                if os.path.exists(compose_file):
                    try:
                        subprocess.Popen(
                            [
                                "docker", "compose", "-f", compose_file,
                                "--profile", "workers", "up", "-d", "--force-recreate",
                            ],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=os.path.dirname(env_path),
                        )
                        msg_parts.append("Worker restart initiated")
                    except Exception as e:
                        msg_parts.append(f"Worker restart failed: {e}")
                else:
                    msg_parts.append("docker-compose.dev.yml not found, skipping restart")

            return ApproveStageResponse(status="success", message=". ".join(msg_parts))

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating pipeline config: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/{doc_id}",
        response_model=DocStatusResponse,
        dependencies=[Depends(combined_auth)],
        summary="Get document status by document ID",
    )
    async def get_document_by_id(
        doc_id: str,
        request: Request,
        workspace_id: Optional[str] = None,
    ) -> DocStatusResponse:
        """
        Get the processing status of a single document by its document ID.
        
        This endpoint retrieves detailed information about a specific document including:
        - Processing status (UPLOADING, EXTRACTING, EXTRACTED, CHUNKING, PROCESSED, FAILED)
        - Content summary and length
        - Chunk count
        - Error messages (if any)
        - Metadata
        - Timestamps (created_at, updated_at)
        
        Args:
            doc_id (str): The document ID (format: doc-{uuid})
            workspace_id (Optional[str]): Workspace identifier (optional, can be inferred from context)

        Returns:
            DocStatusResponse: Document status information

        Raises:
            HTTPException: 
                - 400 if doc_id is invalid
                - 404 if document not found
                - 500 if an error occurs while retrieving document status
        """
        try:
            # Validate doc_id format
            if not doc_id or not doc_id.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Document ID cannot be empty"
                )
            
            doc_id = doc_id.strip()
            
            # Resolve workspace context
            resolved_workspace_id, current_rag, _ = await resolve_request_context(
                request, workspace_id
            )
            
            # Get document by ID
            doc_data = await current_rag.doc_status.get_by_id(doc_id)
            
            if not doc_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document with ID '{doc_id}' not found in workspace '{resolved_workspace_id}'"
                )
            
            # Convert to DocStatusResponse
            # Handle both dict and DocProcessingStatus object
            if isinstance(doc_data, dict):
                return DocStatusResponse(
                    id=doc_id,
                    content_summary=doc_data.get("content_summary", ""),
                    content_length=doc_data.get("content_length", 0),
                    status=doc_data.get("status", DocStatus.FAILED),
                    created_at=format_datetime(doc_data.get("created_at")),
                    updated_at=format_datetime(doc_data.get("updated_at")),
                    workspace_id=resolved_workspace_id,
                    chunks_count=doc_data.get("chunks_count"),
                    error_msg=doc_data.get("error_msg"),
                    metadata=doc_data.get("metadata"),
                    file_path=normalize_file_path(doc_data.get("file_path", "")),
                )
            else:
                # DocProcessingStatus object
                return DocStatusResponse(
                    id=doc_id,
                    content_summary=getattr(doc_data, "content_summary", ""),
                    content_length=getattr(doc_data, "content_length", 0),
                    status=getattr(doc_data, "status", DocStatus.FAILED),
                    created_at=format_datetime(getattr(doc_data, "created_at", None)),
                    updated_at=format_datetime(getattr(doc_data, "updated_at", None)),
                    workspace_id=resolved_workspace_id,
                    chunks_count=getattr(doc_data, "chunks_count", None),
                    error_msg=getattr(doc_data, "error_msg", None),
                    metadata=getattr(doc_data, "metadata", None),
                    file_path=normalize_file_path(getattr(doc_data, "file_path", "")),
                )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting document by ID '{doc_id}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.patch(
        "/{doc_id}/legal_metadata",
        response_model=DocStatusResponse,
        dependencies=[Depends(combined_auth)],
        summary="Update legal metadata for a document",
    )
    @with_workspace_write_lease
    async def update_legal_metadata(
        doc_id: str,
        legal_metadata: Dict[str, Optional[str]],
        request: Request,
        workspace_id: Optional[str] = None,
    ) -> DocStatusResponse:
        """
        Update legal metadata for a document without re-processing.
        
        This endpoint allows updating legal-specific metadata fields (legal_status,
        effective_date, issuing_authority, document_type, document_number) for a
        document that has doc_type="legal". The updates are propagated to all chunks
        associated with the document without triggering re-chunking or re-extraction.
        
        This is useful for:
        - Updating legal status when a law becomes effective or expires
        - Correcting metadata errors without re-processing the entire document
        - Keeping legal information current without re-ingestion
        
        Implements Requirements 20.1, 20.2, 20.3, 20.4, 20.7, 20.8:
        - Validates that document has doc_type="legal"
        - Updates metadata in document storage without re-processing
        - Propagates changes to all existing chunks
        - Returns updated document status
        
        Args:
            doc_id (str): The document ID (format: doc-{uuid})
            legal_metadata (Dict[str, Optional[str]]): Legal metadata fields to update:
                - legal_status: Legal status (e.g., "Còn hiệu lực", "Hết hiệu lực")
                - effective_date: Effective date (e.g., "2008-06-03")
                - issuing_authority: Issuing authority (e.g., "Quốc hội", "Chính phủ")
                - document_type: Document type (e.g., "Luật", "Nghị định", "Thông tư")
                - document_number: Document number (e.g., "13/2008/QH12")
            workspace_id (Optional[str]): Workspace identifier (optional)

        Returns:
            DocStatusResponse: Updated document status information

        Raises:
            HTTPException: 
                - 400 if doc_id is invalid or document is not legal type
                - 404 if document not found
                - 500 if an error occurs during update
        
        Example:
            PATCH /documents/doc-550e8400-e29b-41d4-a716-446655440000/legal_metadata
            {
                "legal_status": "Hết hiệu lực",
                "effective_date": "2024-01-01"
            }
        """
        try:
            # Validate doc_id format
            if not doc_id or not doc_id.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Document ID cannot be empty"
                )
            
            doc_id = doc_id.strip()
            
            # Resolve workspace context
            resolved_workspace_id, current_rag, _ = await resolve_request_context(
                request, workspace_id
            )
            await ensure_workspace_generation_writable(current_rag.workspace)
            
            # Get document by ID
            doc_data = await current_rag.doc_status.get_by_id(doc_id)
            
            if not doc_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {doc_id} not found in workspace {resolved_workspace_id}"
                )
            
            # Validate that document has doc_type="legal" (Requirement 20.7)
            metadata = doc_data.get("metadata", {})
            doc_type = metadata.get("doc_type", "general")
            
            if doc_type != "legal":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot update legal metadata for document with doc_type='{doc_type}'. "
                           f"Only documents with doc_type='legal' can have legal metadata updated."
                )
            
            # Filter out None values and validate field names
            valid_fields = {
                "legal_status",
                "effective_date",
                "issuing_authority",
                "document_type",
                "document_number"
            }
            
            filtered_metadata = {
                k: v for k, v in legal_metadata.items()
                if v is not None and k in valid_fields
            }
            
            if not filtered_metadata:
                raise HTTPException(
                    status_code=400,
                    detail="No valid legal metadata fields provided for update. "
                           f"Valid fields: {', '.join(sorted(valid_fields))}"
                )
            
            logger.info(
                f"Updating legal metadata for document {doc_id} in workspace {resolved_workspace_id}: "
                f"{filtered_metadata}"
            )
            
            # Update metadata in document storage (Requirement 20.4)
            # Get or create legal_info dict within metadata
            legal_info = metadata.get("legal_info", {})
            if not isinstance(legal_info, dict):
                legal_info = {}
            
            # Update legal metadata fields
            for field, value in filtered_metadata.items():
                legal_info[field] = value
            
            # Update metadata with new legal_info
            metadata["legal_info"] = legal_info
            
            # Update document in storage
            updated_doc_data = {
                doc_id: {
                    **doc_data,
                    "metadata": metadata,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            }
            
            await current_rag.doc_status.upsert(updated_doc_data)
            
            # Propagate legal metadata to chunks (Requirement 20.5, 20.6)
            from lightrag.citation import propagate_legal_metadata_to_chunks
            
            try:
                updated_chunks_count = await propagate_legal_metadata_to_chunks(
                    doc_id=doc_id,
                    legal_metadata_update=filtered_metadata,
                    doc_status_storage=current_rag.doc_status,
                    text_chunks_storage=current_rag.text_chunks,
                )
                
                logger.info(
                    f"Successfully propagated legal metadata to {updated_chunks_count} chunks "
                    f"for document {doc_id}"
                )
            except Exception as propagate_error:
                # Log error but don't fail the request - document metadata was updated
                logger.error(
                    f"Failed to propagate legal metadata to chunks for document {doc_id}: "
                    f"{str(propagate_error)}"
                )
                logger.error(traceback.format_exc())
            
            # Retrieve updated document to return
            updated_doc = await current_rag.doc_status.get_by_id(doc_id)
            
            if not updated_doc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Document {doc_id} not found after update"
                )
            
            # Convert to DocProcessingStatus if needed
            if isinstance(updated_doc, dict):
                from lightrag.base import DocProcessingStatus
                updated_doc = DocProcessingStatus(**updated_doc)
            
            # Return updated document status
            return DocStatusResponse(
                id=doc_id,
                content_summary=getattr(updated_doc, "content_summary", ""),
                content_length=getattr(updated_doc, "content_length", 0),
                status=getattr(updated_doc, "status", DocStatus.FAILED),
                created_at=format_datetime(getattr(updated_doc, "created_at", None)),
                updated_at=format_datetime(getattr(updated_doc, "updated_at", None)),
                workspace_id=resolved_workspace_id,
                chunks_count=getattr(updated_doc, "chunks_count", None),
                error_msg=getattr(updated_doc, "error_msg", None),
                metadata=getattr(updated_doc, "metadata", None),
                file_path=normalize_file_path(getattr(updated_doc, "file_path", "")),
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating legal metadata for document '{doc_id}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/cancel_pipeline",
        response_model=CancelPipelineResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def cancel_pipeline(
        request: Request, workspace_id: Optional[str] = None
    ):
        """
        Request cancellation of the currently running pipeline.

        This endpoint sets a cancellation flag in the pipeline status. The pipeline will:
        1. Check this flag at key processing points
        2. Stop processing new documents
        3. Cancel all running document processing tasks
        4. Mark all PROCESSING documents as FAILED with reason "User cancelled"

        The cancellation is graceful and ensures data consistency. Documents that have
        completed processing will remain in PROCESSED status.

        Returns:
            CancelPipelineResponse: Response with status and message
                - status="cancellation_requested": Cancellation flag has been set
                - status="not_busy": Pipeline is not currently running

        Raises:
            HTTPException: If an error occurs while setting cancellation flag (500).
        """
        try:
            _, current_rag, _ = await resolve_request_context(request, workspace_id)

            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_namespace_lock,
            )

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=current_rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=current_rag.workspace
            )

            async with pipeline_status_lock:
                if not pipeline_status.get("busy", False):
                    return CancelPipelineResponse(
                        status="not_busy",
                        message="Pipeline is not currently running. No cancellation needed.",
                    )

                # Set cancellation flag
                pipeline_status["cancellation_requested"] = True
                cancel_msg = "Pipeline cancellation requested by user"
                logger.info(cancel_msg)
                pipeline_status["latest_message"] = cancel_msg
                pipeline_status["history_messages"].append(cancel_msg)

            return CancelPipelineResponse(
                status="cancellation_requested",
                message="Pipeline cancellation has been requested. Documents will be marked as FAILED.",
            )

        except Exception as e:
            logger.error(f"Error requesting pipeline cancellation: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    return router
