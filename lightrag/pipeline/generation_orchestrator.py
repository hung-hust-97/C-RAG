"""Workspace generation rebuild orchestrator, cutover/rollback, and retention cleanup.

This module bridges source-document discovery (read-only from KMS tables) with
the durable ``PipelineCoordinator`` generation manifest, transactional outbox,
atomic active-pointer cutover, 7-day rollback, and expired-generation cleanup.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Sequence

from ..kg.postgres_impl import ClientManager, to_uuid
from ..kg.postgres_shadow_impl import (
    SHADOW_DOC_STATUS_TABLE,
    SHADOW_TEXT_CHUNKS_TABLE,
    ensure_shadow_tables,
)
from ..utils import logger
from .coordinator import (
    PipelineCoordinator,
    WorkspaceGeneration,
    WorkspaceGenerationManifestStatus,
    WorkspaceGenerationStatus,
    WorkspaceGenerationTransitionError,
)
from .models import PipelineStageEvent, StageName

_GENERATION_EVENT_NS = uuid.UUID("7b3c9b12-6e11-4d89-9a44-5b91a0e3d114")

_GENERAL_SNAPSHOT_SQL = """
    SELECT document_id::text AS doc_id,
           COALESCE(document_name, '') AS document_name,
           COALESCE(origin_path, '') AS origin_path
    FROM tbl_document
    WHERE workspace_id = $1
      AND status IN ('EXTRACTED', 'PROCESSED')
    ORDER BY document_id ASC
"""

_LEGAL_SNAPSHOT_SQL = """
    SELECT id AS legal_document_id,
           COALESCE(title, '') AS title
    FROM tbl_legal_document
    WHERE workspace_id = $1
      AND COALESCE(process_status, 'EXTRACTED') IN ('EXTRACTED', 'PROCESSED')
    ORDER BY id ASC
"""


@dataclass(frozen=True)
class SourceDocumentSnapshotItem:
    doc_id: str
    doc_type: str = "general"
    document_name: str = ""
    legal_document_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


StorageCleaner = Callable[[WorkspaceGeneration], Awaitable[None]]


async def fetch_workspace_source_documents(
    workspace_id: str,
    *,
    db: Any = None,
) -> list[SourceDocumentSnapshotItem]:
    """Read-only snapshot of rebuildable documents for ``workspace_id``."""
    release_db = False
    if db is None:
        db = await ClientManager.get_client()
        release_db = True
    try:
        ws_uuid = to_uuid(workspace_id)
        general_rows = await db.query(_GENERAL_SNAPSHOT_SQL, [ws_uuid], multirows=True)
        legal_rows = await db.query(_LEGAL_SNAPSHOT_SQL, [ws_uuid], multirows=True)
        items: list[SourceDocumentSnapshotItem] = []
        for row in general_rows or []:
            raw_id = str(row["doc_id"])
            doc_id = raw_id if raw_id.startswith("doc-") else f"doc-{raw_id}"
            doc_name = row.get("document_name") or row.get("origin_path") or doc_id
            items.append(
                SourceDocumentSnapshotItem(
                    doc_id=doc_id,
                    doc_type="general",
                    document_name=doc_name,
                    metadata={
                        "document_name": doc_name,
                        "file_path": row.get("origin_path") or doc_name,
                        "doc_type": "general",
                    },
                )
            )
        for row in legal_rows or []:
            legal_id = int(row["legal_document_id"])
            doc_id = str(legal_id)
            title = row.get("title") or f"legal-{legal_id}.md"
            items.append(
                SourceDocumentSnapshotItem(
                    doc_id=doc_id,
                    doc_type="legal",
                    document_name=title,
                    legal_document_id=legal_id,
                    metadata={
                        "document_name": title,
                        "file_path": title,
                        "doc_type": "legal",
                        "legal_document_id": legal_id,
                    },
                )
            )
        return items
    finally:
        if release_db and db is not None:
            await ClientManager.release_client(db)


def build_generation_chunking_events(
    workspace_id: str,
    generation: WorkspaceGeneration,
    documents: Sequence[SourceDocumentSnapshotItem],
) -> list[PipelineStageEvent]:
    """Build deterministic chunking stage events for a generation manifest."""
    events: list[PipelineStageEvent] = []
    for item in documents:
        seed = f"{workspace_id}:{generation.generation_id}:{item.doc_id}"
        det_event_id = f"evt-{uuid.uuid5(_GENERATION_EVENT_NS, seed + ':event')}"
        det_run_id = f"run-{uuid.uuid5(_GENERATION_EVENT_NS, seed + ':run')}"
        doc_type = "legal" if item.doc_type == "legal" or item.legal_document_id else "general"
        meta = {
            **item.metadata,
            "doc_type": doc_type,
            "generation_build_run_id": generation.build_run_id,
        }
        if item.document_name and "document_name" not in meta:
            meta["document_name"] = item.document_name
        if item.legal_document_id is not None:
            meta["legal_document_id"] = item.legal_document_id
        events.append(
            PipelineStageEvent(
                event_id=det_event_id,
                run_id=det_run_id,
                workspace_id=workspace_id,
                doc_id=item.doc_id,
                generation_id=generation.generation_id,
                storage_workspace_id=generation.storage_workspace_id,
                doc_type=doc_type,
                stage=StageName.CHUNKING,
                shard_index=0,
                shard_count=1,
                created_at=generation.created_at.isoformat(),
                metadata=meta,
            )
        )
    return events


async def start_workspace_generation_rebuild(
    coordinator: PipelineCoordinator,
    workspace_id: str,
    *,
    documents: Sequence[SourceDocumentSnapshotItem] | None = None,
    generation_id: str | None = None,
    build_run_id: str | None = None,
    batch_size: int = 200,
    db: Any = None,
) -> tuple[WorkspaceGeneration, WorkspaceGenerationManifestStatus]:
    """Start or resume a shadow generation build and register its full manifest."""
    if documents is None:
        documents = await fetch_workspace_source_documents(workspace_id, db=db)

    existing = (
        await coordinator.get_workspace_generation(workspace_id, generation_id)
        if generation_id
        else None
    )
    if existing is not None and existing.status == WorkspaceGenerationStatus.BUILDING:
        generation = existing
    else:
        run_id = build_run_id or f"genbuild-{uuid.uuid4()}"
        generation = await coordinator.begin_workspace_generation(
            workspace_id=workspace_id,
            build_run_id=run_id,
            generation_id=generation_id,
        )

    events = build_generation_chunking_events(workspace_id, generation, documents)
    step = max(1, batch_size)
    for idx in range(0, len(events), step):
        await coordinator.add_workspace_generation_manifest_events(
            workspace_id,
            generation.generation_id,
            events[idx : idx + step],
        )

    await coordinator.complete_workspace_generation_manifest_snapshot(
        workspace_id,
        generation.generation_id,
        expected_documents=len(events),
    )
    await coordinator.resume_workspace_generation_manifest(
        workspace_id, generation.generation_id
    )
    status = await coordinator.get_workspace_generation_manifest_status(
        workspace_id, generation.generation_id
    )
    return generation, status


async def cutover_workspace_generation(
    coordinator: PipelineCoordinator,
    workspace_id: str,
    generation_id: str,
) -> WorkspaceGeneration:
    """Atomically promote a completed shadow generation to ``active``."""
    generation = await coordinator.get_workspace_generation(workspace_id, generation_id)
    if generation is None:
        raise WorkspaceGenerationTransitionError(
            f"Generation not found: workspace_id={workspace_id}, generation_id={generation_id}"
        )
    if generation.status == WorkspaceGenerationStatus.BUILDING:
        generation = await coordinator.mark_workspace_generation_ready(
            workspace_id, generation_id
        )
    if generation.status == WorkspaceGenerationStatus.READY:
        await coordinator.lock_workspace_for_generation(workspace_id, generation_id)
    return await coordinator.activate_workspace_generation(workspace_id, generation_id)


async def rollback_workspace_generation(
    coordinator: PipelineCoordinator,
    workspace_id: str,
    target_generation_id: str,
) -> WorkspaceGeneration:
    """Restore a retained generation (within the 7-day window) as ``active``."""
    generation = await coordinator.get_workspace_generation(
        workspace_id, target_generation_id
    )
    if generation is None or generation.status != WorkspaceGenerationStatus.RETAINED:
        raise WorkspaceGenerationTransitionError(
            f"Target generation {target_generation_id!r} is not in retained status"
        )
    await coordinator.lock_workspace_for_generation(workspace_id, target_generation_id)
    return await coordinator.activate_workspace_generation(
        workspace_id, target_generation_id
    )


async def _delete_shadow_postgres_rows(
    storage_workspace_id: str, *, db: Any = None
) -> None:
    release_db = False
    if db is None:
        db = await ClientManager.get_client()
        release_db = True
    try:
        await ensure_shadow_tables(db)
        await db.execute(
            f"DELETE FROM {SHADOW_TEXT_CHUNKS_TABLE} WHERE storage_workspace_id = $1",
            [storage_workspace_id],
        )
        await db.execute(
            f"DELETE FROM {SHADOW_DOC_STATUS_TABLE} WHERE storage_workspace_id = $1",
            [storage_workspace_id],
        )
    finally:
        if release_db and db is not None:
            await ClientManager.release_client(db)


async def cleanup_expired_retained_generations(
    coordinator: PipelineCoordinator,
    *,
    db: Any = None,
    storage_cleaner: StorageCleaner | None = None,
    now: datetime | None = None,
) -> list[str]:
    """Idempotently clean up shadow data for expired ``retained`` generations."""
    expired = await coordinator.list_expired_retained_generations(now=now)
    cleaned: list[str] = []
    for gen in expired:
        active_id = await coordinator.get_active_workspace_generation(gen.workspace_id)
        if gen.status != WorkspaceGenerationStatus.RETAINED or gen.generation_id == active_id:
            continue
        try:
            await _delete_shadow_postgres_rows(gen.storage_workspace_id, db=db)
            if storage_cleaner is not None:
                await storage_cleaner(gen)
            if await coordinator.mark_workspace_generation_cleaned(
                gen.workspace_id, gen.generation_id
            ):
                cleaned.append(gen.generation_id)
                logger.info(
                    "[Pipeline] Cleaned expired retained generation workspace_id=%s generation_id=%s storage_workspace_id=%s",
                    gen.workspace_id,
                    gen.generation_id,
                    gen.storage_workspace_id,
                )
        except Exception:
            logger.exception(
                "[Pipeline] Failed cleaning expired retained generation workspace_id=%s generation_id=%s",
                gen.workspace_id,
                gen.generation_id,
            )
    return cleaned

