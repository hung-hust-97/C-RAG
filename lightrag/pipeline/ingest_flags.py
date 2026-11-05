"""Feature flags and helpers for migrating ingest/reprocess entrypoints to Kafka.

Flags:
- ``PIPELINE_INGEST_BACKEND``: ``kafka`` | ``celery`` | ``sync``.
  When unset, falls back to ``celery`` if ``EMBEDDING_USE_CELERY=true`` else ``sync``.
- ``PIPELINE_REPROCESS_BACKEND``: ``kafka`` | ``celery``.
  When unset, defaults to ``celery`` for backward compatibility.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Sequence

from ..utils import logger
from .coordinator import PipelineCoordinator, create_coordinator_from_env
from .models import PipelineStageEvent, StageName

IngestBackend = Literal["kafka", "celery", "sync"]
ReprocessBackend = Literal["kafka", "celery"]


def resolve_ingest_backend() -> IngestBackend:
    """Return the configured ingest backend with legacy Celery fallback."""
    raw = os.getenv("PIPELINE_INGEST_BACKEND", "").strip().lower()
    if raw in {"kafka", "celery", "sync"}:
        return raw  # type: ignore[return-value]
    if raw:
        logger.warning(
            "[Pipeline] Unknown PIPELINE_INGEST_BACKEND=%r; falling back to legacy setting",
            raw,
        )
    use_celery = os.getenv("EMBEDDING_USE_CELERY", "false").strip().lower() == "true"
    return "celery" if use_celery else "sync"


def resolve_reprocess_backend() -> ReprocessBackend:
    """Return the configured reprocess backend (defaults to celery)."""
    raw = os.getenv("PIPELINE_REPROCESS_BACKEND", "").strip().lower()
    if raw in {"kafka", "celery"}:
        return raw  # type: ignore[return-value]
    if raw:
        logger.warning(
            "[Pipeline] Unknown PIPELINE_REPROCESS_BACKEND=%r; defaulting to 'celery'",
            raw,
        )
    return "celery"


def build_ingest_stage_event(
    *,
    workspace_id: str,
    doc_id: str,
    stage: StageName,
    doc_type: str = "general",
    file_path: str = "",
    metadata: dict[str, Any] | None = None,
) -> PipelineStageEvent:
    canonical_doc_type = "legal" if str(doc_type).lower() == "legal" else "general"
    merged_meta = {
        **(metadata or {}),
        "doc_type": canonical_doc_type,
    }
    if file_path:
        merged_meta.setdefault("file_path", file_path)
        merged_meta.setdefault("document_name", os.path.basename(file_path))
    return PipelineStageEvent(
        workspace_id=workspace_id or "default",
        doc_id=doc_id,
        doc_type=canonical_doc_type,
        stage=stage,
        shard_index=0,
        shard_count=1,
        metadata=merged_meta,
    )


async def enqueue_kafka_stage_events(
    events: Sequence[PipelineStageEvent],
    *,
    coordinator: PipelineCoordinator | None = None,
) -> int:
    """Persist ``events`` into the transactional coordinator outbox."""
    if not events:
        return 0
    own_coordinator = coordinator is None
    active_coordinator = coordinator or create_coordinator_from_env()
    try:
        await active_coordinator.enqueue_events(list(events))
        return len(events)
    finally:
        if own_coordinator:
            await active_coordinator.finalize()

