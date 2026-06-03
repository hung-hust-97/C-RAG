from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageName(str, Enum):
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    CHUNK_EMBEDDING = "chunk_embedding"
    ENTITY_EXTRACTION = "entity_extraction"
    KG_MERGE = "kg_merge"
    KG_EMBEDDING = "kg_embedding"
    FINALIZE = "finalize"


class ArtifactRef(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    uri: str
    kind: str
    row_count: int = 0
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineError(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str
    stage: StageName | None = None
    error_type: str | None = None
    traceback: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class PipelineStageEvent(BaseModel):
    """Internal stage event independent from KMS-facing events."""

    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    event_id: str = Field(default_factory=lambda: f"evt-{uuid4()}")
    run_id: str = Field(default_factory=lambda: f"run-{uuid4()}")
    workspace_id: str
    doc_id: str
    doc_type: str = "general"
    stage: StageName
    artifact_uri: str | None = None
    manifest_uri: str | None = None
    shard_index: int = 0
    shard_count: int = 1
    attempt: int = 1
    status: str = "pending"
    counts: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    fast_path: bool = False
    debug_sync: bool = False
    created_at: str = Field(default_factory=utc_now_iso)

    @property
    def stage_name(self) -> StageName:
        return StageName(self.stage)

    def log_context(self) -> str:
        return (
            f"run_id={self.run_id} doc_id={self.doc_id} "
            f"workspace_id={self.workspace_id} stage={self.stage_name.value} "
            f"shard={self.shard_index}/{self.shard_count} "
            f"event_id={self.event_id} attempt={self.attempt}"
        )

    def partition_key(self) -> str:
        return (
            f"{self.workspace_id}:{self.run_id}:{self.doc_id}:"
            f"{self.stage_name.value}:{self.shard_index}"
        )

    def next_event(
        self,
        stage: StageName,
        *,
        artifact_uri: str | None = None,
        manifest_uri: str | None = None,
        shard_index: int = 0,
        shard_count: int = 1,
        counts: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PipelineStageEvent":
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)

        return PipelineStageEvent(
            run_id=self.run_id,
            workspace_id=self.workspace_id,
            doc_id=self.doc_id,
            doc_type=self.doc_type,
            stage=stage,
            artifact_uri=artifact_uri,
            manifest_uri=manifest_uri,
            shard_index=shard_index,
            shard_count=shard_count,
            attempt=1,
            counts=counts or dict(self.counts),
            metadata=merged_metadata,
            fast_path=self.fast_path,
            debug_sync=self.debug_sync,
        )


class StageResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)

    output_artifacts: list[ArtifactRef] = Field(default_factory=list)
    manifest: ArtifactRef | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    next_events: list[PipelineStageEvent] = Field(default_factory=list)
    join_next_event: PipelineStageEvent | None = None
    finalize_event: PipelineStageEvent | None = None
    finalize_after_stages: list[StageName] = Field(default_factory=list)
    message: str | None = None
