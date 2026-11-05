from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from lightrag.utils import logger

from .models import ArtifactRef, PipelineError, PipelineStageEvent, StageName, utc_now_iso

MAX_STAGE_RETRY_ATTEMPTS_DEFAULT = 3


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


def max_stage_retry_attempts() -> int:
    max_attempts = _env_int(
        "PIPELINE_MAX_STAGE_RETRY_ATTEMPTS", MAX_STAGE_RETRY_ATTEMPTS_DEFAULT
    )
    if max_attempts < 0:
        logger.warning(
            "[Pipeline] Invalid PIPELINE_MAX_STAGE_RETRY_ATTEMPTS=%s, using default=%s",
            max_attempts,
            MAX_STAGE_RETRY_ATTEMPTS_DEFAULT,
        )
        return MAX_STAGE_RETRY_ATTEMPTS_DEFAULT
    return max_attempts


def _stage_env_name(prefix: str, stage: StageName | str) -> str:
    return f"{prefix}_{StageName(stage).value.upper()}"


def _stage_int_env(prefix: str, stage: StageName | str, default: int) -> int:
    stage_env_name = _stage_env_name(prefix, stage)
    raw_value = os.getenv(stage_env_name)
    env_name = stage_env_name
    if raw_value is None:
        raw_value = os.getenv(prefix)
        env_name = prefix
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "[Pipeline] Invalid %s=%r, using default=%s",
            env_name,
            raw_value,
            default,
        )
        return default


class ClaimStatus(str, Enum):
    ACQUIRED = "acquired"
    ALREADY_COMPLETED = "already_completed"
    ALREADY_RUNNING = "already_running"
    FAILED_ATTEMPT_RECORDED = "failed_attempt_recorded"
    WORKSPACE_WRITE_LOCKED = "workspace_write_locked"


@dataclass(frozen=True)
class ClaimResult:
    status: ClaimStatus
    message: str | None = None

    @property
    def acquired(self) -> bool:
        return self.status == ClaimStatus.ACQUIRED


@dataclass(frozen=True)
class CompletionResult:
    all_shards_done: bool
    done_count: int
    shard_count: int
    next_events: list[PipelineStageEvent] = field(default_factory=list)


@dataclass(frozen=True)
class StageReplayResult:
    event: PipelineStageEvent
    status: str


class WorkspaceGenerationStatus(str, Enum):
    BUILDING = "building"
    READY = "ready"
    ACTIVE = "active"
    RETAINED = "retained"
    FAILED = "failed"


@dataclass(frozen=True)
class WorkspaceGeneration:
    workspace_id: str
    generation_id: str
    storage_workspace_id: str
    build_run_id: str
    status: WorkspaceGenerationStatus
    created_at: datetime
    updated_at: datetime
    ready_at: datetime | None = None
    activated_at: datetime | None = None
    retain_until: datetime | None = None
    error: dict[str, Any] | None = None


@dataclass(frozen=True)
class WorkspaceGenerationManifestStatus:
    workspace_id: str
    generation_id: str
    snapshot_complete: bool
    expected_documents: int | None
    pending_documents: int
    running_documents: int
    completed_documents: int
    failed_documents: int


@dataclass(frozen=True)
class WorkspaceWriteLease:
    lease_id: str
    workspace_id: str
    expires_at: datetime


class StageRunNotFoundError(LookupError):
    pass


class StageNotReplayableError(RuntimeError):
    pass


class StageRetryBudgetExhausted(PermissionError):
    pass


class WorkspaceGenerationConflictError(RuntimeError):
    pass


class WorkspaceGenerationNotFoundError(LookupError):
    pass


class WorkspaceGenerationTransitionError(RuntimeError):
    pass


OutboxPublisher = Callable[[PipelineStageEvent], Awaitable[None]]


def _stage_key(run_id: str, stage: StageName | str, shard_index: int) -> tuple[str, str, int]:
    return run_id, StageName(stage).value, shard_index


def _artifact_to_json(artifact: ArtifactRef) -> dict[str, Any]:
    return artifact.model_dump(mode="json")


def _artifact_from_json(data: dict[str, Any]) -> ArtifactRef:
    return ArtifactRef.model_validate(data)


def _is_transient_postgres_connect_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "database system is in recovery mode",
            "database system is not yet accepting connections",
            "consistent recovery state has not been yet reached",
            "connection refused",
            "connection reset",
            "server closed the connection",
            "connection was closed",
        )
    )


class PipelineCoordinator:
    async def acquire_workspace_write_lease(
        self, workspace_id: str, lease_seconds: int = 600
    ) -> WorkspaceWriteLease:
        raise NotImplementedError

    async def renew_workspace_write_lease(
        self, lease: WorkspaceWriteLease, lease_seconds: int = 600
    ) -> WorkspaceWriteLease:
        raise NotImplementedError

    async def release_workspace_write_lease(
        self, lease: WorkspaceWriteLease
    ) -> None:
        raise NotImplementedError

    async def begin_workspace_generation(
        self, workspace_id: str, build_run_id: str, generation_id: str | None = None
    ) -> WorkspaceGeneration:
        raise NotImplementedError

    async def add_workspace_generation_manifest_events(
        self,
        workspace_id: str,
        generation_id: str,
        events: list[PipelineStageEvent],
    ) -> int:
        raise NotImplementedError

    async def complete_workspace_generation_manifest_snapshot(
        self, workspace_id: str, generation_id: str, expected_documents: int
    ) -> None:
        raise NotImplementedError

    async def get_workspace_generation_manifest_status(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGenerationManifestStatus:
        raise NotImplementedError

    async def resume_workspace_generation_manifest(
        self, workspace_id: str, generation_id: str
    ) -> int:
        raise NotImplementedError

    async def mark_workspace_generation_ready(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGeneration:
        raise NotImplementedError

    async def fail_workspace_generation(
        self, workspace_id: str, generation_id: str, error: dict[str, Any]
    ) -> WorkspaceGeneration:
        raise NotImplementedError

    async def activate_workspace_generation(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGeneration:
        raise NotImplementedError

    async def lock_workspace_for_generation(
        self, workspace_id: str, generation_id: str
    ) -> None:
        raise NotImplementedError

    async def get_workspace_generation(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGeneration | None:
        raise NotImplementedError

    async def get_active_workspace_generation(self, workspace_id: str) -> str | None:
        raise NotImplementedError

    async def is_workspace_write_locked(self, workspace_id: str) -> bool:
        raise NotImplementedError

    async def list_expired_retained_generations(
        self, *, now: datetime | None = None
    ) -> list[WorkspaceGeneration]:
        raise NotImplementedError

    async def mark_workspace_generation_cleaned(
        self, workspace_id: str, generation_id: str
    ) -> bool:
        raise NotImplementedError

    async def claim(self, event: PipelineStageEvent) -> ClaimResult:
        raise NotImplementedError

    async def complete(
        self, event: PipelineStageEvent, artifacts: list[ArtifactRef]
    ) -> CompletionResult:
        raise NotImplementedError

    async def complete_and_enqueue(
        self,
        event: PipelineStageEvent,
        artifacts: list[ArtifactRef],
        *,
        next_events: list[PipelineStageEvent],
        join_next_event: PipelineStageEvent | None = None,
        finalize_event: PipelineStageEvent | None = None,
        finalize_after_stages: list[StageName] | None = None,
    ) -> CompletionResult:
        completion = await self.complete(event, artifacts)
        outgoing_events = list(next_events)
        if join_next_event and completion.all_shards_done:
            stage_outputs = await self.get_stage_outputs(event.run_id, event.stage_name)
            metadata = dict(join_next_event.metadata)
            metadata["input_artifact_uris"] = [
                artifact.uri
                for artifact in stage_outputs
                if artifact.kind != "manifest"
            ]
            metadata["input_manifest_uris"] = [
                artifact.uri
                for artifact in stage_outputs
                if artifact.kind == "manifest"
            ]
            outgoing_events.append(
                join_next_event.model_copy(update={"metadata": metadata})
            )
        if finalize_event and finalize_after_stages:
            if await self.are_stages_complete(event.run_id, finalize_after_stages):
                outgoing_events.append(finalize_event)
        if outgoing_events and not event.fast_path:
            await self.enqueue_events(outgoing_events)
        return CompletionResult(
            completion.all_shards_done,
            completion.done_count,
            completion.shard_count,
            outgoing_events,
        )

    async def enqueue_events(self, events: list[PipelineStageEvent]) -> None:
        raise NotImplementedError

    async def publish_outbox(
        self,
        publisher: OutboxPublisher,
        *,
        event_ids: list[str] | None = None,
        limit: int = 100,
    ) -> int:
        raise NotImplementedError

    async def replay_failed_stage(
        self,
        *,
        run_id: str,
        stage: StageName | str,
        shard_index: int,
        workspace_id: str,
        reason: str,
        requested_by: str,
    ) -> StageReplayResult:
        raise NotImplementedError

    async def fail(self, event: PipelineStageEvent, error: PipelineError) -> None:
        raise NotImplementedError

    async def heartbeat(self, event: PipelineStageEvent) -> None:
        pass

    async def get_stage_outputs(
        self, run_id: str, stage: StageName | str
    ) -> list[ArtifactRef]:
        raise NotImplementedError

    async def are_stages_complete(
        self, run_id: str, stages: list[StageName | str]
    ) -> bool:
        raise NotImplementedError

    async def has_any_stage_records(
        self, run_id: str, stages: list[StageName | str]
    ) -> bool:
        raise NotImplementedError

    async def finalize(self) -> None:
        pass

    async def register_paused(self, event: PipelineStageEvent) -> None:
        raise NotImplementedError


class InMemoryPipelineCoordinator(PipelineCoordinator):
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._records: dict[tuple[str, str, int], dict[str, Any]] = {}
        self._outbox: dict[str, dict[str, Any]] = {}
        self._generations: dict[tuple[str, str], WorkspaceGeneration] = {}
        self._generation_manifest: dict[
            tuple[str, str, str], dict[str, Any]
        ] = {}
        self._generation_manifest_snapshots: dict[
            tuple[str, str], tuple[bool, int | None]
        ] = {}
        self._active_generations: dict[str, str] = {}
        self._write_locks: dict[str, str] = {}
        self._write_leases: dict[str, WorkspaceWriteLease] = {}

    async def acquire_workspace_write_lease(
        self, workspace_id: str, lease_seconds: int = 600
    ) -> WorkspaceWriteLease:
        if not workspace_id.strip() or lease_seconds <= 0:
            raise ValueError("workspace_id and lease_seconds must be valid")
        now = datetime.now(timezone.utc)
        async with self._lock:
            self._write_leases = {
                lease_id: lease
                for lease_id, lease in self._write_leases.items()
                if lease.expires_at > now
            }
            if workspace_id in self._write_locks:
                raise WorkspaceGenerationConflictError(
                    f"Workspace is locked for generation operation: {workspace_id}"
                )
            lease = WorkspaceWriteLease(
                lease_id=str(uuid4()),
                workspace_id=workspace_id,
                expires_at=now + timedelta(seconds=lease_seconds),
            )
            self._write_leases[lease.lease_id] = lease
            return lease

    async def renew_workspace_write_lease(
        self, lease: WorkspaceWriteLease, lease_seconds: int = 600
    ) -> WorkspaceWriteLease:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = datetime.now(timezone.utc)
        async with self._lock:
            current = self._write_leases.get(lease.lease_id)
            if current is None or current.expires_at <= now:
                raise WorkspaceGenerationConflictError(
                    f"Workspace write lease expired: {lease.lease_id}"
                )
            if current.workspace_id in self._write_locks:
                raise WorkspaceGenerationConflictError(
                    f"Workspace was locked during write lease: {current.workspace_id}"
                )
            renewed = replace(
                current, expires_at=now + timedelta(seconds=lease_seconds)
            )
            self._write_leases[lease.lease_id] = renewed
            return renewed

    async def release_workspace_write_lease(
        self, lease: WorkspaceWriteLease
    ) -> None:
        async with self._lock:
            self._write_leases.pop(lease.lease_id, None)

    async def begin_workspace_generation(
        self, workspace_id: str, build_run_id: str, generation_id: str | None = None
    ) -> WorkspaceGeneration:
        generation_id = generation_id or str(uuid4())
        key = (workspace_id, generation_id)
        now = datetime.now(timezone.utc)
        async with self._lock:
            existing = self._generations.get(key)
            if existing is not None:
                if (
                    existing.build_run_id == build_run_id
                    and existing.status == WorkspaceGenerationStatus.BUILDING
                    and self._write_locks.get(workspace_id) == generation_id
                ):
                    return existing
                raise WorkspaceGenerationConflictError(
                    f"Generation already exists: workspace_id={workspace_id}, generation_id={generation_id}"
                )
            if workspace_id in self._write_locks:
                raise WorkspaceGenerationConflictError(
                    f"Workspace already has a generation build in progress: {workspace_id}"
                )
            now = datetime.now(timezone.utc)
            self._write_leases = {
                lease_id: active_lease
                for lease_id, active_lease in self._write_leases.items()
                if active_lease.expires_at > now
            }
            if any(
                lease.workspace_id == workspace_id
                for lease in self._write_leases.values()
            ):
                raise WorkspaceGenerationConflictError(
                    f"Workspace has admitted writes in progress: {workspace_id}"
                )
            if any(
                record.get("workspace_id") == workspace_id
                and record.get("status") == "running"
                for record in self._records.values()
            ):
                raise WorkspaceGenerationConflictError(
                    f"Workspace has pipeline stages in progress: {workspace_id}"
                )
            generation = WorkspaceGeneration(
                workspace_id=workspace_id,
                generation_id=generation_id,
                storage_workspace_id=str(uuid4()),
                build_run_id=build_run_id,
                status=WorkspaceGenerationStatus.BUILDING,
                created_at=now,
                updated_at=now,
            )
            self._generations[key] = generation
            self._write_locks[workspace_id] = generation_id
            return generation

    @staticmethod
    def _validate_generation_manifest_events(
        workspace_id: str,
        generation: WorkspaceGeneration,
        events: list[PipelineStageEvent],
    ) -> None:
        for event in events:
            if (
                event.workspace_id != workspace_id
                or event.generation_id != generation.generation_id
                or event.storage_workspace_id != generation.storage_workspace_id
                or not event.doc_id
                or event.stage_name != StageName.CHUNKING
            ):
                raise ValueError(
                    "Generation manifest events must target the locked generation "
                    "and begin at the chunking stage"
                )

    async def add_workspace_generation_manifest_events(
        self,
        workspace_id: str,
        generation_id: str,
        events: list[PipelineStageEvent],
    ) -> int:
        async with self._lock:
            generation = self._generations.get((workspace_id, generation_id))
            if (
                generation is None
                or generation.status != WorkspaceGenerationStatus.BUILDING
                or self._write_locks.get(workspace_id) != generation_id
            ):
                raise WorkspaceGenerationTransitionError(
                    "Manifest can only be extended while its generation is building "
                    "and holds the workspace write lock"
                )
            self._validate_generation_manifest_events(
                workspace_id, generation, events
            )
            snapshot_complete, _ = self._generation_manifest_snapshots.get(
                (workspace_id, generation_id), (False, None)
            )
            if snapshot_complete and any(
                (workspace_id, generation_id, event.doc_id)
                not in self._generation_manifest
                for event in events
            ):
                raise WorkspaceGenerationTransitionError(
                    "Cannot add documents after the generation manifest snapshot "
                    "has completed"
                )
            added = 0
            for event in events:
                key = (workspace_id, generation_id, event.doc_id)
                existing = self._generation_manifest.get(key)
                serialized_event = event.model_dump(mode="json")
                if existing is not None:
                    if existing["event"] != serialized_event:
                        raise WorkspaceGenerationConflictError(
                            "Generation manifest document changed during resume: "
                            f"workspace_id={workspace_id}, generation_id={generation_id}, "
                            f"doc_id={event.doc_id}"
                        )
                    continue
                self._generation_manifest[key] = {
                    "event": serialized_event,
                    "status": "pending",
                    "error": None,
                }
                self._outbox.setdefault(
                    event.event_id,
                    {"event": event, "status": "pending", "attempts": 0},
                )
                added += 1
            return added

    async def complete_workspace_generation_manifest_snapshot(
        self, workspace_id: str, generation_id: str, expected_documents: int
    ) -> None:
        if expected_documents < 0:
            raise ValueError("expected_documents cannot be negative")
        async with self._lock:
            generation = self._generations.get((workspace_id, generation_id))
            if (
                generation is None
                or generation.status != WorkspaceGenerationStatus.BUILDING
                or self._write_locks.get(workspace_id) != generation_id
            ):
                raise WorkspaceGenerationTransitionError(
                    "Manifest snapshot can only be completed while its generation "
                    "is building and holds the workspace write lock"
                )
            actual_documents = sum(
                1
                for (manifest_workspace, manifest_generation, _doc_id)
                in self._generation_manifest
                if manifest_workspace == workspace_id
                and manifest_generation == generation_id
            )
            if actual_documents != expected_documents:
                raise WorkspaceGenerationTransitionError(
                    "Manifest snapshot count does not match registered documents: "
                    f"expected={expected_documents}, registered={actual_documents}"
                )
            self._generation_manifest_snapshots[(workspace_id, generation_id)] = (
                True,
                expected_documents,
            )

    async def get_workspace_generation_manifest_status(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGenerationManifestStatus:
        async with self._lock:
            if (workspace_id, generation_id) not in self._generations:
                raise WorkspaceGenerationNotFoundError(
                    f"Generation not found: workspace_id={workspace_id}, "
                    f"generation_id={generation_id}"
                )
            snapshot_complete, expected_documents = (
                self._generation_manifest_snapshots.get(
                    (workspace_id, generation_id), (False, None)
                )
            )
            entries = [
                entry
                for (entry_workspace, entry_generation, _doc_id), entry
                in self._generation_manifest.items()
                if entry_workspace == workspace_id
                and entry_generation == generation_id
            ]
            counts = {
                status: sum(entry["status"] == status for entry in entries)
                for status in ("pending", "running", "completed", "failed")
            }
            return WorkspaceGenerationManifestStatus(
                workspace_id=workspace_id,
                generation_id=generation_id,
                snapshot_complete=snapshot_complete,
                expected_documents=expected_documents,
                pending_documents=counts["pending"],
                running_documents=counts["running"],
                completed_documents=counts["completed"],
                failed_documents=counts["failed"],
            )

    async def resume_workspace_generation_manifest(
        self, workspace_id: str, generation_id: str
    ) -> int:
        async with self._lock:
            generation = self._generations.get((workspace_id, generation_id))
            if (
                generation is None
                or generation.status != WorkspaceGenerationStatus.BUILDING
                or self._write_locks.get(workspace_id) != generation_id
            ):
                raise WorkspaceGenerationTransitionError(
                    "Only a locked building generation can resume its manifest"
                )
            resumed = 0
            for outbox_entry in self._outbox.values():
                event = outbox_entry["event"]
                if (
                    event.workspace_id != workspace_id
                    or event.generation_id != generation_id
                    or outbox_entry["status"] != "published"
                ):
                    continue
                stage_key = _stage_key(
                    event.run_id, event.stage_name, event.shard_index
                )
                stage_record = self._records.get(stage_key)
                if stage_record and stage_record.get("status") == "done":
                    continue
                outbox_entry["status"] = "pending"
                resumed += 1
            return resumed

    async def mark_workspace_generation_ready(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGeneration:
        key = (workspace_id, generation_id)
        now = datetime.now(timezone.utc)
        async with self._lock:
            generation = self._generations.get(key)
            if generation is None:
                raise WorkspaceGenerationNotFoundError(
                    f"Generation not found: workspace_id={workspace_id}, generation_id={generation_id}"
                )
            if generation.status == WorkspaceGenerationStatus.READY:
                if self._write_locks.get(workspace_id) != generation_id:
                    raise WorkspaceGenerationTransitionError(
                        "A ready generation must still hold the workspace write lock"
                    )
                return generation
            if (
                generation.status != WorkspaceGenerationStatus.BUILDING
                or self._write_locks.get(workspace_id) != generation_id
            ):
                raise WorkspaceGenerationTransitionError(
                    f"Cannot mark generation ready from status={generation.status.value}"
                )
            if any(
                record.get("status") == "running"
                and record.get("workspace_id") == workspace_id
                and (record.get("event") or {}).get("generation_id")
                == generation_id
                for record in self._records.values()
            ):
                raise WorkspaceGenerationTransitionError(
                    "Cannot mark generation ready while its pipeline stages are running"
                )
            snapshot_complete, expected_documents = (
                self._generation_manifest_snapshots.get(
                    (workspace_id, generation_id), (False, None)
                )
            )
            manifest_entries = [
                entry
                for (entry_workspace, entry_generation, _doc_id), entry
                in self._generation_manifest.items()
                if entry_workspace == workspace_id
                and entry_generation == generation_id
            ]
            if (
                not snapshot_complete
                or expected_documents != len(manifest_entries)
                or any(entry["status"] != "completed" for entry in manifest_entries)
            ):
                raise WorkspaceGenerationTransitionError(
                    "Cannot mark generation ready until its document manifest "
                    "snapshot is complete and every document has completed"
                )

            updated = replace(
                generation,
                status=WorkspaceGenerationStatus.READY,
                updated_at=now,
                ready_at=now,
            )
            self._generations[key] = updated
            return updated

    async def fail_workspace_generation(
        self, workspace_id: str, generation_id: str, error: dict[str, Any]
    ) -> WorkspaceGeneration:
        key = (workspace_id, generation_id)
        now = datetime.now(timezone.utc)
        async with self._lock:
            generation = self._generations.get(key)
            if generation is None:
                raise WorkspaceGenerationNotFoundError(
                    f"Generation not found: workspace_id={workspace_id}, generation_id={generation_id}"
                )
            if generation.status == WorkspaceGenerationStatus.FAILED:
                if self._write_locks.get(workspace_id) == generation_id:
                    del self._write_locks[workspace_id]
                return generation
            if generation.status not in {
                WorkspaceGenerationStatus.BUILDING,
                WorkspaceGenerationStatus.READY,
            }:
                raise WorkspaceGenerationTransitionError(
                    f"Cannot fail generation from status={generation.status.value}"
                )
            updated = replace(
                generation,
                status=WorkspaceGenerationStatus.FAILED,
                updated_at=now,
                error=error,
            )
            self._generations[key] = updated
            if self._write_locks.get(workspace_id) == generation_id:
                del self._write_locks[workspace_id]
            return updated

    async def activate_workspace_generation(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGeneration:
        key = (workspace_id, generation_id)
        now = datetime.now(timezone.utc)
        retain_until = now + timedelta(days=7)
        async with self._lock:
            generation = self._generations.get(key)
            if generation is None:
                raise WorkspaceGenerationNotFoundError(
                    f"Generation not found: workspace_id={workspace_id}, generation_id={generation_id}"
                )
            if generation.status == WorkspaceGenerationStatus.ACTIVE:
                return generation
            can_activate = generation.status == WorkspaceGenerationStatus.READY
            if generation.status == WorkspaceGenerationStatus.RETAINED:
                can_activate = bool(generation.retain_until and generation.retain_until > now)
            if not can_activate:
                raise WorkspaceGenerationTransitionError(
                    f"Cannot activate generation from status={generation.status.value}"
                )
            if self._write_locks.get(workspace_id) != generation_id:
                raise WorkspaceGenerationTransitionError(
                    "Workspace writes must be locked for the generation being activated"
                )
            if any(
                record.get("status") == "running"
                and record.get("workspace_id") == workspace_id
                and (record.get("event") or {}).get("generation_id")
                == generation_id
                for record in self._records.values()
            ):
                raise WorkspaceGenerationTransitionError(
                    "Cannot activate generation while its pipeline stages are running"
                )

            old_generation_id = self._active_generations.get(workspace_id)
            if old_generation_id and old_generation_id != generation_id:
                old_key = (workspace_id, old_generation_id)
                old_generation = self._generations[old_key]
                self._generations[old_key] = replace(
                    old_generation,
                    status=WorkspaceGenerationStatus.RETAINED,
                    updated_at=now,
                    retain_until=retain_until,
                )

            activated = replace(
                generation,
                status=WorkspaceGenerationStatus.ACTIVE,
                updated_at=now,
                activated_at=now,
                retain_until=None,
            )
            self._generations[key] = activated
            self._active_generations[workspace_id] = generation_id
            if self._write_locks.get(workspace_id) == generation_id:
                del self._write_locks[workspace_id]
            return activated

    async def lock_workspace_for_generation(
        self, workspace_id: str, generation_id: str
    ) -> None:
        key = (workspace_id, generation_id)
        now = datetime.now(timezone.utc)
        async with self._lock:
            generation = self._generations.get(key)
            if generation is None:
                raise WorkspaceGenerationNotFoundError(
                    f"Generation not found: workspace_id={workspace_id}, generation_id={generation_id}"
                )
            allowed = generation.status in {
                WorkspaceGenerationStatus.BUILDING,
                WorkspaceGenerationStatus.READY,
            }
            if generation.status == WorkspaceGenerationStatus.RETAINED:
                allowed = bool(generation.retain_until and generation.retain_until > now)
            if not allowed:
                raise WorkspaceGenerationTransitionError(
                    f"Cannot lock workspace for generation status={generation.status.value}"
                )
            locked_id = self._write_locks.get(workspace_id)
            if locked_id and locked_id != generation_id:
                raise WorkspaceGenerationConflictError(
                    f"Workspace already has a generation operation in progress: {workspace_id}"
                )
            self._write_locks[workspace_id] = generation_id

    async def get_workspace_generation(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGeneration | None:
        async with self._lock:
            return self._generations.get((workspace_id, generation_id))

    async def get_active_workspace_generation(self, workspace_id: str) -> str | None:
        async with self._lock:
            return self._active_generations.get(workspace_id)

    async def is_workspace_write_locked(self, workspace_id: str) -> bool:
        async with self._lock:
            return workspace_id in self._write_locks

    async def list_expired_retained_generations(
        self, *, now: datetime | None = None
    ) -> list[WorkspaceGeneration]:
        ref_now = now or datetime.now(timezone.utc)
        async with self._lock:
            expired: list[WorkspaceGeneration] = []
            for gen in self._generations.values():
                if gen.status != WorkspaceGenerationStatus.RETAINED:
                    continue
                if gen.retain_until is None or gen.retain_until > ref_now:
                    continue
                if self._active_generations.get(gen.workspace_id) == gen.generation_id:
                    continue
                if self._write_locks.get(gen.workspace_id) == gen.generation_id:
                    continue
                expired.append(gen)
            return expired

    async def mark_workspace_generation_cleaned(
        self, workspace_id: str, generation_id: str
    ) -> bool:
        async with self._lock:
            key = (workspace_id, generation_id)
            gen = self._generations.get(key)
            if (
                gen is None
                or gen.status != WorkspaceGenerationStatus.RETAINED
                or self._active_generations.get(workspace_id) == generation_id
                or self._write_locks.get(workspace_id) == generation_id
            ):
                return False
            if gen.retain_until is None:
                return False
            self._generations[key] = replace(
                gen,
                retain_until=None,
                updated_at=datetime.now(timezone.utc),
            )
            return True

    async def claim(self, event: PipelineStageEvent) -> ClaimResult:
        key = _stage_key(event.run_id, event.stage_name, event.shard_index)
        max_attempts = max_stage_retry_attempts()
        async with self._lock:
            locked_generation_id = self._write_locks.get(event.workspace_id)
            if locked_generation_id is not None:
                generation = self._generations.get(
                    (event.workspace_id, locked_generation_id)
                )
                if (
                    event.generation_id != locked_generation_id
                    or generation is None
                    or generation.status != WorkspaceGenerationStatus.BUILDING
                    or not self._generation_manifest_snapshots.get(
                        (event.workspace_id, locked_generation_id), (False, None)
                    )[0]
                    or event.storage_workspace_id
                    != generation.storage_workspace_id
                ):
                    return ClaimResult(ClaimStatus.WORKSPACE_WRITE_LOCKED)
            elif event.generation_id is not None or event.storage_workspace_id is not None:
                return ClaimResult(ClaimStatus.WORKSPACE_WRITE_LOCKED)
            record = self._records.get(key)
            if event.generation_id is not None:
                manifest_entry = self._generation_manifest.get(
                    (event.workspace_id, event.generation_id, event.doc_id)
                )
                if manifest_entry is None or (
                    manifest_entry["status"] == "completed"
                    and (record is None or record.get("status") != "done")
                ):
                    return ClaimResult(ClaimStatus.WORKSPACE_WRITE_LOCKED)
            attempt = event.attempt
            if record:
                status = record["status"]
                if status == "done":
                    return ClaimResult(ClaimStatus.ALREADY_COMPLETED)
                if status == "running":
                    return ClaimResult(ClaimStatus.ALREADY_RUNNING)
                if status == "failed":
                    recorded_attempt = record.get("attempt", 0)
                    if recorded_attempt >= max_attempts:
                        return ClaimResult(
                            ClaimStatus.FAILED_ATTEMPT_RECORDED,
                            f"max retries ({max_attempts}) exhausted at attempt={recorded_attempt}",
                        )
                    attempt = max(recorded_attempt + 1, event.attempt)
                    logger.info(
                        "[Pipeline] Auto-retrying failed stage run_id=%s stage=%s attempt=%s/%s",
                        event.run_id,
                        StageName(event.stage_name).value if isinstance(event.stage_name, str) else event.stage_name.value,
                        recorded_attempt + 1,
                        max_attempts,
                    )
                    # Fall through to update record with incremented attempt

            self._records[key] = {
                "status": "running",
                "run_id": event.run_id,
                "stage": event.stage_name.value,
                "shard_index": event.shard_index,
                "shard_count": event.shard_count,
                "attempt": attempt,
                "workspace_id": event.workspace_id,
                "doc_id": event.doc_id,
                "event": event.model_dump(mode="json"),
                "artifacts": [],
                "started_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
                "completed_at": None,
                "error": None,
            }
            if event.generation_id is not None:
                manifest_entry = self._generation_manifest.get(
                    (event.workspace_id, event.generation_id, event.doc_id)
                )
                if manifest_entry is not None:
                    manifest_entry["status"] = "running"
                    manifest_entry["error"] = None
            return ClaimResult(ClaimStatus.ACQUIRED)

    async def register_paused(self, event: PipelineStageEvent) -> None:
        key = _stage_key(event.run_id, event.stage_name, event.shard_index)
        async with self._lock:
            self._records[key] = {
                "status": "paused",
                "run_id": event.run_id,
                "stage": event.stage_name.value,
                "shard_index": event.shard_index,
                "shard_count": event.shard_count,
                "attempt": event.attempt,
                "workspace_id": event.workspace_id,
                "doc_id": event.doc_id,
                "event": event.model_dump(mode="json"),
                "artifacts": [],
                "started_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
                "completed_at": None,
                "error": None,
            }

    async def heartbeat(self, event: PipelineStageEvent) -> None:
        key = _stage_key(event.run_id, event.stage_name, event.shard_index)
        async with self._lock:
            record = self._records.get(key)
            if record and record.get("status") == "running":
                record["updated_at"] = utc_now_iso()

    async def complete(
        self, event: PipelineStageEvent, artifacts: list[ArtifactRef]
    ) -> CompletionResult:
        key = _stage_key(event.run_id, event.stage_name, event.shard_index)
        async with self._lock:
            record = self._records.setdefault(
                key,
                {
                    "run_id": event.run_id,
                    "stage": event.stage_name.value,
                    "shard_index": event.shard_index,
                    "shard_count": event.shard_count,
                },
            )
            record.update(
                {
                    "status": "done",
                    "attempt": event.attempt,
                    "artifacts": [_artifact_to_json(a) for a in artifacts],
                    "completed_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                    "error": None,
                }
            )
            if (
                event.generation_id is not None
                and event.stage_name == StageName.FINALIZE
            ):
                manifest_entry = self._generation_manifest.get(
                    (event.workspace_id, event.generation_id, event.doc_id)
                )
                if manifest_entry is not None:
                    manifest_entry["status"] = "completed"
                    manifest_entry["error"] = None
            done_count = sum(
                1
                for (run_id, stage, _), item in self._records.items()
                if run_id == event.run_id
                and stage == event.stage_name.value
                and item.get("status") == "done"
            )
            return CompletionResult(
                all_shards_done=done_count >= event.shard_count,
                done_count=done_count,
                shard_count=event.shard_count,
                next_events=[],
            )

    async def complete_and_enqueue(
        self,
        event: PipelineStageEvent,
        artifacts: list[ArtifactRef],
        *,
        next_events: list[PipelineStageEvent],
        join_next_event: PipelineStageEvent | None = None,
        finalize_event: PipelineStageEvent | None = None,
        finalize_after_stages: list[StageName] | None = None,
    ) -> CompletionResult:
        key = _stage_key(event.run_id, event.stage_name, event.shard_index)
        async with self._lock:
            record = self._records.setdefault(
                key,
                {
                    "run_id": event.run_id,
                    "stage": event.stage_name.value,
                    "shard_index": event.shard_index,
                    "shard_count": event.shard_count,
                },
            )
            record.update(
                {
                    "status": "done",
                    "attempt": event.attempt,
                    "artifacts": [_artifact_to_json(item) for item in artifacts],
                    "completed_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                    "error": None,
                }
            )
            stage_rows = [
                item
                for (run_id, stage, _), item in self._records.items()
                if run_id == event.run_id and stage == event.stage_name.value
            ]
            done_count = sum(item.get("status") == "done" for item in stage_rows)
            shard_count = max(
                [event.shard_count]
                + [int(item.get("shard_count", 1)) for item in stage_rows]
            )
            all_shards_done = done_count >= shard_count
            outgoing_events = list(next_events)
            if join_next_event and all_shards_done:
                outputs = [
                    _artifact_from_json(artifact)
                    for row in sorted(stage_rows, key=lambda item: item["shard_index"])
                    if row.get("status") == "done"
                    for artifact in row.get("artifacts", [])
                ]
                metadata = dict(join_next_event.metadata)
                metadata["input_artifact_uris"] = [
                    artifact.uri for artifact in outputs if artifact.kind != "manifest"
                ]
                metadata["input_manifest_uris"] = [
                    artifact.uri for artifact in outputs if artifact.kind == "manifest"
                ]
                outgoing_events.append(
                    join_next_event.model_copy(update={"metadata": metadata})
                )
            if finalize_event and finalize_after_stages:
                stages_complete = True
                for stage in finalize_after_stages:
                    rows = [
                        item
                        for (run_id, stage_value, _), item in self._records.items()
                        if run_id == event.run_id
                        and stage_value == StageName(stage).value
                    ]
                    expected = max(
                        (int(item.get("shard_count", 1)) for item in rows),
                        default=0,
                    )
                    completed = sum(item.get("status") == "done" for item in rows)
                    if not expected or completed < expected:
                        stages_complete = False
                        break
                if stages_complete:
                    outgoing_events.append(finalize_event)
            if outgoing_events and not event.fast_path:
                self._enqueue_events_locked(outgoing_events)

        return CompletionResult(
            all_shards_done, done_count, shard_count, outgoing_events
        )

    def _enqueue_events_locked(self, events: list[PipelineStageEvent]) -> None:
        for event in events:
            self._outbox.setdefault(
                event.event_id,
                {"event": event, "status": "pending", "attempts": 0},
            )

    async def enqueue_events(self, events: list[PipelineStageEvent]) -> None:
        async with self._lock:
            self._enqueue_events_locked(events)

    async def publish_outbox(
        self,
        publisher: OutboxPublisher,
        *,
        event_ids: list[str] | None = None,
        limit: int = 100,
    ) -> int:
        async with self._lock:
            selected = [
                (event_id, row)
                for event_id, row in self._outbox.items()
                if row["status"] == "pending"
                and (event_ids is None or event_id in event_ids)
                and (
                    row["event"].generation_id is None
                    or self._generation_manifest_snapshots.get(
                        (row["event"].workspace_id, row["event"].generation_id),
                        (False, None),
                    )[0]
                )
            ][: max(0, limit)]
            for _, row in selected:
                row["status"] = "publishing"
                row["attempts"] += 1

        published = 0
        for event_id, row in selected:
            try:
                await publisher(row["event"])
            except asyncio.CancelledError:
                async with self._lock:
                    row["status"] = "pending"
                raise
            except Exception:
                async with self._lock:
                    row["status"] = "pending"
                logger.exception(
                    "[Pipeline] Outbox publish failed event_id=%s",
                    event_id,
                )
                continue
            async with self._lock:
                row["status"] = "published"
            published += 1
        return published

    async def replay_failed_stage(
        self,
        *,
        run_id: str,
        stage: StageName | str,
        shard_index: int,
        workspace_id: str,
        reason: str,
        requested_by: str,
    ) -> StageReplayResult:
        key = _stage_key(run_id, stage, shard_index)
        async with self._lock:
            row = self._records.get(key)
            if row is None or row.get("workspace_id") != workspace_id:
                raise StageRunNotFoundError("Pipeline stage run was not found")
            if row.get("status") != "failed":
                raise StageNotReplayableError(
                    f"Pipeline stage run is {row.get('status')}, not failed"
                )
            attempt = int(row.get("attempt", 0))
            max_attempts = max_stage_retry_attempts()
            if attempt >= max_attempts:
                raise StageRetryBudgetExhausted(
                    f"Retry budget exhausted at attempt={attempt}/{max_attempts}"
                )
            event = PipelineStageEvent.model_validate(row["event"])
            payload = event.model_dump(mode="json")
            payload.pop("event_id", None)
            payload.update(
                {
                    "attempt": max(attempt + 1, event.attempt + 1),
                    "status": "pending",
                    "created_at": utc_now_iso(),
                }
            )
            metadata = dict(payload.get("metadata") or {})
            metadata["operator_replay"] = {
                "requested_by": requested_by,
                "reason": reason,
                "requested_at": utc_now_iso(),
            }
            payload["metadata"] = metadata
            replay_event = PipelineStageEvent.model_validate(payload)
            row.update(
                {
                    "status": "pending",
                    "event": replay_event.model_dump(mode="json"),
                    "updated_at": utc_now_iso(),
                }
            )
            if event.generation_id is not None:
                manifest_entry = self._generation_manifest.get(
                    (event.workspace_id, event.generation_id, event.doc_id)
                )
                if manifest_entry is not None:
                    manifest_entry["status"] = "pending"
                    manifest_entry["error"] = None
            self._enqueue_events_locked([replay_event])
            return StageReplayResult(event=replay_event, status="queued")

    async def fail(self, event: PipelineStageEvent, error: PipelineError) -> None:
        key = _stage_key(event.run_id, event.stage_name, event.shard_index)
        async with self._lock:
            record = self._records.setdefault(
                key,
                {
                    "run_id": event.run_id,
                    "stage": event.stage_name.value,
                    "shard_index": event.shard_index,
                    "shard_count": event.shard_count,
                },
            )
            record.update(
                {
                    "status": "failed",
                    "attempt": max(record.get("attempt", 0), event.attempt),
                    "error": error.model_dump(mode="json"),
                    "completed_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                }
            )
            if event.generation_id is not None:
                manifest_entry = self._generation_manifest.get(
                    (event.workspace_id, event.generation_id, event.doc_id)
                )
                if manifest_entry is not None:
                    manifest_entry["status"] = "failed"
                    manifest_entry["error"] = error.model_dump(mode="json")

    async def get_stage_outputs(
        self, run_id: str, stage: StageName | str
    ) -> list[ArtifactRef]:
        stage_value = StageName(stage).value
        async with self._lock:
            rows = [
                item
                for (rid, stg, _), item in self._records.items()
                if rid == run_id and stg == stage_value and item.get("status") == "done"
            ]
            rows.sort(key=lambda item: item.get("shard_index", 0))
            artifacts = []
            for row in rows:
                artifacts.extend(_artifact_from_json(a) for a in row.get("artifacts", []))
            return artifacts

    async def are_stages_complete(
        self, run_id: str, stages: list[StageName | str]
    ) -> bool:
        async with self._lock:
            for stage in stages:
                stage_value = StageName(stage).value
                rows = [
                    item
                    for (rid, stg, _), item in self._records.items()
                    if rid == run_id and stg == stage_value
                ]
                if not rows:
                    return False
                shard_count = max(int(row.get("shard_count", 1)) for row in rows)
                done_count = sum(1 for row in rows if row.get("status") == "done")
                if done_count < shard_count:
                    return False
            return True

    async def has_any_stage_records(
        self, run_id: str, stages: list[StageName | str]
    ) -> bool:
        stage_values = {StageName(stage).value for stage in stages}
        async with self._lock:
            return any(
                rid == run_id and stg in stage_values
                for (rid, stg, _), _item in self._records.items()
            )


class PostgresPipelineCoordinator(PipelineCoordinator):
    def __init__(self, table_name: str | None = None, auto_create: bool | None = None):
        self.table_name = table_name or os.getenv(
            "PIPELINE_COORDINATOR_TABLE", "lightrag_pipeline_stage_runs"
        )
        self.outbox_table_name = f"{self.table_name}_outbox"
        self.generations_table_name = f"{self.table_name}_workspace_generations"
        self.generation_manifest_table_name = (
            f"{self.table_name}_workspace_generation_manifest"
        )
        self.workspace_state_table_name = f"{self.table_name}_workspace_state"
        self.write_leases_table_name = f"{self.table_name}_workspace_write_leases"
        self.auto_create = (
            os.getenv("PIPELINE_COORDINATOR_AUTO_CREATE", "true").lower()
            in {"1", "true", "yes", "on"}
            if auto_create is None
            else auto_create
        )
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self.running_stale_seconds = _env_int(
            "PIPELINE_STAGE_RUNNING_STALE_SECONDS", 10800
        )
        self._db_instance = None
        self._db_lock = asyncio.Lock()

    def _running_stale_seconds_for(self, stage: StageName | str) -> int:
        return _stage_int_env(
            "PIPELINE_STAGE_RUNNING_STALE_SECONDS",
            stage,
            self.running_stale_seconds,
        )

    async def _run_with_connect_retry(self, label: str, operation):
        attempts = max(1, int(os.getenv("PIPELINE_COORDINATOR_DB_RETRY_ATTEMPTS", "6")))
        delay = max(1, int(os.getenv("PIPELINE_COORDINATOR_DB_RETRY_DELAY_SECONDS", "2")))
        for attempt in range(1, attempts + 1):
            try:
                return await operation()
            except Exception as exc:
                if attempt >= attempts or not _is_transient_postgres_connect_error(exc):
                    raise
                sleep_seconds = delay * attempt
                logger.warning(
                    "[Pipeline] Transient coordinator DB error during %s; "
                    "retrying attempt=%s/%s sleep_seconds=%s error=%s",
                    label,
                    attempt,
                    attempts,
                    sleep_seconds,
                    exc,
                )
                await asyncio.sleep(sleep_seconds)

    async def _db(self):
        from lightrag.kg.postgres_impl import ClientManager

        async with self._db_lock:
            if self._db_instance is None:
                self._db_instance = await ClientManager.get_client()

        db = self._db_instance
        
        # In a Celery worker, the event loop might have changed.
        # db._ensure_pool() handles this and reinitializes the pool safely.
        await db._ensure_pool()
        
        if db.pool is None:
            raise RuntimeError("Postgres pool is not initialized even after _ensure_pool()")
        return db

    async def finalize(self) -> None:
        if self._db_instance is None:
            return

        from lightrag.kg.postgres_impl import ClientManager

        async with self._db_lock:
            if self._db_instance is not None:
                try:
                    await ClientManager.release_client(self._db_instance)
                finally:
                    self._db_instance = None

    async def _ensure_schema(self) -> None:
        if self._initialized or not self.auto_create:
            return
        async with self._init_lock:
            if self._initialized:
                return
            db = await self._db()
            sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                shard_count INTEGER NOT NULL,
                workspace_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 1,
                event_json JSONB,
                artifacts_json JSONB,
                error_json JSONB,
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (run_id, stage, shard_index)
            )
            """
            await db.execute(sql)
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.outbox_table_name} (
                    event_id TEXT PRIMARY KEY,
                    event_json JSONB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    lease_until TIMESTAMPTZ,
                    last_error TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    published_at TIMESTAMPTZ
                )
                """
            )
            await db.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.outbox_table_name}_pending
                ON {self.outbox_table_name} (available_at, created_at)
                WHERE status IN ('pending', 'publishing')
                """
            )
            # Performance indexes for stage_runs pagination, order, and status queries
            indexes = [
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_status ON {self.table_name} (run_id, stage, status);",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_order ON {self.table_name} (updated_at DESC, started_at DESC);",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_ws_order ON {self.table_name} (workspace_id, updated_at DESC);",
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_doc_type ON {self.table_name} ((event_json->>'doc_type'));",
            ]
            for idx_stmt in indexes:
                try:
                    await db.execute(idx_stmt)
                except Exception:
                    pass
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.generations_table_name} (
                    workspace_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL,
                    storage_workspace_id UUID NOT NULL UNIQUE,
                    build_run_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('building', 'ready', 'active', 'retained', 'failed')
                    ),
                    error_json JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    ready_at TIMESTAMPTZ,
                    activated_at TIMESTAMPTZ,
                    retain_until TIMESTAMPTZ,
                    snapshot_complete BOOLEAN NOT NULL DEFAULT FALSE,
                    expected_documents INTEGER,
                    PRIMARY KEY (workspace_id, generation_id)
                )
                """
            )
            await db.execute(
                f"""
                ALTER TABLE {self.generations_table_name}
                ADD COLUMN IF NOT EXISTS snapshot_complete BOOLEAN NOT NULL DEFAULT FALSE
                """
            )
            await db.execute(
                f"""
                ALTER TABLE {self.generations_table_name}
                ADD COLUMN IF NOT EXISTS expected_documents INTEGER
                """
            )
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.generation_manifest_table_name} (
                    workspace_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    event_id TEXT NOT NULL UNIQUE,
                    event_json JSONB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
                    error_json JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (workspace_id, generation_id, doc_id),
                    FOREIGN KEY (workspace_id, generation_id)
                        REFERENCES {self.generations_table_name}
                        (workspace_id, generation_id)
                )
                """
            )
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.workspace_state_table_name} (
                    workspace_id TEXT PRIMARY KEY,
                    active_generation_id TEXT,
                    write_lock_generation_id TEXT,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    FOREIGN KEY (workspace_id, active_generation_id)
                        REFERENCES {self.generations_table_name} (workspace_id, generation_id),
                    FOREIGN KEY (workspace_id, write_lock_generation_id)
                        REFERENCES {self.generations_table_name} (workspace_id, generation_id)
                )
                """
            )
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.write_leases_table_name} (
                    lease_id UUID PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await db.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_write_lease_expiry
                ON {self.write_leases_table_name} (workspace_id, expires_at)
                """
            )
            self._initialized = True

    @staticmethod
    def _workspace_generation_from_row(row: Any) -> WorkspaceGeneration:
        error = row["error_json"]
        if isinstance(error, str):
            error = json.loads(error)
        return WorkspaceGeneration(
            workspace_id=row["workspace_id"],
            generation_id=row["generation_id"],
            storage_workspace_id=str(row["storage_workspace_id"]),
            build_run_id=row["build_run_id"],
            status=WorkspaceGenerationStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            ready_at=row["ready_at"],
            activated_at=row["activated_at"],
            retain_until=row["retain_until"],
            error=error,
        )

    async def acquire_workspace_write_lease(
        self, workspace_id: str, lease_seconds: int = 600
    ) -> WorkspaceWriteLease:
        if not workspace_id.strip() or lease_seconds <= 0:
            raise ValueError("workspace_id and lease_seconds must be valid")
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"""
                    INSERT INTO {self.workspace_state_table_name} (workspace_id)
                    VALUES ($1)
                    ON CONFLICT (workspace_id) DO NOTHING
                    """,
                    workspace_id,
                )
                state = await conn.fetchrow(
                    f"""
                    SELECT write_lock_generation_id
                    FROM {self.workspace_state_table_name}
                    WHERE workspace_id=$1
                    FOR UPDATE
                    """,
                    workspace_id,
                )
                if state["write_lock_generation_id"]:
                    raise WorkspaceGenerationConflictError(
                        f"Workspace is locked for generation operation: {workspace_id}"
                    )
                await conn.execute(
                    f"""
                    DELETE FROM {self.write_leases_table_name}
                    WHERE workspace_id=$1 AND expires_at <= NOW()
                    """,
                    workspace_id,
                )
                row = await conn.fetchrow(
                    f"""
                    INSERT INTO {self.write_leases_table_name}
                        (lease_id, workspace_id, expires_at)
                    VALUES ($1::uuid, $2, NOW() + ($3 * INTERVAL '1 second'))
                    RETURNING lease_id, workspace_id, expires_at
                    """,
                    str(uuid4()),
                    workspace_id,
                    lease_seconds,
                )
                return WorkspaceWriteLease(
                    lease_id=str(row["lease_id"]),
                    workspace_id=row["workspace_id"],
                    expires_at=row["expires_at"],
                )

    async def renew_workspace_write_lease(
        self, lease: WorkspaceWriteLease, lease_seconds: int = 600
    ) -> WorkspaceWriteLease:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                state = await conn.fetchrow(
                    f"""
                    SELECT write_lock_generation_id
                    FROM {self.workspace_state_table_name}
                    WHERE workspace_id=$1
                    FOR UPDATE
                    """,
                    lease.workspace_id,
                )
                if state is None or state["write_lock_generation_id"]:
                    raise WorkspaceGenerationConflictError(
                        f"Workspace cannot renew write lease: {lease.workspace_id}"
                    )
                row = await conn.fetchrow(
                    f"""
                    UPDATE {self.write_leases_table_name}
                    SET expires_at=NOW() + ($3 * INTERVAL '1 second')
                    WHERE lease_id=$1::uuid AND workspace_id=$2 AND expires_at > NOW()
                    RETURNING lease_id, workspace_id, expires_at
                    """,
                    lease.lease_id,
                    lease.workspace_id,
                    lease_seconds,
                )
                if row is None:
                    raise WorkspaceGenerationConflictError(
                        f"Workspace write lease expired: {lease.lease_id}"
                    )
                return WorkspaceWriteLease(
                    lease_id=str(row["lease_id"]),
                    workspace_id=row["workspace_id"],
                    expires_at=row["expires_at"],
                )

    async def release_workspace_write_lease(
        self, lease: WorkspaceWriteLease
    ) -> None:
        await self._ensure_schema()
        db = await self._db()
        await db.execute(
            f"""
            DELETE FROM {self.write_leases_table_name}
            WHERE lease_id=$1::uuid AND workspace_id=$2
            """,
            lease.lease_id,
            lease.workspace_id,
        )

    async def begin_workspace_generation(
        self, workspace_id: str, build_run_id: str, generation_id: str | None = None
    ) -> WorkspaceGeneration:
        if not workspace_id.strip() or not build_run_id.strip():
            raise ValueError("workspace_id and build_run_id must be non-empty")
        generation_id = generation_id or str(uuid4())
        storage_workspace_id = str(uuid4())
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"""
                    INSERT INTO {self.workspace_state_table_name} (workspace_id)
                    VALUES ($1)
                    ON CONFLICT (workspace_id) DO NOTHING
                    """,
                    workspace_id,
                )
                state = await conn.fetchrow(
                    f"""
                    SELECT write_lock_generation_id
                    FROM {self.workspace_state_table_name}
                    WHERE workspace_id=$1
                    FOR UPDATE
                    """,
                    workspace_id,
                )
                if state["write_lock_generation_id"]:
                    locked_id = state["write_lock_generation_id"]
                    existing = await conn.fetchrow(
                        f"""
                        SELECT *
                        FROM {self.generations_table_name}
                        WHERE workspace_id=$1 AND generation_id=$2
                        """,
                        workspace_id,
                        locked_id,
                    )
                    if (
                        existing
                        and locked_id == generation_id
                        and existing["build_run_id"] == build_run_id
                        and existing["status"] == WorkspaceGenerationStatus.BUILDING.value
                    ):
                        return self._workspace_generation_from_row(existing)
                    raise WorkspaceGenerationConflictError(
                        f"Workspace already has a generation build in progress: {workspace_id}"
                    )

                await conn.execute(
                    f"""
                    DELETE FROM {self.write_leases_table_name}
                    WHERE workspace_id=$1 AND expires_at <= NOW()
                    """,
                    workspace_id,
                )
                active_write = await conn.fetchval(
                    f"""
                    SELECT EXISTS (
                        SELECT 1
                        FROM {self.write_leases_table_name}
                        WHERE workspace_id=$1 AND expires_at > NOW()
                    )
                    """,
                    workspace_id,
                )
                if active_write:
                    raise WorkspaceGenerationConflictError(
                        f"Workspace has admitted writes in progress: {workspace_id}"
                    )

                running_stage = await conn.fetchval(
                    f"""
                    SELECT EXISTS (
                        SELECT 1
                        FROM {self.table_name}
                        WHERE workspace_id=$1 AND status='running'
                    )
                    """,
                    workspace_id,
                )
                if running_stage:
                    raise WorkspaceGenerationConflictError(
                        f"Workspace has pipeline stages in progress: {workspace_id}"
                    )

                existing = await conn.fetchrow(
                    f"""
                    SELECT 1
                    FROM {self.generations_table_name}
                    WHERE workspace_id=$1 AND generation_id=$2
                    """,
                    workspace_id,
                    generation_id,
                )
                if existing:
                    raise WorkspaceGenerationConflictError(
                        f"Generation already exists: workspace_id={workspace_id}, generation_id={generation_id}"
                    )

                row = await conn.fetchrow(
                    f"""
                    INSERT INTO {self.generations_table_name}
                        (workspace_id, generation_id, storage_workspace_id,
                         build_run_id, status)
                    VALUES ($1, $2, $3::uuid, $4, 'building')
                    RETURNING *
                    """,
                    workspace_id,
                    generation_id,
                    storage_workspace_id,
                    build_run_id,
                )
                await conn.execute(
                    f"""
                    UPDATE {self.workspace_state_table_name}
                    SET write_lock_generation_id=$2, updated_at=NOW()
                    WHERE workspace_id=$1
                    """,
                    workspace_id,
                    generation_id,
                )
                return self._workspace_generation_from_row(row)

    async def _get_locked_building_generation(
        self, conn, workspace_id: str, generation_id: str
    ) -> Any:
        state = await conn.fetchrow(
            f"""
            SELECT write_lock_generation_id
            FROM {self.workspace_state_table_name}
            WHERE workspace_id=$1
            FOR UPDATE
            """,
            workspace_id,
        )
        generation = await conn.fetchrow(
            f"""
            SELECT *
            FROM {self.generations_table_name}
            WHERE workspace_id=$1 AND generation_id=$2
            FOR UPDATE
            """,
            workspace_id,
            generation_id,
        )
        if generation is None:
            raise WorkspaceGenerationNotFoundError(
                f"Generation not found: workspace_id={workspace_id}, "
                f"generation_id={generation_id}"
            )
        if (
            state is None
            or state["write_lock_generation_id"] != generation_id
            or generation["status"] != WorkspaceGenerationStatus.BUILDING.value
        ):
            raise WorkspaceGenerationTransitionError(
                "Manifest operations require a locked building generation"
            )
        return generation

    async def add_workspace_generation_manifest_events(
        self,
        workspace_id: str,
        generation_id: str,
        events: list[PipelineStageEvent],
    ) -> int:
        await self._ensure_schema()
        db = await self._db()
        added = 0
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                generation = await self._get_locked_building_generation(
                    conn, workspace_id, generation_id
                )
                generation_obj = self._workspace_generation_from_row(generation)
                self._validate_generation_manifest_events(
                    workspace_id, generation_obj, events
                )
                if generation["snapshot_complete"] and events:
                    new_document = await conn.fetchval(
                        f"""
                        SELECT EXISTS (
                            SELECT 1
                            FROM unnest($3::text[]) AS requested(doc_id)
                            WHERE NOT EXISTS (
                                SELECT 1
                                FROM {self.generation_manifest_table_name} AS manifest
                                WHERE manifest.workspace_id=$1
                                  AND manifest.generation_id=$2
                                  AND manifest.doc_id=requested.doc_id
                            )
                        )
                        """,
                        workspace_id,
                        generation_id,
                        [event.doc_id for event in events],
                    )
                    if new_document:
                        raise WorkspaceGenerationTransitionError(
                            "Cannot add documents after the generation manifest "
                            "snapshot has completed"
                        )
                for event in events:
                    event_json = event.model_dump_json()
                    result = await conn.execute(
                        f"""
                        INSERT INTO {self.generation_manifest_table_name}
                            (workspace_id, generation_id, doc_id, event_id, event_json)
                        VALUES ($1, $2, $3, $4, $5::jsonb)
                        ON CONFLICT (workspace_id, generation_id, doc_id)
                        DO NOTHING
                        """,
                        workspace_id,
                        generation_id,
                        event.doc_id,
                        event.event_id,
                        event_json,
                    )
                    inserted = result.endswith(" 1")
                    existing = await conn.fetchrow(
                        f"""
                        SELECT event_json
                        FROM {self.generation_manifest_table_name}
                        WHERE workspace_id=$1 AND generation_id=$2 AND doc_id=$3
                        """,
                        workspace_id,
                        generation_id,
                        event.doc_id,
                    )
                    stored_event = existing["event_json"]
                    if isinstance(stored_event, str):
                        stored_event = json.loads(stored_event)
                    if stored_event != event.model_dump(mode="json"):
                        raise WorkspaceGenerationConflictError(
                            "Generation manifest document changed during resume: "
                            f"workspace_id={workspace_id}, "
                            f"generation_id={generation_id}, doc_id={event.doc_id}"
                        )
                    if inserted:
                        added += 1
                await self._insert_outbox_events(conn, events)
        return added

    async def complete_workspace_generation_manifest_snapshot(
        self, workspace_id: str, generation_id: str, expected_documents: int
    ) -> None:
        if expected_documents < 0:
            raise ValueError("expected_documents cannot be negative")
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                await self._get_locked_building_generation(
                    conn, workspace_id, generation_id
                )
                registered_documents = int(
                    await conn.fetchval(
                        f"""
                        SELECT COUNT(*)
                        FROM {self.generation_manifest_table_name}
                        WHERE workspace_id=$1 AND generation_id=$2
                        """,
                        workspace_id,
                        generation_id,
                    )
                    or 0
                )
                if registered_documents != expected_documents:
                    raise WorkspaceGenerationTransitionError(
                        "Manifest snapshot count does not match registered documents: "
                        f"expected={expected_documents}, registered={registered_documents}"
                    )
                await conn.execute(
                    f"""
                    UPDATE {self.generations_table_name}
                    SET snapshot_complete=TRUE, expected_documents=$3,
                        updated_at=NOW()
                    WHERE workspace_id=$1 AND generation_id=$2
                    """,
                    workspace_id,
                    generation_id,
                    expected_documents,
                )

    async def get_workspace_generation_manifest_status(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGenerationManifestStatus:
        await self._ensure_schema()
        db = await self._db()
        row = await db.query(
            f"""
            SELECT g.snapshot_complete, g.expected_documents,
                   COUNT(*) FILTER (WHERE m.status='pending') AS pending_documents,
                   COUNT(*) FILTER (WHERE m.status='running') AS running_documents,
                   COUNT(*) FILTER (WHERE m.status='completed') AS completed_documents,
                   COUNT(*) FILTER (WHERE m.status='failed') AS failed_documents
            FROM {self.generations_table_name} AS g
            LEFT JOIN {self.generation_manifest_table_name} AS m
              ON m.workspace_id=g.workspace_id
             AND m.generation_id=g.generation_id
            WHERE g.workspace_id=$1 AND g.generation_id=$2
            GROUP BY g.snapshot_complete, g.expected_documents
            """,
            [workspace_id, generation_id],
        )
        if row is None:
            raise WorkspaceGenerationNotFoundError(
                f"Generation not found: workspace_id={workspace_id}, "
                f"generation_id={generation_id}"
            )
        return WorkspaceGenerationManifestStatus(
            workspace_id=workspace_id,
            generation_id=generation_id,
            snapshot_complete=bool(row["snapshot_complete"]),
            expected_documents=row["expected_documents"],
            pending_documents=int(row["pending_documents"] or 0),
            running_documents=int(row["running_documents"] or 0),
            completed_documents=int(row["completed_documents"] or 0),
            failed_documents=int(row["failed_documents"] or 0),
        )

    async def resume_workspace_generation_manifest(
        self, workspace_id: str, generation_id: str
    ) -> int:
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                await self._get_locked_building_generation(
                    conn, workspace_id, generation_id
                )
                rows = await conn.fetch(
                    f"""
                    UPDATE {self.outbox_table_name} AS outbox
                    SET status='pending', available_at=NOW(), lease_until=NULL,
                        last_error=NULL, updated_at=NOW()
                    FROM {self.generation_manifest_table_name} AS manifest
                    WHERE manifest.workspace_id=$1
                      AND manifest.generation_id=$2
                      AND manifest.status <> 'completed'
                      AND outbox.event_json->>'workspace_id'=manifest.workspace_id
                      AND outbox.event_json->>'generation_id'=manifest.generation_id
                      AND outbox.event_json->>'doc_id'=manifest.doc_id
                      AND outbox.status='published'
                      AND NOT EXISTS (
                          SELECT 1
                          FROM {self.table_name} AS stage_run
                          WHERE stage_run.run_id=outbox.event_json->>'run_id'
                            AND stage_run.stage=outbox.event_json->>'stage'
                            AND stage_run.shard_index=(
                                outbox.event_json->>'shard_index'
                            )::INTEGER
                            AND stage_run.status='done'
                      )
                    RETURNING outbox.event_id
                    """,
                    workspace_id,
                    generation_id,
                )
                return len(rows)

    async def mark_workspace_generation_ready(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGeneration:
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                state = await conn.fetchrow(
                    f"""
                    SELECT write_lock_generation_id
                    FROM {self.workspace_state_table_name}
                    WHERE workspace_id=$1
                    FOR UPDATE
                    """,
                    workspace_id,
                )
                row = await conn.fetchrow(
                    f"""
                    SELECT *
                    FROM {self.generations_table_name}
                    WHERE workspace_id=$1 AND generation_id=$2
                    FOR UPDATE
                    """,
                    workspace_id,
                    generation_id,
                )
                if row is None:
                    raise WorkspaceGenerationNotFoundError(
                        f"Generation not found: workspace_id={workspace_id}, generation_id={generation_id}"
                    )
                status = WorkspaceGenerationStatus(row["status"])
                if status == WorkspaceGenerationStatus.READY:
                    if (
                        state is None
                        or state["write_lock_generation_id"] != generation_id
                    ):
                        raise WorkspaceGenerationTransitionError(
                            "A ready generation must still hold the workspace write lock"
                        )
                    return self._workspace_generation_from_row(row)
                if (
                    status != WorkspaceGenerationStatus.BUILDING
                    or state is None
                    or state["write_lock_generation_id"] != generation_id
                ):
                    raise WorkspaceGenerationTransitionError(
                        f"Cannot mark generation ready from status={status.value}"
                    )
                manifest_state = await conn.fetchrow(
                    f"""
                    SELECT g.snapshot_complete, g.expected_documents,
                           COUNT(m.doc_id) AS registered_documents,
                           COUNT(*) FILTER (WHERE m.status='completed')
                               AS completed_documents,
                           COUNT(*) FILTER (WHERE m.status='running')
                               AS running_documents
                    FROM {self.generations_table_name} AS g
                    LEFT JOIN {self.generation_manifest_table_name} AS m
                      ON m.workspace_id=g.workspace_id
                     AND m.generation_id=g.generation_id
                    WHERE g.workspace_id=$1 AND g.generation_id=$2
                    GROUP BY g.snapshot_complete, g.expected_documents
                    """,
                    workspace_id,
                    generation_id,
                )
                if (
                    manifest_state is None
                    or not manifest_state["snapshot_complete"]
                    or manifest_state["expected_documents"]
                    != int(manifest_state["registered_documents"] or 0)
                    or manifest_state["expected_documents"]
                    != int(manifest_state["completed_documents"] or 0)
                    or int(manifest_state["running_documents"] or 0) > 0
                ):
                    raise WorkspaceGenerationTransitionError(
                        "Cannot mark generation ready until its document manifest "
                        "snapshot is complete and every document has completed"
                    )
                running_stage = await conn.fetchval(
                    f"""
                    SELECT EXISTS (
                        SELECT 1
                        FROM {self.table_name}
                        WHERE workspace_id=$1
                          AND status='running'
                          AND event_json->>'generation_id'=$2
                    )
                    """,
                    workspace_id,
                    generation_id,
                )
                if running_stage:
                    raise WorkspaceGenerationTransitionError(
                        "Cannot mark generation ready while its pipeline stages are running"
                    )
                ready = await conn.fetchrow(
                    f"""
                    UPDATE {self.generations_table_name}
                    SET status='ready', ready_at=NOW(), updated_at=NOW()
                    WHERE workspace_id=$1 AND generation_id=$2
                    RETURNING *
                    """,
                    workspace_id,
                    generation_id,
                )
                return self._workspace_generation_from_row(ready)

    async def fail_workspace_generation(
        self, workspace_id: str, generation_id: str, error: dict[str, Any]
    ) -> WorkspaceGeneration:
        await self._ensure_schema()
        db = await self._db()
        error_json = json.dumps(error, ensure_ascii=False)
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                await conn.fetchrow(
                    f"""
                    SELECT workspace_id
                    FROM {self.workspace_state_table_name}
                    WHERE workspace_id=$1
                    FOR UPDATE
                    """,
                    workspace_id,
                )
                row = await conn.fetchrow(
                    f"""
                    SELECT *
                    FROM {self.generations_table_name}
                    WHERE workspace_id=$1 AND generation_id=$2
                    FOR UPDATE
                    """,
                    workspace_id,
                    generation_id,
                )
                if row is None:
                    raise WorkspaceGenerationNotFoundError(
                        f"Generation not found: workspace_id={workspace_id}, generation_id={generation_id}"
                    )
                status = WorkspaceGenerationStatus(row["status"])
                if status == WorkspaceGenerationStatus.FAILED:
                    await conn.execute(
                        f"""
                        UPDATE {self.workspace_state_table_name}
                        SET write_lock_generation_id=NULL, updated_at=NOW()
                        WHERE workspace_id=$1 AND write_lock_generation_id=$2
                        """,
                        workspace_id,
                        generation_id,
                    )
                    return self._workspace_generation_from_row(row)
                if status not in {
                    WorkspaceGenerationStatus.BUILDING,
                    WorkspaceGenerationStatus.READY,
                }:
                    raise WorkspaceGenerationTransitionError(
                        f"Cannot fail generation from status={status.value}"
                    )
                failed = await conn.fetchrow(
                    f"""
                    UPDATE {self.generations_table_name}
                    SET status='failed', error_json=$3::jsonb, updated_at=NOW()
                    WHERE workspace_id=$1 AND generation_id=$2
                    RETURNING *
                    """,
                    workspace_id,
                    generation_id,
                    error_json,
                )
                await conn.execute(
                    f"""
                    UPDATE {self.workspace_state_table_name}
                    SET write_lock_generation_id=NULL, updated_at=NOW()
                    WHERE workspace_id=$1 AND write_lock_generation_id=$2
                    """,
                    workspace_id,
                    generation_id,
                )
                return self._workspace_generation_from_row(failed)

    async def activate_workspace_generation(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGeneration:
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                state = await conn.fetchrow(
                    f"""
                    SELECT active_generation_id, write_lock_generation_id
                    FROM {self.workspace_state_table_name}
                    WHERE workspace_id=$1
                    FOR UPDATE
                    """,
                    workspace_id,
                )
                if state is None:
                    raise WorkspaceGenerationNotFoundError(
                        f"Workspace generation state not found: {workspace_id}"
                    )
                row = await conn.fetchrow(
                    f"""
                    SELECT *
                    FROM {self.generations_table_name}
                    WHERE workspace_id=$1 AND generation_id=$2
                    FOR UPDATE
                    """,
                    workspace_id,
                    generation_id,
                )
                if row is None:
                    raise WorkspaceGenerationNotFoundError(
                        f"Generation not found: workspace_id={workspace_id}, generation_id={generation_id}"
                    )
                status = WorkspaceGenerationStatus(row["status"])
                if (
                    status == WorkspaceGenerationStatus.ACTIVE
                    and state["active_generation_id"] == generation_id
                ):
                    return self._workspace_generation_from_row(row)
                now = datetime.now(timezone.utc)
                retained_is_valid = (
                    status == WorkspaceGenerationStatus.RETAINED
                    and row["retain_until"] is not None
                    and row["retain_until"] > now
                )
                if status != WorkspaceGenerationStatus.READY and not retained_is_valid:
                    raise WorkspaceGenerationTransitionError(
                        f"Cannot activate generation from status={status.value}"
                    )
                write_lock_id = state["write_lock_generation_id"]
                if write_lock_id != generation_id:
                    raise WorkspaceGenerationTransitionError(
                        "Workspace writes must be locked for the generation being activated"
                    )
                running_stage = await conn.fetchval(
                    f"""
                    SELECT EXISTS (
                        SELECT 1
                        FROM {self.table_name}
                        WHERE workspace_id=$1
                          AND status='running'
                          AND event_json->>'generation_id'=$2
                    )
                    """,
                    workspace_id,
                    generation_id,
                )
                if running_stage:
                    raise WorkspaceGenerationTransitionError(
                        "Cannot activate generation while its pipeline stages are running"
                    )

                old_generation_id = state["active_generation_id"]
                if old_generation_id and old_generation_id != generation_id:
                    await conn.execute(
                        f"""
                        UPDATE {self.generations_table_name}
                        SET status='retained',
                            retain_until=NOW() + INTERVAL '7 days',
                            updated_at=NOW()
                        WHERE workspace_id=$1 AND generation_id=$2 AND status='active'
                        """,
                        workspace_id,
                        old_generation_id,
                    )
                activated = await conn.fetchrow(
                    f"""
                    UPDATE {self.generations_table_name}
                    SET status='active', activated_at=NOW(),
                        retain_until=NULL, updated_at=NOW()
                    WHERE workspace_id=$1 AND generation_id=$2
                    RETURNING *
                    """,
                    workspace_id,
                    generation_id,
                )
                await conn.execute(
                    f"""
                    UPDATE {self.workspace_state_table_name}
                    SET active_generation_id=$2,
                        write_lock_generation_id=NULL, updated_at=NOW()
                    WHERE workspace_id=$1
                    """,
                    workspace_id,
                    generation_id,
                )
                return self._workspace_generation_from_row(activated)

    async def lock_workspace_for_generation(
        self, workspace_id: str, generation_id: str
    ) -> None:
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                state = await conn.fetchrow(
                    f"""
                    SELECT write_lock_generation_id
                    FROM {self.workspace_state_table_name}
                    WHERE workspace_id=$1
                    FOR UPDATE
                    """,
                    workspace_id,
                )
                if state is None:
                    raise WorkspaceGenerationNotFoundError(
                        f"Workspace generation state not found: {workspace_id}"
                    )
                row = await conn.fetchrow(
                    f"""
                    SELECT status, retain_until
                    FROM {self.generations_table_name}
                    WHERE workspace_id=$1 AND generation_id=$2
                    FOR UPDATE
                    """,
                    workspace_id,
                    generation_id,
                )
                if row is None:
                    raise WorkspaceGenerationNotFoundError(
                        f"Generation not found: workspace_id={workspace_id}, generation_id={generation_id}"
                    )
                status = WorkspaceGenerationStatus(row["status"])
                retained_is_valid = (
                    status == WorkspaceGenerationStatus.RETAINED
                    and row["retain_until"] is not None
                    and row["retain_until"] > datetime.now(timezone.utc)
                )
                if status not in {
                    WorkspaceGenerationStatus.BUILDING,
                    WorkspaceGenerationStatus.READY,
                } and not retained_is_valid:
                    raise WorkspaceGenerationTransitionError(
                        f"Cannot lock workspace for generation status={status.value}"
                    )
                locked_id = state["write_lock_generation_id"]
                if locked_id and locked_id != generation_id:
                    raise WorkspaceGenerationConflictError(
                        f"Workspace already has a generation operation in progress: {workspace_id}"
                    )
                await conn.execute(
                    f"""
                    UPDATE {self.workspace_state_table_name}
                    SET write_lock_generation_id=$2, updated_at=NOW()
                    WHERE workspace_id=$1
                    """,
                    workspace_id,
                    generation_id,
                )

    async def get_workspace_generation(
        self, workspace_id: str, generation_id: str
    ) -> WorkspaceGeneration | None:
        await self._ensure_schema()
        db = await self._db()
        row = await db.query(
            f"""
            SELECT *
            FROM {self.generations_table_name}
            WHERE workspace_id=$1 AND generation_id=$2
            """,
            [workspace_id, generation_id],
        )
        return self._workspace_generation_from_row(row) if row else None

    async def get_active_workspace_generation(self, workspace_id: str) -> str | None:
        await self._ensure_schema()
        db = await self._db()
        row = await db.query(
            f"""
            SELECT active_generation_id
            FROM {self.workspace_state_table_name}
            WHERE workspace_id=$1
            """,
            [workspace_id],
        )
        return row.get("active_generation_id") if row else None

    async def is_workspace_write_locked(self, workspace_id: str) -> bool:
        await self._ensure_schema()
        db = await self._db()
        row = await db.query(
            f"""
            SELECT write_lock_generation_id IS NOT NULL AS is_locked
            FROM {self.workspace_state_table_name}
            WHERE workspace_id=$1
            """,
            [workspace_id],
        )
        return bool(row and row.get("is_locked"))

    async def list_expired_retained_generations(
        self, *, now: datetime | None = None
    ) -> list[WorkspaceGeneration]:
        await self._ensure_schema()
        db = await self._db()
        ref_now = now or datetime.now(timezone.utc)
        rows = await db.query(
            f"""
            SELECT g.*
            FROM {self.generations_table_name} g
            LEFT JOIN {self.workspace_state_table_name} s
                ON s.workspace_id = g.workspace_id
            WHERE g.status = 'retained'
              AND g.retain_until IS NOT NULL
              AND g.retain_until <= $1
              AND COALESCE(s.active_generation_id, '') <> g.generation_id
              AND COALESCE(s.write_lock_generation_id, '') <> g.generation_id
            ORDER BY g.retain_until ASC
            """,
            [ref_now],
            multirows=True,
        )
        return [self._workspace_generation_from_row(row) for row in (rows or [])]

    async def mark_workspace_generation_cleaned(
        self, workspace_id: str, generation_id: str
    ) -> bool:
        await self._ensure_schema()
        db = await self._db()
        row = await db.query(
            f"""
            UPDATE {self.generations_table_name} g
            SET retain_until = NULL,
                updated_at = NOW()
            FROM {self.workspace_state_table_name} s
            WHERE g.workspace_id = $1
              AND g.generation_id = $2
              AND g.status = 'retained'
              AND g.retain_until IS NOT NULL
              AND s.workspace_id = g.workspace_id
              AND COALESCE(s.active_generation_id, '') <> g.generation_id
              AND COALESCE(s.write_lock_generation_id, '') <> g.generation_id
            RETURNING g.generation_id
            """,
            [workspace_id, generation_id],
        )
        return bool(row)

    async def _claim_once(self, event: PipelineStageEvent) -> ClaimResult:
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"""
                    INSERT INTO {self.workspace_state_table_name} (workspace_id)
                    VALUES ($1)
                    ON CONFLICT (workspace_id) DO NOTHING
                    """,
                    event.workspace_id,
                )
                workspace_state = await conn.fetchrow(
                    f"""
                    SELECT write_lock_generation_id
                    FROM {self.workspace_state_table_name}
                    WHERE workspace_id=$1
                    FOR UPDATE
                    """,
                    event.workspace_id,
                )
                locked_generation_id = workspace_state["write_lock_generation_id"]
                if locked_generation_id:
                    generation_row = await conn.fetchrow(
                        f"""
                        SELECT status, storage_workspace_id, snapshot_complete
                        FROM {self.generations_table_name}
                        WHERE workspace_id=$1 AND generation_id=$2
                        """,
                        event.workspace_id,
                        locked_generation_id,
                    )
                    if (
                        event.generation_id != locked_generation_id
                        or generation_row is None
                        or generation_row["status"] != WorkspaceGenerationStatus.BUILDING.value
                        or not generation_row["snapshot_complete"]
                        or str(generation_row["storage_workspace_id"])
                        != event.storage_workspace_id
                    ):
                        return ClaimResult(ClaimStatus.WORKSPACE_WRITE_LOCKED)
                elif (
                    event.generation_id is not None
                    or event.storage_workspace_id is not None
                ):
                    return ClaimResult(ClaimStatus.WORKSPACE_WRITE_LOCKED)

                manifest_row = None
                if event.generation_id is not None:
                    manifest_row = await conn.fetchrow(
                        f"""
                        SELECT status
                        FROM {self.generation_manifest_table_name}
                        WHERE workspace_id=$1 AND generation_id=$2 AND doc_id=$3
                        FOR UPDATE
                        """,
                        event.workspace_id,
                        event.generation_id,
                        event.doc_id,
                    )
                    if manifest_row is None:
                        return ClaimResult(ClaimStatus.WORKSPACE_WRITE_LOCKED)

                row = await conn.fetchrow(
                    f"""
                    SELECT status, attempt,
                           EXTRACT(EPOCH FROM (NOW() - COALESCE(updated_at, started_at, NOW()))) AS age_seconds
                    FROM {self.table_name}
                    WHERE run_id=$1 AND stage=$2 AND shard_index=$3
                    FOR UPDATE
                    """,
                    event.run_id,
                    event.stage_name.value,
                    event.shard_index,
                )
                if (
                    manifest_row is not None
                    and manifest_row["status"] == "completed"
                    and (row is None or row["status"] != "done")
                ):
                    return ClaimResult(ClaimStatus.WORKSPACE_WRITE_LOCKED)
                if row:
                    status = row["status"]
                    if status == "done":
                        return ClaimResult(ClaimStatus.ALREADY_COMPLETED)
                    if status == "running":
                        running_stale_seconds = self._running_stale_seconds_for(
                            event.stage_name
                        )
                        age_seconds = float(row["age_seconds"] or 0)
                        if (
                            running_stale_seconds <= 0
                            or age_seconds < running_stale_seconds
                        ):
                            return ClaimResult(ClaimStatus.ALREADY_RUNNING)
                        logger.warning(
                            "[Pipeline] Reclaiming stale running stage run_id=%s stage=%s shard=%s age_seconds=%.1f stale_seconds=%s",
                            event.run_id,
                            event.stage_name.value,
                            event.shard_index,
                            age_seconds,
                            running_stale_seconds,
                        )
                    if status == "failed":
                        recorded_attempt = int(row["attempt"])
                        max_attempts = max_stage_retry_attempts()
                        if recorded_attempt >= max_attempts:
                            return ClaimResult(
                                ClaimStatus.FAILED_ATTEMPT_RECORDED,
                                f"max retries ({max_attempts}) exhausted at attempt={recorded_attempt}",
                            )
                        logger.info(
                            "[Pipeline] Auto-retrying failed stage run_id=%s stage=%s shard=%s attempt=%s/%s",
                            event.run_id,
                            event.stage_name.value,
                            event.shard_index,
                            recorded_attempt + 1,
                            max_attempts,
                        )
                        # Fall through to UPDATE below
                    if not await self._mark_generation_manifest_running(conn, event):
                        return ClaimResult(ClaimStatus.WORKSPACE_WRITE_LOCKED)
                    await conn.execute(
                        f"""
                        UPDATE {self.table_name}
                        SET status='running',
                            attempt=GREATEST(attempt + 1, $4),
                            event_json=$5::jsonb,
                            error_json=NULL, started_at=NOW(), updated_at=NOW()
                        WHERE run_id=$1 AND stage=$2 AND shard_index=$3
                        """,
                        event.run_id,
                        event.stage_name.value,
                        event.shard_index,
                        event.attempt,
                        json.dumps(event.model_dump(mode="json"), ensure_ascii=False),
                    )
                    return ClaimResult(ClaimStatus.ACQUIRED)

                if not await self._mark_generation_manifest_running(conn, event):
                    return ClaimResult(ClaimStatus.WORKSPACE_WRITE_LOCKED)
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name} AS current
                        (run_id, stage, shard_index, shard_count, workspace_id, doc_id,
                         status, attempt, event_json, artifacts_json, started_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, 'running', $7, $8::jsonb,
                            '[]'::jsonb, NOW(), NOW())
                    ON CONFLICT (workspace_id, doc_id, stage, shard_index)
                    DO UPDATE SET
                        run_id = EXCLUDED.run_id,
                        shard_count = EXCLUDED.shard_count,
                        status = 'running',
                        attempt = EXCLUDED.attempt,
                        event_json = EXCLUDED.event_json,
                        started_at = NOW(),
                        updated_at = NOW()
                    """,
                    event.run_id,
                    event.stage_name.value,
                    event.shard_index,
                    event.shard_count,
                    event.workspace_id,
                    event.doc_id,
                    event.attempt,
                    json.dumps(event.model_dump(mode="json"), ensure_ascii=False),
                )
                return ClaimResult(ClaimStatus.ACQUIRED)

    async def _mark_generation_manifest_running(
        self, conn: Any, event: PipelineStageEvent
    ) -> bool:
        if event.generation_id is None:
            return True
        row = await conn.fetchrow(
            f"""
            UPDATE {self.generation_manifest_table_name}
            SET status='running', error_json=NULL, updated_at=NOW()
            WHERE workspace_id=$1 AND generation_id=$2 AND doc_id=$3
              AND status <> 'completed'
            RETURNING doc_id
            """,
            event.workspace_id,
            event.generation_id,
            event.doc_id,
        )
        return row is not None

    async def _mark_generation_manifest_outcome(
        self,
        conn: Any,
        event: PipelineStageEvent,
        *,
        status: str,
        error_json: str | None = None,
    ) -> None:
        if event.generation_id is None:
            return
        row = await conn.fetchrow(
            f"""
            UPDATE {self.generation_manifest_table_name}
            SET status=$4, error_json=$5::jsonb, updated_at=NOW()
            WHERE workspace_id=$1 AND generation_id=$2 AND doc_id=$3
            RETURNING doc_id
            """,
            event.workspace_id,
            event.generation_id,
            event.doc_id,
            status,
            error_json,
        )
        if row is None:
            raise WorkspaceGenerationTransitionError(
                "Generation pipeline event has no matching manifest entry: "
                f"generation_id={event.generation_id}, doc_id={event.doc_id}"
            )

    async def claim(self, event: PipelineStageEvent) -> ClaimResult:
        return await self._run_with_connect_retry(
            f"claim stage={event.stage_name.value}",
            lambda: self._claim_once(event),
        )

    async def _heartbeat_once(self, event: PipelineStageEvent) -> None:
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE {self.table_name}
                SET updated_at=NOW()
                WHERE run_id=$1
                  AND stage=$2
                  AND shard_index=$3
                  AND status='running'
                """,
                event.run_id,
                event.stage_name.value,
                event.shard_index,
            )

    async def heartbeat(self, event: PipelineStageEvent) -> None:
        await self._run_with_connect_retry(
            f"heartbeat stage={event.stage_name.value}",
            lambda: self._heartbeat_once(event),
        )

    async def _register_paused_once(self, event: PipelineStageEvent) -> None:
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name}
                    (run_id, stage, shard_index, shard_count, workspace_id, doc_id,
                     status, attempt, event_json, artifacts_json, started_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, 'paused', $7, $8::jsonb,
                        '[]'::jsonb, NOW(), NOW())
                ON CONFLICT (run_id, stage, shard_index)
                DO UPDATE SET status='paused', attempt=EXCLUDED.attempt,
                    event_json=EXCLUDED.event_json, updated_at=NOW()
                """,
                event.run_id,
                event.stage_name.value,
                event.shard_index,
                event.shard_count,
                event.workspace_id,
                event.doc_id,
                event.attempt,
                json.dumps(event.model_dump(mode="json"), ensure_ascii=False),
            )

    async def register_paused(self, event: PipelineStageEvent) -> None:
        await self._run_with_connect_retry(
            f"register_paused stage={event.stage_name.value}",
            lambda: self._register_paused_once(event),
        )

    async def _complete_once(
        self, event: PipelineStageEvent, artifacts: list[ArtifactRef]
    ) -> CompletionResult:
        await self._ensure_schema()
        db = await self._db()
        artifacts_json = json.dumps(
            [_artifact_to_json(artifact) for artifact in artifacts],
            ensure_ascii=False,
        )
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET status='done', artifacts_json=$4::jsonb,
                        completed_at=NOW(), updated_at=NOW(), error_json=NULL
                    WHERE run_id=$1 AND stage=$2 AND shard_index=$3
                    """,
                    event.run_id,
                    event.stage_name.value,
                    event.shard_index,
                    artifacts_json,
                )
                if (
                    event.generation_id is not None
                    and event.stage_name == StageName.FINALIZE
                ):
                    await self._mark_generation_manifest_outcome(
                        conn, event, status="completed"
                    )
                done_count = await conn.fetchval(
                    f"""
                    SELECT COUNT(*)
                    FROM {self.table_name}
                    WHERE run_id=$1 AND stage=$2 AND status='done'
                    """,
                    event.run_id,
                    event.stage_name.value,
                )
        done_count = int(done_count or 0)
        return CompletionResult(
            all_shards_done=done_count >= event.shard_count,
            done_count=done_count,
            shard_count=event.shard_count,
            next_events=[],
        )

    async def complete(
        self, event: PipelineStageEvent, artifacts: list[ArtifactRef]
    ) -> CompletionResult:
        return await self._run_with_connect_retry(
            f"complete stage={event.stage_name.value}",
            lambda: self._complete_once(event, artifacts),
        )

    async def _insert_outbox_events(self, conn, events: list[PipelineStageEvent]) -> None:
        if not events:
            return
        await conn.executemany(
            f"""
            INSERT INTO {self.outbox_table_name} (event_id, event_json)
            VALUES ($1, $2::jsonb)
            ON CONFLICT (event_id) DO NOTHING
            """,
            [
                (
                    outgoing.event_id,
                    outgoing.model_dump_json(),
                )
                for outgoing in events
            ],
        )

    async def _complete_and_enqueue_once(
        self,
        event: PipelineStageEvent,
        artifacts: list[ArtifactRef],
        *,
        next_events: list[PipelineStageEvent],
        join_next_event: PipelineStageEvent | None,
        finalize_event: PipelineStageEvent | None,
        finalize_after_stages: list[StageName] | None,
    ) -> CompletionResult:
        await self._ensure_schema()
        db = await self._db()
        artifacts_json = json.dumps(
            [_artifact_to_json(artifact) for artifact in artifacts],
            ensure_ascii=False,
        )
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1), hashtext($2))",
                    event.run_id,
                    "pipeline_stage_completion",
                )
                await conn.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET status='done', artifacts_json=$4::jsonb,
                        completed_at=NOW(), updated_at=NOW(), error_json=NULL
                    WHERE run_id=$1 AND stage=$2 AND shard_index=$3
                    """,
                    event.run_id,
                    event.stage_name.value,
                    event.shard_index,
                    artifacts_json,
                )
                if (
                    event.generation_id is not None
                    and event.stage_name == StageName.FINALIZE
                ):
                    await self._mark_generation_manifest_outcome(
                        conn, event, status="completed"
                    )
                done_count = int(
                    await conn.fetchval(
                        f"""
                        SELECT COUNT(*)
                        FROM {self.table_name}
                        WHERE run_id=$1 AND stage=$2 AND status='done'
                        """,
                        event.run_id,
                        event.stage_name.value,
                    )
                    or 0
                )
                outgoing_events = list(next_events)
                all_shards_done = done_count >= event.shard_count
                if join_next_event and all_shards_done:
                    rows = await conn.fetch(
                        f"""
                        SELECT artifacts_json
                        FROM {self.table_name}
                        WHERE run_id=$1 AND stage=$2 AND status='done'
                        ORDER BY shard_index
                        """,
                        event.run_id,
                        event.stage_name.value,
                    )
                    stage_outputs: list[ArtifactRef] = []
                    for row in rows:
                        raw = row["artifacts_json"] or []
                        if isinstance(raw, str):
                            raw = json.loads(raw)
                        stage_outputs.extend(
                            _artifact_from_json(item) for item in raw
                        )
                    metadata = dict(join_next_event.metadata)
                    metadata["input_artifact_uris"] = [
                        artifact.uri
                        for artifact in stage_outputs
                        if artifact.kind != "manifest"
                    ]
                    metadata["input_manifest_uris"] = [
                        artifact.uri
                        for artifact in stage_outputs
                        if artifact.kind == "manifest"
                    ]
                    outgoing_events.append(
                        join_next_event.model_copy(update={"metadata": metadata})
                    )
                if finalize_event and finalize_after_stages:
                    stage_values = [
                        StageName(stage).value for stage in finalize_after_stages
                    ]
                    rows = await conn.fetch(
                        f"""
                        SELECT stage,
                               COUNT(*) FILTER (WHERE status='done') AS done_count,
                               COALESCE(MAX(shard_count), 0) AS shard_count
                        FROM {self.table_name}
                        WHERE run_id=$1 AND stage = ANY($2::text[])
                        GROUP BY stage
                        """,
                        event.run_id,
                        stage_values,
                    )
                    completed_by_stage = {
                        row["stage"]: (
                            int(row["done_count"] or 0),
                            int(row["shard_count"] or 0),
                        )
                        for row in rows
                    }
                    if all(
                        stage_val in completed_by_stage
                        and completed_by_stage[stage_val][1] > 0
                        and completed_by_stage[stage_val][0]
                        >= completed_by_stage[stage_val][1]
                        for stage_val in stage_values
                    ):
                        outgoing_events.append(finalize_event)
                if outgoing_events and not event.fast_path:
                    await self._insert_outbox_events(conn, outgoing_events)

        return CompletionResult(
            all_shards_done=all_shards_done,
            done_count=done_count,
            shard_count=event.shard_count,
            next_events=outgoing_events,
        )

    async def complete_and_enqueue(
        self,
        event: PipelineStageEvent,
        artifacts: list[ArtifactRef],
        *,
        next_events: list[PipelineStageEvent],
        join_next_event: PipelineStageEvent | None = None,
        finalize_event: PipelineStageEvent | None = None,
        finalize_after_stages: list[StageName] | None = None,
    ) -> CompletionResult:
        return await self._run_with_connect_retry(
            f"complete and enqueue stage={event.stage_name.value}",
            lambda: self._complete_and_enqueue_once(
                event,
                artifacts,
                next_events=next_events,
                join_next_event=join_next_event,
                finalize_event=finalize_event,
                finalize_after_stages=finalize_after_stages,
            ),
        )

    async def enqueue_events(self, events: list[PipelineStageEvent]) -> None:
        if not events:
            return
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            await self._insert_outbox_events(conn, events)

    async def publish_outbox(
        self,
        publisher: OutboxPublisher,
        *,
        event_ids: list[str] | None = None,
        limit: int = 100,
    ) -> int:
        await self._ensure_schema()
        db = await self._db()
        limit = max(0, limit)
        if limit == 0:
            return 0
        lease_seconds = max(1, _env_int("PIPELINE_OUTBOX_LEASE_SECONDS", 60))
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    f"""
                    WITH pending AS (
                        SELECT event_id
                        FROM {self.outbox_table_name}
                        WHERE available_at <= NOW()
                          AND (
                              status='pending'
                              OR (status='publishing' AND lease_until <= NOW())
                          )
                          AND ($1::text[] IS NULL OR event_id = ANY($1::text[]))
                          AND (
                              event_json->>'generation_id' IS NULL
                              OR EXISTS (
                                  SELECT 1
                                  FROM {self.generations_table_name} AS generation
                                  WHERE generation.workspace_id=
                                      event_json->>'workspace_id'
                                    AND generation.generation_id=
                                      event_json->>'generation_id'
                                    AND generation.snapshot_complete=TRUE
                              )
                          )
                        ORDER BY created_at
                        LIMIT $2
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE {self.outbox_table_name} AS outbox
                    SET status='publishing',
                        attempts=outbox.attempts + 1,
                        lease_until=NOW() + ($3 * INTERVAL '1 second'),
                        updated_at=NOW()
                    FROM pending
                    WHERE outbox.event_id=pending.event_id
                    RETURNING outbox.event_id, outbox.event_json, outbox.attempts
                    """,
                    event_ids,
                    limit,
                    lease_seconds,
                )

        claimed_event_ids = [row["event_id"] for row in rows]
        published = 0
        for row in rows:
            event_id = row["event_id"]
            raw_event = row["event_json"]
            if isinstance(raw_event, str):
                outgoing = PipelineStageEvent.model_validate_json(raw_event)
            else:
                outgoing = PipelineStageEvent.model_validate(raw_event)
            await db.execute(
                f"""
                UPDATE {self.outbox_table_name}
                SET lease_until=NOW() + ($2 * INTERVAL '1 second'),
                    updated_at=NOW()
                WHERE event_id = ANY($1::text[]) AND status='publishing'
                """,
                [claimed_event_ids, lease_seconds],
            )
            try:
                await publisher(outgoing)
            except Exception as exc:
                retry_delay = min(300, 2 ** min(int(row["attempts"]), 8))
                await db.execute(
                    f"""
                    UPDATE {self.outbox_table_name}
                    SET status='pending',
                        available_at=NOW() + ($2 * INTERVAL '1 second'),
                        lease_until=NULL, last_error=$3, updated_at=NOW()
                    WHERE event_id=$1 AND status='publishing'
                    """,
                    [event_id, retry_delay, str(exc)],
                )
                logger.exception(
                    "[Pipeline] Outbox publish failed event_id=%s attempt=%s retry_seconds=%s",
                    event_id,
                    row["attempts"],
                    retry_delay,
                )
                continue
            await db.execute(
                f"""
                UPDATE {self.outbox_table_name}
                SET status='published', published_at=NOW(),
                    lease_until=NULL, last_error=NULL, updated_at=NOW()
                WHERE event_id=$1 AND status='publishing'
                """,
                [event_id],
            )
            published += 1
        return published

    async def replay_failed_stage(
        self,
        *,
        run_id: str,
        stage: StageName | str,
        shard_index: int,
        workspace_id: str,
        reason: str,
        requested_by: str,
    ) -> StageReplayResult:
        await self._ensure_schema()
        db = await self._db()
        stage_name = StageName(stage)
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    f"""
                    SELECT status, attempt, event_json
                    FROM {self.table_name}
                    WHERE run_id=$1 AND stage=$2 AND shard_index=$3
                      AND workspace_id=$4
                    FOR UPDATE
                    """,
                    run_id,
                    stage_name.value,
                    shard_index,
                    workspace_id,
                )
                if row is None:
                    raise StageRunNotFoundError("Pipeline stage run was not found")
                if row["status"] != "failed":
                    raise StageNotReplayableError(
                        f"Pipeline stage run is {row['status']}, not failed"
                    )

                recorded_attempt = int(row["attempt"])
                max_attempts = max_stage_retry_attempts()
                if recorded_attempt >= max_attempts:
                    raise StageRetryBudgetExhausted(
                        "Retry budget exhausted at "
                        f"attempt={recorded_attempt}/{max_attempts}"
                    )

                raw_event = row["event_json"]
                if isinstance(raw_event, str):
                    payload = json.loads(raw_event)
                else:
                    payload = dict(raw_event)
                original_event = PipelineStageEvent.model_validate(payload)
                payload.pop("event_id", None)
                payload.update(
                    {
                        "attempt": max(
                            recorded_attempt + 1, original_event.attempt + 1
                        ),
                        "status": "pending",
                        "created_at": utc_now_iso(),
                    }
                )
                metadata = dict(payload.get("metadata") or {})
                metadata["operator_replay"] = {
                    "requested_by": requested_by,
                    "reason": reason,
                    "requested_at": utc_now_iso(),
                }
                payload["metadata"] = metadata
                replay_event = PipelineStageEvent.model_validate(payload)

                await conn.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET status='pending', event_json=$5::jsonb, updated_at=NOW()
                    WHERE run_id=$1 AND stage=$2 AND shard_index=$3
                      AND workspace_id=$4 AND status='failed'
                    """,
                    run_id,
                    stage_name.value,
                    shard_index,
                    workspace_id,
                    replay_event.model_dump_json(),
                )
                await self._insert_outbox_events(conn, [replay_event])

        return StageReplayResult(event=replay_event, status="queued")

    async def _fail_once(self, event: PipelineStageEvent, error: PipelineError) -> None:
        await self._ensure_schema()
        db = await self._db()
        error_json = json.dumps(error.model_dump(mode="json"), ensure_ascii=False)
        async with db.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name} AS current
                        (run_id, stage, shard_index, shard_count, workspace_id, doc_id,
                         status, attempt, event_json, artifacts_json, error_json,
                         started_at, completed_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, 'failed', $7, $8::jsonb,
                            '[]'::jsonb, $9::jsonb, NOW(), NOW(), NOW())
                    ON CONFLICT (run_id, stage, shard_index)
                    DO UPDATE SET status='failed',
                        attempt=GREATEST(current.attempt, EXCLUDED.attempt),
                        error_json=EXCLUDED.error_json, completed_at=NOW(), updated_at=NOW()
                    """,
                    event.run_id,
                    event.stage_name.value,
                    event.shard_index,
                    event.shard_count,
                    event.workspace_id,
                    event.doc_id,
                    event.attempt,
                    json.dumps(event.model_dump(mode="json"), ensure_ascii=False),
                    error_json,
                )
                if event.generation_id is not None:
                    await self._mark_generation_manifest_outcome(
                        conn,
                        event,
                        status="failed",
                        error_json=error_json,
                    )

    async def fail(self, event: PipelineStageEvent, error: PipelineError) -> None:
        await self._run_with_connect_retry(
            f"fail stage={event.stage_name.value}",
            lambda: self._fail_once(event, error),
        )

    async def get_stage_outputs(
        self, run_id: str, stage: StageName | str
    ) -> list[ArtifactRef]:
        await self._ensure_schema()
        db = await self._db()
        rows = await db.query(
            f"""
            SELECT artifacts_json
            FROM {self.table_name}
            WHERE run_id=$1 AND stage=$2 AND status='done'
            ORDER BY shard_index
            """,
            [run_id, StageName(stage).value],
            multirows=True,
        )
        artifacts: list[ArtifactRef] = []
        for row in rows or []:
            raw_artifacts = row.get("artifacts_json") or []
            if isinstance(raw_artifacts, str):
                raw_artifacts = json.loads(raw_artifacts)
            artifacts.extend(_artifact_from_json(item) for item in raw_artifacts)
        return artifacts

    async def are_stages_complete(
        self, run_id: str, stages: list[StageName | str]
    ) -> bool:
        await self._ensure_schema()
        db = await self._db()
        for stage in stages:
            row = await db.query(
                f"""
                SELECT COUNT(*) FILTER (WHERE status='done') AS done_count,
                       COALESCE(MAX(shard_count), 0) AS shard_count
                FROM {self.table_name}
                WHERE run_id=$1 AND stage=$2
                """,
                [run_id, StageName(stage).value],
            )
            if not row:
                return False
            done_count = int(row.get("done_count") or 0)
            shard_count = int(row.get("shard_count") or 0)
            if shard_count == 0 or done_count < shard_count:
                return False
        return True

    async def has_any_stage_records(
        self, run_id: str, stages: list[StageName | str]
    ) -> bool:
        await self._ensure_schema()
        db = await self._db()
        stage_values = [StageName(stage).value for stage in stages]
        row = await db.query(
            f"""
            SELECT EXISTS (
                SELECT 1
                FROM {self.table_name}
                WHERE run_id=$1 AND stage = ANY($2::text[])
            ) AS has_records
            """,
            [run_id, stage_values],
        )
        return bool(row and row.get("has_records"))


def create_coordinator_from_env() -> PipelineCoordinator:
    backend = os.getenv("PIPELINE_COORDINATOR_BACKEND", "postgres").lower()
    if backend in {"memory", "inmemory"}:
        logger.warning("Using in-memory pipeline coordinator; use Postgres in production")
        return InMemoryPipelineCoordinator()
    return PostgresPipelineCoordinator()
