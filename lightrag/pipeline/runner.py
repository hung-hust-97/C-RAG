from __future__ import annotations

import asyncio
import os
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from lightrag.base import DocStatus
from lightrag.utils import logger

from .artifacts import PipelineArtifactStore
from .coordinator import (
    ClaimStatus,
    PipelineCoordinator,
    WorkspaceGenerationStatus,
)
from .models import PipelineError, PipelineStageEvent, StageName, StageResult

StageHandler = Callable[
    [PipelineStageEvent, "StageExecutionContext"], Awaitable[StageResult]
]
EventPublisher = Callable[[PipelineStageEvent], Awaitable[None]]


class RagFactory(Protocol):
    async def __call__(
        self,
        workspace_id: str,
        profile: object | None = None,
        *,
        storage_workspace_id: str | None = None,
    ) -> Any: ...

TERMINAL_SUCCESS_STATUSES = {"PROCESSED", "COMPLETED", "DUPLICATED"}


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


def _stage_env_name(prefix: str, stage: StageName | str) -> str:
    return f"{prefix}_{StageName(stage).value.upper()}"


def _stage_env_int(prefix: str, stage: StageName | str, default: int) -> int:
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


async def fail_fast_gather(
    *awaitables: Awaitable[Any],
    debug_sync: bool = False,
) -> list[Any]:
    """Await all work and raise the first error immediately.

    This helper intentionally owns task creation so callers do not leave
    fire-and-forget work behind. In debug mode it runs sequentially.
    """
    if debug_sync:
        results = []
        for index, awaitable in enumerate(awaitables):
            try:
                results.append(await awaitable)
            except Exception:
                for pending_awaitable in awaitables[index + 1 :]:
                    close = getattr(pending_awaitable, "close", None)
                    if callable(close):
                        close()
                raise
        return results

    tasks = [asyncio.create_task(awaitable) for awaitable in awaitables]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    first_exception: BaseException | None = None
    for task in done:
        exc = task.exception()
        if exc is not None and first_exception is None:
            first_exception = exc

    if first_exception is not None:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        raise first_exception

    if pending:
        await asyncio.gather(*pending)

    return [task.result() for task in tasks]


@dataclass
class StageExecutionContext:
    artifact_store: PipelineArtifactStore
    coordinator: PipelineCoordinator
    rag_factory: RagFactory
    debug_sync: bool = False
    partition_allocator: Any = None
    producer: Any = None

    async def get_rag(
        self,
        workspace_id: str,
        profile: object | None = None,
        *,
        generation_id: str | None = None,
        storage_workspace_id: str | None = None,
    ) -> Any:
        if generation_id is not None or storage_workspace_id is not None:
            if not generation_id or not storage_workspace_id:
                raise ValueError(
                    "generation_id and storage_workspace_id must be provided together"
                )
            generation = await self.coordinator.get_workspace_generation(
                workspace_id, generation_id
            )
            if (
                generation is None
                or generation.status != WorkspaceGenerationStatus.BUILDING
                or generation.storage_workspace_id != storage_workspace_id
                or not await self.coordinator.is_workspace_write_locked(workspace_id)
            ):
                raise RuntimeError(
                    "Generation stage event does not match the locked building "
                    f"generation: workspace_id={workspace_id}, "
                    f"generation_id={generation_id}"
                )
            return await self.rag_factory(
                workspace_id,
                profile,
                storage_workspace_id=storage_workspace_id,
            )

        return await self.rag_factory(workspace_id, profile)

    async def get_rag_for_event(
        self, event: PipelineStageEvent, profile: object | None = None
    ) -> Any:
        return await self.get_rag(
            event.workspace_id,
            profile=profile,
            generation_id=event.generation_id,
            storage_workspace_id=event.storage_workspace_id,
        )


@dataclass
class StageRunOutcome:
    event: PipelineStageEvent
    status: str
    next_events: list[PipelineStageEvent]
    message: str | None = None


class StageRunner:
    def __init__(
        self,
        *,
        handlers: dict[StageName, StageHandler],
        context: StageExecutionContext,
        event_publisher: EventPublisher | None = None,
        failure_publisher: EventPublisher | None = None,
        raise_stage_errors: bool = False,
    ) -> None:
        self.handlers = handlers
        self.context = context
        self.event_publisher = event_publisher
        self.failure_publisher = failure_publisher
        self.raise_stage_errors = raise_stage_errors

    async def run(self, event: PipelineStageEvent) -> StageRunOutcome:
        context_line = event.log_context()
        logger.info("[Pipeline] Stage start %s", context_line)

        claim = await self.context.coordinator.claim(event)
        if claim.status == ClaimStatus.ALREADY_COMPLETED:
            try:
                next_events = await self._recover_next_events_for_completed_stage(event)
                if next_events:
                    await self._emit_events(event, next_events)
                    logger.info(
                        "[Pipeline] Stage already completed; replayed downstream events %s",
                        context_line,
                    )
                    return StageRunOutcome(
                        event,
                        "already_completed",
                        next_events,
                        "replayed downstream events for completed stage",
                    )
                logger.info("[Pipeline] Stage already completed %s", context_line)
                return StageRunOutcome(event, "already_completed", [])
            except Exception as exc:
                logger.exception(
                    "[Pipeline] Failed to replay downstream events for completed stage %s",
                    context_line,
                )
                return await self._handle_failure(event, exc)
        if not claim.acquired:
            logger.info(
                "[Pipeline] Stage skipped (%s) %s", claim.status.value, context_line
            )
            return StageRunOutcome(event, claim.status.value, [])

        if await self._document_already_terminal(event):
            await self.context.coordinator.complete(event, [])
            logger.info(
                "[Pipeline] Stage skipped for terminal document %s", context_line
            )
            return StageRunOutcome(
                event,
                "already_completed",
                [],
                "document already reached terminal status",
            )

        handler = self.handlers.get(event.stage_name)
        if handler is None:
            error = RuntimeError(
                f"No handler registered for stage {event.stage_name.value}"
            )
            return await self._handle_failure(event, error)

        heartbeat_task = self._start_stage_heartbeat(event)
        try:
            result = await handler(event, self.context)
            artifacts = list(result.output_artifacts)
            if result.manifest:
                artifacts.append(result.manifest)

            completion = await self.context.coordinator.complete_and_enqueue(
                event,
                artifacts,
                next_events=result.next_events,
                join_next_event=result.join_next_event,
                finalize_event=result.finalize_event,
                finalize_after_stages=result.finalize_after_stages,
            )
            next_events = completion.next_events

            await self._emit_events(event, next_events)
            logger.info("[Pipeline] Stage done %s", context_line)
            return StageRunOutcome(event, "done", next_events, result.message)

        except Exception as exc:
            return await self._handle_failure(event, exc)
        finally:
            await self._stop_stage_heartbeat(heartbeat_task)

    def _start_stage_heartbeat(
        self, event: PipelineStageEvent
    ) -> asyncio.Task[None] | None:
        interval_seconds = _stage_env_int(
            "PIPELINE_STAGE_HEARTBEAT_SECONDS",
            event.stage_name,
            60,
        )
        if interval_seconds <= 0:
            return None
        return asyncio.create_task(
            self._stage_heartbeat_loop(event, interval_seconds),
            name=(
                "pipeline-heartbeat-"
                f"{event.stage_name.value}-{event.run_id}-{event.shard_index}"
            ),
        )

    async def _stop_stage_heartbeat(
        self, heartbeat_task: asyncio.Task[None] | None
    ) -> None:
        if heartbeat_task is None:
            return
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    async def _stage_heartbeat_loop(
        self, event: PipelineStageEvent, interval_seconds: int
    ) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self.context.coordinator.heartbeat(event)
            except Exception:
                logger.warning(
                    "[Pipeline] Stage heartbeat failed %s",
                    event.log_context(),
                    exc_info=True,
                )

    async def _emit_events(
        self, current_event: PipelineStageEvent, next_events: list[PipelineStageEvent]
    ) -> None:
        # Pause logic disabled per user request
        if current_event.fast_path:
            for next_event in next_events:
                logger.info("[Pipeline] Fast-path handoff %s", next_event.log_context())
                await self.run(next_event)
            return
        if not next_events:
            return
        await self.context.coordinator.enqueue_events(next_events)
        if self.event_publisher is None:
            logger.info(
                "[Pipeline] No publisher configured; %s downstream events retained in outbox",
                len(next_events),
            )
            return
        try:
            await self.context.coordinator.publish_outbox(
                self.event_publisher,
                event_ids=[event.event_id for event in next_events],
                limit=len(next_events),
            )
        except Exception:
            logger.exception(
                "[Pipeline] Could not dispatch durable downstream events %s",
                current_event.log_context(),
            )

    async def _recover_next_events_for_completed_stage(
        self, event: PipelineStageEvent
    ) -> list[PipelineStageEvent]:
        if event.stage_name == StageName.KG_MERGE:
            return await self._recover_kg_merge_next_events(event)

        if event.stage_name != StageName.CHUNKING:
            return []

        try:
            downstream_started = await self.context.coordinator.has_any_stage_records(
                event.run_id,
                [StageName.CHUNK_EMBEDDING, StageName.ENTITY_EXTRACTION],
            )
            if downstream_started:
                logger.info(
                    "[Pipeline] Stage already completed; downstream records exist, "
                    "skipping chunking replay %s",
                    event.log_context(),
                )
                return []

            stage_outputs = await self.context.coordinator.get_stage_outputs(
                event.run_id, event.stage_name
            )
            chunks_artifact = next(
                (artifact for artifact in stage_outputs if artifact.kind == "chunks"),
                None,
            )
            if chunks_artifact is None:
                logger.warning(
                    "[Pipeline] Cannot replay chunking downstream events; chunks artifact missing %s",
                    event.log_context(),
                )
                return []

            from .stages.common import rows_to_chunks

            rows = await self.context.artifact_store.read_jsonl(chunks_artifact.uri)
            chunks = rows_to_chunks(rows)
            chunk_ids = list(chunks.keys())
            if not chunk_ids:
                logger.warning(
                    "[Pipeline] Cannot replay chunking downstream events; chunks artifact is empty %s",
                    event.log_context(),
                )
                return []

            chunk_embedding_shard_size = _env_int(
                "PIPELINE_CHUNK_EMBEDDING_SHARD_SIZE", 512
            )
            if event.debug_sync or self.context.debug_sync:
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

            shard_size = _env_int("PIPELINE_EXTRACT_CHUNK_SHARD_SIZE", 32)
            if event.debug_sync or self.context.debug_sync:
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
                            "legal_single_shard": False,
                        },
                    )
                )

            logger.warning(
                "[Pipeline] Replaying chunking downstream events for completed stage %s chunks=%s chunk_embedding_shards=%s extraction_shards=%s",
                event.log_context(),
                len(chunks),
                chunk_embedding_shard_count,
                shard_count,
            )
            return next_events
        except Exception:
            logger.exception(
                "[Pipeline] Failed to replay downstream events for completed stage %s",
                event.log_context(),
            )
            raise

    async def _recover_kg_merge_next_events(
        self, event: PipelineStageEvent
    ) -> list[PipelineStageEvent]:
        try:
            stage_outputs = await self.context.coordinator.get_stage_outputs(
                event.run_id, event.stage_name
            )
            payload_artifacts = [
                artifact
                for artifact in stage_outputs
                if artifact.kind == "kg_vector_payload"
            ]
            if not payload_artifacts:
                logger.warning(
                    "[Pipeline] Cannot replay kg_merge downstream events; vector payload artifact missing %s",
                    event.log_context(),
                )
                return []

            payload_artifacts.sort(
                key=lambda artifact: int(artifact.metadata.get("shard_index", 0))
            )
            shard_count = max(
                len(payload_artifacts),
                max(
                    int(artifact.metadata.get("shard_count", 1))
                    for artifact in payload_artifacts
                ),
            )
            next_events = []
            for fallback_index, artifact in enumerate(payload_artifacts):
                shard_index = int(artifact.metadata.get("shard_index", fallback_index))
                next_events.append(
                    event.next_event(
                        StageName.KG_EMBEDDING,
                        artifact_uri=artifact.uri,
                        shard_index=shard_index,
                        shard_count=shard_count,
                        counts={
                            "entities": int(artifact.metadata.get("entities", 0)),
                            "relations": int(artifact.metadata.get("relations", 0)),
                            "vector_rows": int(
                                artifact.metadata.get("vector_rows", artifact.row_count)
                            ),
                        },
                        metadata={
                            "kg_vector_payload_artifact_uri": artifact.uri,
                            "replayed_from_completed_stage": event.stage_name.value,
                        },
                    )
                )

            logger.warning(
                "[Pipeline] Replaying kg_merge downstream events for completed stage %s kg_embedding_shards=%s",
                event.log_context(),
                len(next_events),
            )
            return next_events
        except Exception:
            logger.exception(
                "[Pipeline] Failed to replay downstream events for completed kg_merge stage %s",
                event.log_context(),
            )
            raise

    async def _handle_failure(
        self, event: PipelineStageEvent, exc: BaseException
    ) -> StageRunOutcome:
        if isinstance(exc, FileNotFoundError):
            logger.error("[Pipeline] Stage failed (artifact missing) %s: %s", event.log_context(), str(exc))
        else:
            logger.exception("[Pipeline] Stage failed %s", event.log_context())
            
        pipeline_error = PipelineError(
            message=str(exc),
            stage=event.stage_name,
            error_type=type(exc).__name__,
            traceback=traceback.format_exc() if not isinstance(exc, FileNotFoundError) else "",
        )
        await self.context.coordinator.fail(event, pipeline_error)
        await self._mark_document_failed(event, pipeline_error)

        if event.stage_name != StageName.FINALIZE:
            finalize_event = event.next_event(
                StageName.FINALIZE,
                metadata={
                    "error": pipeline_error.model_dump(mode="json"),
                },
            )
            if self.event_publisher is not None:
                await self.event_publisher(finalize_event)
        else:
            failure_event = event.model_copy(deep=True)
            failure_event.status = "failed"
            failure_event.metadata = {
                **failure_event.metadata,
                "error": pipeline_error.model_dump(mode="json"),
            }
            if self.failure_publisher is not None:
                await self.failure_publisher(failure_event)

        if self.raise_stage_errors:
            raise exc
        return StageRunOutcome(event, "failed", [], str(exc))

    async def _mark_document_failed(
        self, event: PipelineStageEvent, error: PipelineError
    ) -> None:
        try:
            try:
                if await self.context.coordinator.are_stages_complete(
                    event.run_id, [StageName.FINALIZE]
                ):
                    logger.warning(
                        "[Pipeline] Not marking document as failed because finalize is already completed %s run_id=%s",
                        event.log_context(),
                        event.run_id,
                    )
                    return
            except Exception:
                pass

            rag = await self.context.get_rag_for_event(
                event, profile=StageName.FINALIZE.value
            )
            status_doc = await rag.doc_status.get_by_id(event.doc_id)
            existing_metadata = {}
            if status_doc is not None:
                existing_status = _status_get(status_doc, "status")
                if _is_terminal_success_status(existing_status):
                    logger.warning(
                        "[Pipeline] Not marking terminal document as failed %s current_status=%s",
                        event.log_context(),
                        existing_status,
                    )
                    return
                existing_metadata = getattr(status_doc, "metadata", None)
                if existing_metadata is None and isinstance(status_doc, dict):
                    existing_metadata = status_doc.get("metadata", {})
                existing_metadata = existing_metadata or {}
            await rag.doc_status.upsert(
                {
                    event.doc_id: {
                        "status": DocStatus.FAILED,
                        "error_msg": error.message,
                        "error_stage": event.stage_name.value,
                        "updated_at": error.created_at,
                        "metadata": {
                            **existing_metadata,
                            "pipeline_run_id": event.run_id,
                            "pipeline_event_id": event.event_id,
                        },
                    }
                }
            )
        except Exception:
            if os.getenv("PIPELINE_STRICT_FAILURE_STATUS", "false").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                raise
            logger.exception("[Pipeline] Failed to update doc_status after stage error")

    async def _document_already_terminal(self, event: PipelineStageEvent) -> bool:
        if event.stage_name == StageName.FINALIZE or not event.doc_id:
            return False

        try:
            if await self.context.coordinator.are_stages_complete(
                event.run_id, [StageName.FINALIZE]
            ):
                logger.warning(
                    "[Pipeline] Ignoring stale stage for already finalized run_id=%s %s",
                    event.run_id,
                    event.log_context(),
                )
                return True
        except Exception:
            pass

        try:
            rag = await self.context.get_rag_for_event(event, profile=event.stage)
            status_doc = await rag.doc_status.get_by_id(event.doc_id)
        except Exception:
            logger.exception(
                "[Pipeline] Failed to check terminal status before stage %s",
                event.log_context(),
            )
            return False

        status = _status_get(status_doc, "status")
        if _is_terminal_success_status(status):
            logger.warning(
                "[Pipeline] Ignoring stale stage for terminal document %s current_status=%s",
                event.log_context(),
                status,
            )
            return True
        return False
