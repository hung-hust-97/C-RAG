from __future__ import annotations

import asyncio
import os
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from lightrag.base import DocStatus
from lightrag.utils import logger

from .artifacts import PipelineArtifactStore
from .coordinator import ClaimStatus, PipelineCoordinator
from .models import PipelineError, PipelineStageEvent, StageName, StageResult

StageHandler = Callable[[PipelineStageEvent, "StageExecutionContext"], Awaitable[StageResult]]
EventPublisher = Callable[[PipelineStageEvent], Awaitable[None]]
RagFactory = Callable[[str, object | None], Awaitable[Any]]

TERMINAL_SUCCESS_STATUSES = {"PROCESSED", "COMPLETED", "DUPLICATED"}


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

    async def get_rag(self, workspace_id: str, profile: object | None = None) -> Any:
        return await self.rag_factory(workspace_id, profile)


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
            logger.info("[Pipeline] Stage already completed %s", context_line)
            return StageRunOutcome(event, "already_completed", [])
        if not claim.acquired:
            logger.info("[Pipeline] Stage skipped (%s) %s", claim.status.value, context_line)
            return StageRunOutcome(event, claim.status.value, [])

        if await self._document_already_terminal(event):
            await self.context.coordinator.complete(event, [])
            logger.info("[Pipeline] Stage skipped for terminal document %s", context_line)
            return StageRunOutcome(
                event,
                "already_completed",
                [],
                "document already reached terminal status",
            )

        handler = self.handlers.get(event.stage_name)
        if handler is None:
            error = RuntimeError(f"No handler registered for stage {event.stage_name.value}")
            return await self._handle_failure(event, error)

        try:
            result = await handler(event, self.context)
            artifacts = list(result.output_artifacts)
            if result.manifest:
                artifacts.append(result.manifest)

            completion = await self.context.coordinator.complete(event, artifacts)
            next_events = list(result.next_events)

            if result.join_next_event and completion.all_shards_done:
                stage_outputs = await self.context.coordinator.get_stage_outputs(
                    event.run_id, event.stage_name
                )
                join_event = result.join_next_event.model_copy(deep=True)
                join_metadata = dict(join_event.metadata)
                join_metadata["input_artifact_uris"] = [
                    artifact.uri for artifact in stage_outputs if artifact.kind != "manifest"
                ]
                join_metadata["input_manifest_uris"] = [
                    artifact.uri for artifact in stage_outputs if artifact.kind == "manifest"
                ]
                join_event.metadata = join_metadata
                next_events.append(join_event)

            if result.finalize_event and result.finalize_after_stages:
                ready = await self.context.coordinator.are_stages_complete(
                    event.run_id, result.finalize_after_stages
                )
                if ready:
                    next_events.append(result.finalize_event)

            await self._emit_events(event, next_events)
            logger.info("[Pipeline] Stage done %s", context_line)
            return StageRunOutcome(event, "done", next_events, result.message)

        except Exception as exc:
            return await self._handle_failure(event, exc)

    async def _emit_events(
        self, current_event: PipelineStageEvent, next_events: list[PipelineStageEvent]
    ) -> None:
        for next_event in next_events:
            if current_event.fast_path:
                logger.info("[Pipeline] Fast-path handoff %s", next_event.log_context())
                await self.run(next_event)
                continue
            if self.event_publisher is None:
                logger.info(
                    "[Pipeline] No publisher configured; next event retained %s",
                    next_event.log_context(),
                )
                continue
            await self.event_publisher(next_event)

    async def _handle_failure(
        self, event: PipelineStageEvent, exc: BaseException
    ) -> StageRunOutcome:
        logger.exception("[Pipeline] Stage failed %s", event.log_context())
        pipeline_error = PipelineError(
            message=str(exc),
            stage=event.stage_name,
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
        await self.context.coordinator.fail(event, pipeline_error)
        await self._mark_document_failed(event, pipeline_error)

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
            rag = await self.context.get_rag(
                event.workspace_id, profile=StageName.FINALIZE.value
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
            }:
                raise
            logger.exception("[Pipeline] Failed to update doc_status after stage error")

    async def _document_already_terminal(self, event: PipelineStageEvent) -> bool:
        if event.stage_name == StageName.FINALIZE or not event.doc_id:
            return False

        try:
            rag = await self.context.get_rag(event.workspace_id, profile=event.stage)
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
