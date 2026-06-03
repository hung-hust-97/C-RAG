from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from lightrag.utils import logger

from .models import ArtifactRef, PipelineError, PipelineStageEvent, StageName, utc_now_iso


class ClaimStatus(str, Enum):
    ACQUIRED = "acquired"
    ALREADY_COMPLETED = "already_completed"
    ALREADY_RUNNING = "already_running"
    FAILED_ATTEMPT_RECORDED = "failed_attempt_recorded"


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


def _stage_key(run_id: str, stage: StageName | str, shard_index: int) -> tuple[str, str, int]:
    return run_id, StageName(stage).value, shard_index


def _artifact_to_json(artifact: ArtifactRef) -> dict[str, Any]:
    return artifact.model_dump(mode="json")


def _artifact_from_json(data: dict[str, Any]) -> ArtifactRef:
    return ArtifactRef.model_validate(data)


class PipelineCoordinator:
    async def claim(self, event: PipelineStageEvent) -> ClaimResult:
        raise NotImplementedError

    async def complete(
        self, event: PipelineStageEvent, artifacts: list[ArtifactRef]
    ) -> CompletionResult:
        raise NotImplementedError

    async def fail(self, event: PipelineStageEvent, error: PipelineError) -> None:
        raise NotImplementedError

    async def get_stage_outputs(
        self, run_id: str, stage: StageName | str
    ) -> list[ArtifactRef]:
        raise NotImplementedError

    async def are_stages_complete(
        self, run_id: str, stages: list[StageName | str]
    ) -> bool:
        raise NotImplementedError

    async def finalize(self) -> None:
        pass


class InMemoryPipelineCoordinator(PipelineCoordinator):
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._records: dict[tuple[str, str, int], dict[str, Any]] = {}

    async def claim(self, event: PipelineStageEvent) -> ClaimResult:
        key = _stage_key(event.run_id, event.stage_name, event.shard_index)
        async with self._lock:
            record = self._records.get(key)
            if record:
                status = record["status"]
                if status == "done":
                    return ClaimResult(ClaimStatus.ALREADY_COMPLETED)
                if status == "running":
                    return ClaimResult(ClaimStatus.ALREADY_RUNNING)
                if status == "failed" and event.attempt <= record.get("attempt", 0):
                    return ClaimResult(ClaimStatus.FAILED_ATTEMPT_RECORDED)

            self._records[key] = {
                "status": "running",
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
                "completed_at": None,
                "error": None,
            }
            return ClaimResult(ClaimStatus.ACQUIRED)

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
                    "error": None,
                }
            )
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
            )

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
                    "attempt": event.attempt,
                    "error": error.model_dump(mode="json"),
                    "completed_at": utc_now_iso(),
                }
            )

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


class PostgresPipelineCoordinator(PipelineCoordinator):
    def __init__(self, table_name: str | None = None, auto_create: bool | None = None):
        self.table_name = table_name or os.getenv(
            "PIPELINE_COORDINATOR_TABLE", "lightrag_pipeline_stage_runs"
        )
        self.auto_create = (
            os.getenv("PIPELINE_COORDINATOR_AUTO_CREATE", "true").lower()
            in {"1", "true", "yes", "on"}
            if auto_create is None
            else auto_create
        )
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self.running_stale_seconds = int(
            os.getenv("PIPELINE_STAGE_RUNNING_STALE_SECONDS", "120")
        )
        self._db_instance = None
        self._db_lock = asyncio.Lock()

    async def _db(self):
        from lightrag.kg.postgres_impl import ClientManager

        if self._db_instance is None:
            async with self._db_lock:
                if self._db_instance is None:
                    self._db_instance = await ClientManager.get_client()

        db = self._db_instance
        if db.pool is None:
            raise RuntimeError("Postgres pool is not initialized")
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
            self._initialized = True

    async def claim(self, event: PipelineStageEvent) -> ClaimResult:
        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            async with conn.transaction():
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
                if row:
                    status = row["status"]
                    if status == "done":
                        return ClaimResult(ClaimStatus.ALREADY_COMPLETED)
                    if status == "running":
                        age_seconds = float(row["age_seconds"] or 0)
                        if (
                            self.running_stale_seconds <= 0
                            or age_seconds < self.running_stale_seconds
                        ):
                            return ClaimResult(ClaimStatus.ALREADY_RUNNING)
                        logger.warning(
                            "[Pipeline] Reclaiming stale running stage run_id=%s stage=%s shard=%s age_seconds=%.1f stale_seconds=%s",
                            event.run_id,
                            event.stage_name.value,
                            event.shard_index,
                            age_seconds,
                            self.running_stale_seconds,
                        )
                    if status == "failed" and event.attempt <= int(row["attempt"]):
                        return ClaimResult(ClaimStatus.FAILED_ATTEMPT_RECORDED)
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

                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name}
                        (run_id, stage, shard_index, shard_count, workspace_id, doc_id,
                         status, attempt, event_json, artifacts_json, started_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, 'running', $7, $8::jsonb,
                            '[]'::jsonb, NOW(), NOW())
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

    async def complete(
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
        )

    async def fail(self, event: PipelineStageEvent, error: PipelineError) -> None:
        await self._ensure_schema()
        db = await self._db()
        error_json = json.dumps(error.model_dump(mode="json"), ensure_ascii=False)
        async with db.pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name}
                    (run_id, stage, shard_index, shard_count, workspace_id, doc_id,
                     status, attempt, event_json, artifacts_json, error_json,
                     started_at, completed_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, 'failed', $7, $8::jsonb,
                        '[]'::jsonb, $9::jsonb, NOW(), NOW(), NOW())
                ON CONFLICT (run_id, stage, shard_index)
                DO UPDATE SET status='failed', attempt=EXCLUDED.attempt,
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


def create_coordinator_from_env() -> PipelineCoordinator:
    backend = os.getenv("PIPELINE_COORDINATOR_BACKEND", "postgres").lower()
    if backend in {"memory", "inmemory"}:
        logger.warning("Using in-memory pipeline coordinator; use Postgres in production")
        return InMemoryPipelineCoordinator()
    return PostgresPipelineCoordinator()
