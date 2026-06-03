from __future__ import annotations

import asyncio
import os


class InMemoryPipelinePartitionAllocator:
    """Round-robin allocator used by tests and non-Postgres dry runs."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._next_partitions: dict[str, int] = {}

    async def allocate(self, topic: str, partition_count: int) -> int:
        if partition_count <= 0:
            raise ValueError("partition_count must be positive")

        async with self._lock:
            next_partition = self._next_partitions.get(topic, 0)
            self._next_partitions[topic] = next_partition + 1
            return next_partition % partition_count

    async def finalize(self) -> None:
        return None


class PipelinePartitionAllocator:
    """Central Postgres-backed round-robin allocator for Kafka partitions."""

    def __init__(self, table_name: str | None = None, auto_create: bool | None = None):
        self.table_name = table_name or os.getenv(
            "PIPELINE_PARTITION_CURSOR_TABLE",
            "lightrag_pipeline_partition_cursors",
        )
        self.auto_create = (
            os.getenv("PIPELINE_PARTITION_CURSOR_AUTO_CREATE", "true").lower()
            in {"1", "true", "yes", "on"}
            if auto_create is None
            else auto_create
        )
        self._initialized = False
        self._init_lock = asyncio.Lock()
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

    async def _ensure_schema(self) -> None:
        if self._initialized or not self.auto_create:
            return
        async with self._init_lock:
            if self._initialized:
                return
            db = await self._db()
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    topic TEXT PRIMARY KEY,
                    next_partition BIGINT NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            self._initialized = True

    async def allocate(self, topic: str, partition_count: int) -> int:
        if partition_count <= 0:
            raise ValueError("partition_count must be positive")

        await self._ensure_schema()
        db = await self._db()
        async with db.pool.acquire() as conn:
            allocated = await conn.fetchval(
                f"""
                INSERT INTO {self.table_name} (topic, next_partition, updated_at)
                VALUES ($1, 1, NOW())
                ON CONFLICT (topic)
                DO UPDATE SET
                    next_partition = {self.table_name}.next_partition + 1,
                    updated_at = NOW()
                RETURNING next_partition - 1
                """,
                topic,
            )
        return int(allocated or 0) % partition_count

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


def create_partition_allocator_from_env() -> PipelinePartitionAllocator:
    return PipelinePartitionAllocator()
