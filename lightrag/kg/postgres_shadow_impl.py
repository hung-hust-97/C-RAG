"""Pipeline-owned PostgreSQL storage for shadow (generation) rebuilds.

The KMS-managed tables (``tbl_document``, ``tbl_lightrag_document_chunk``,
``tbl_legal_document``) are shared by every generation, so a shadow rebuild must
never write to them. These adapters keep shadow document status and text chunks
in additive, self-managed tables keyed by the generation ``storage_workspace``
and only *read* source-document metadata from the KMS tables, scoped to the
logical workspace.

Only the ``text_chunks`` and ``doc_status`` namespaces are supported; they are
not registered in ``STORAGES`` yet, so production routing is unchanged.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any, Union, final

from ..base import DocProcessingStatus, DocStatus, DocStatusStorage, BaseKVStorage
from ..kg.shared_storage import get_data_init_lock
from ..namespace import NameSpace, is_namespace
from .postgres_impl import ClientManager, PostgreSQLDB, to_uuid

SHADOW_DOC_STATUS_TABLE = "lightrag_shadow_doc_status"
SHADOW_TEXT_CHUNKS_TABLE = "lightrag_shadow_text_chunks"

SHADOW_DDL = [
    f"""
    CREATE TABLE IF NOT EXISTS {SHADOW_DOC_STATUS_TABLE} (
        storage_workspace_id TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        status TEXT NOT NULL,
        content_summary TEXT,
        content_length INTEGER,
        file_path TEXT,
        content_hash TEXT,
        chunks_count INTEGER,
        chunks_list JSONB NOT NULL DEFAULT '[]'::jsonb,
        error_msg TEXT,
        error_stage TEXT,
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        created_at TIMESTAMP NOT NULL,
        updated_at TIMESTAMP NOT NULL,
        PRIMARY KEY (storage_workspace_id, doc_id)
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{SHADOW_DOC_STATUS_TABLE}_status
    ON {SHADOW_DOC_STATUS_TABLE} (storage_workspace_id, status)
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {SHADOW_TEXT_CHUNKS_TABLE} (
        storage_workspace_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        full_doc_id TEXT NOT NULL,
        source_document_id UUID,
        legal_document_id BIGINT,
        tokens INTEGER,
        order_index INTEGER,
        content TEXT NOT NULL,
        extra JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        create_time TIMESTAMP NOT NULL,
        update_time TIMESTAMP NOT NULL,
        PRIMARY KEY (storage_workspace_id, chunk_id)
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{SHADOW_TEXT_CHUNKS_TABLE}_doc
    ON {SHADOW_TEXT_CHUNKS_TABLE} (storage_workspace_id, full_doc_id)
    """,
]

# Source rows are read-only and always filtered by the *logical* workspace.
_SHADOW_CHUNK_SELECT = f"""
    SELECT c.chunk_id AS id, c.tokens, c.content,
           c.order_index AS chunk_order_index,
           c.full_doc_id, c.legal_document_id, c.extra,
           COALESCE(d.origin_path, d.document_name, ld.title) AS source_file_path,
           ld.status::text AS legal_status,
           EXTRACT(EPOCH FROM c.create_time)::BIGINT AS create_time,
           EXTRACT(EPOCH FROM c.update_time)::BIGINT AS update_time
    FROM {SHADOW_TEXT_CHUNKS_TABLE} c
    LEFT JOIN tbl_document d
        ON d.document_id = c.source_document_id AND d.workspace_id = $2
    LEFT JOIN tbl_legal_document ld
        ON ld.id = c.legal_document_id AND ld.workspace_id = $2
"""

_SOURCE_DOC_SELECT = """
    SELECT document_id::text AS document_id, content_summary, content_length,
           document_name, content_hash, create_time, update_time
    FROM tbl_document
    WHERE workspace_id = $1 AND document_id = $2
"""

_SOURCE_LEGAL_SELECT = """
    SELECT id::text AS document_id, summary AS content_summary,
           length(COALESCE(summary, '')) AS content_length,
           title AS document_name, created_at AS create_time,
           updated_at AS update_time
    FROM tbl_legal_document
    WHERE workspace_id = $1 AND id = $2
"""


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(timezone.utc).replace(tzinfo=None)


def _to_naive_utc(value: Any) -> datetime.datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.datetime.fromisoformat(value)
        except ValueError:
            return None
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    return None


def _iso(value: datetime.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _json_load(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _status_value(status: Any) -> str:
    status = getattr(status, "value", status)
    return str(status or "UNKNOWN")


def _require_logical_workspace(global_config: dict[str, Any]) -> str:
    logical = global_config.get("workspace")
    if not logical:
        raise ValueError(
            "Shadow storage requires the logical workspace in global_config"
        )
    return str(logical)


def _require_explicit_storage_workspace(
    storage_workspace: str | None, logical_workspace: str, global_config: dict[str, Any]
) -> str:
    # The shadow tables must never be addressed by the logical workspace: that
    # would mix shadow rows into the data served by the active generation.
    if not global_config.get("_storage_workspace_explicit", False):
        raise ValueError("Shadow storage requires an explicit storage_workspace")
    if not storage_workspace or storage_workspace == logical_workspace:
        raise ValueError(
            "Shadow storage_workspace must differ from the logical workspace"
        )
    return storage_workspace


async def ensure_shadow_tables(db: PostgreSQLDB) -> None:
    """Create the additive, pipeline-owned shadow tables if missing."""
    for statement in SHADOW_DDL:
        await db.execute(statement)


@dataclass
class _ShadowStorageMixin:
    db: PostgreSQLDB = field(default=None)

    async def _initialize_shadow(self) -> str:
        logical = _require_logical_workspace(self.global_config)
        self.workspace = _require_explicit_storage_workspace(
            self.workspace, logical, self.global_config
        )
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()
            await ensure_shadow_tables(self.db)
        return logical

    @property
    def _logical_workspace_uuid(self):
        return to_uuid(_require_logical_workspace(self.global_config))

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    async def index_done_callback(self) -> None:
        pass


@final
@dataclass
class PGShadowTextChunksStorage(_ShadowStorageMixin, BaseKVStorage):
    """Shadow ``text_chunks`` KV store; reads source metadata, never writes KMS."""

    def __post_init__(self):
        if not is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            raise ValueError(
                f"PGShadowTextChunksStorage only supports text_chunks, got {self.namespace}"
            )

    async def initialize(self):
        await self._initialize_shadow()

    @staticmethod
    def _row_to_chunk(row: dict[str, Any]) -> dict[str, Any]:
        extra = _json_load(row.get("extra"), {})
        chunk = dict(extra)
        chunk.update(
            {
                "id": row["id"],
                "tokens": row.get("tokens"),
                "content": row.get("content") or "",
                "chunk_order_index": row.get("chunk_order_index"),
                "full_doc_id": row.get("full_doc_id"),
                "llm_cache_list": extra.get("llm_cache_list", []),
                "create_time": row.get("create_time") or 0,
            }
        )
        chunk["update_time"] = row.get("update_time") or chunk["create_time"]
        # Source-document metadata wins over whatever was stored at write time.
        if row.get("source_file_path"):
            chunk["file_path"] = row["source_file_path"]
        if row.get("legal_document_id") is not None:
            chunk["legal_document_id"] = row["legal_document_id"]
            chunk["doc_type"] = "legal"
            chunk["legal_status"] = row.get("legal_status")
        return chunk

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        rows = await self.get_by_ids([id])
        return rows[0] if rows else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        sql = (
            _SHADOW_CHUNK_SELECT
            + " WHERE c.storage_workspace_id = $1 AND c.chunk_id = ANY($3)"
        )
        rows = await self.db.query(
            sql, [self.workspace, self._logical_workspace_uuid, list(ids)], True
        )
        by_id = {row["id"]: self._row_to_chunk(row) for row in rows or []}
        # Preserve request order and drop unknown ids, like the other KV stores.
        return [by_id[i] for i in ids if i in by_id]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        rows = await self.db.query(
            f"SELECT chunk_id FROM {SHADOW_TEXT_CHUNKS_TABLE} "
            "WHERE storage_workspace_id = $1 AND chunk_id = ANY($2)",
            [self.workspace, list(keys)],
            True,
        )
        existing = {row["chunk_id"] for row in rows or []}
        return {key for key in keys if key not in existing}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        now = _utcnow()
        sql = f"""
            INSERT INTO {SHADOW_TEXT_CHUNKS_TABLE}
            (storage_workspace_id, chunk_id, full_doc_id, source_document_id,
             legal_document_id, tokens, order_index, content, extra,
             create_time, update_time)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $10)
            ON CONFLICT (storage_workspace_id, chunk_id) DO UPDATE SET
                full_doc_id = EXCLUDED.full_doc_id,
                source_document_id = EXCLUDED.source_document_id,
                legal_document_id = EXCLUDED.legal_document_id,
                tokens = EXCLUDED.tokens,
                order_index = EXCLUDED.order_index,
                content = EXCLUDED.content,
                extra = EXCLUDED.extra,
                update_time = EXCLUDED.update_time
        """
        known = {
            "tokens",
            "content",
            "chunk_order_index",
            "full_doc_id",
            "legal_document_id",
        }
        batch = []
        for chunk_id, value in data.items():
            legal_id = value.get("legal_document_id")
            full_doc_id = str(value["full_doc_id"])
            batch.append(
                (
                    self.workspace,
                    chunk_id,
                    full_doc_id,
                    None if legal_id else to_uuid(full_doc_id),
                    int(legal_id) if legal_id else None,
                    value.get("tokens"),
                    value.get("chunk_order_index"),
                    value["content"],
                    json.dumps({k: v for k, v in value.items() if k not in known}),
                    now,
                )
            )

        async def _run(connection: Any) -> None:
            await connection.executemany(sql, batch)

        await self.db._run_with_retry(_run)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        await self.db.execute(
            f"DELETE FROM {SHADOW_TEXT_CHUNKS_TABLE} "
            "WHERE storage_workspace_id = $1 AND chunk_id = ANY($2)",
            [self.workspace, list(ids)],
        )

    async def is_empty(self) -> bool:
        row = await self.db.query(
            f"SELECT 1 AS present FROM {SHADOW_TEXT_CHUNKS_TABLE} "
            "WHERE storage_workspace_id = $1 LIMIT 1",
            [self.workspace],
        )
        return row is None

    async def drop(self) -> dict[str, str]:
        try:
            await self.db.execute(
                f"DELETE FROM {SHADOW_TEXT_CHUNKS_TABLE} WHERE storage_workspace_id = $1",
                [self.workspace],
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": str(e)}


@final
@dataclass
class PGShadowDocStatusStorage(_ShadowStorageMixin, DocStatusStorage):
    """Shadow doc-status store.

    Rows are written only to the shadow table. When a document has no shadow row
    yet, ``get_by_id`` seeds descriptive metadata (name, summary, hash) from the
    source document but reports ``EXTRACTED`` so the stages do not mistake the
    already-``PROCESSED`` source for a finished shadow document.
    """

    def __post_init__(self):
        if not is_namespace(self.namespace, NameSpace.DOC_STATUS):
            raise ValueError(
                f"PGShadowDocStatusStorage only supports doc_status, got {self.namespace}"
            )

    async def initialize(self):
        await self._initialize_shadow()

    @staticmethod
    def _row_to_doc(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["doc_id"],
            "content_summary": row.get("content_summary"),
            "content_length": row.get("content_length"),
            "status": row["status"],
            "created_at": _iso(row.get("created_at")),
            "updated_at": _iso(row.get("updated_at")),
            "file_path": row.get("file_path"),
            "content_hash": row.get("content_hash"),
            "chunks_count": row.get("chunks_count"),
            "chunks_list": _json_load(row.get("chunks_list"), []),
            "error_msg": row.get("error_msg"),
            "error_stage": row.get("error_stage"),
            "metadata": _json_load(row.get("metadata"), {}),
        }

    async def _get_source_seed(self, doc_id: str) -> dict[str, Any] | None:
        clean = str(doc_id)
        for prefix in ("legal-", "doc-"):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
                break
        if clean.isdigit():
            row = await self.db.query(
                _SOURCE_LEGAL_SELECT, [self._logical_workspace_uuid, int(clean)]
            )
        else:
            try:
                key = to_uuid(doc_id)
            except ValueError:
                return None
            row = await self.db.query(
                _SOURCE_DOC_SELECT, [self._logical_workspace_uuid, key]
            )
        if not row:
            return None
        return {
            "id": str(doc_id),
            "content_summary": row.get("content_summary"),
            "content_length": row.get("content_length"),
            "status": DocStatus.EXTRACTED.value,
            "created_at": _iso(_to_naive_utc(row.get("create_time"))),
            "updated_at": _iso(_to_naive_utc(row.get("update_time"))),
            "file_path": row.get("document_name"),
            "content_hash": row.get("content_hash"),
            "chunks_count": None,
            "chunks_list": [],
            "error_msg": None,
            "error_stage": None,
            "metadata": {"shadow_seeded_from_source": True},
        }

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        row = await self.db.query(
            f"SELECT * FROM {SHADOW_DOC_STATUS_TABLE} "
            "WHERE storage_workspace_id = $1 AND doc_id = $2",
            [self.workspace, str(id)],
        )
        if row:
            return self._row_to_doc(row)
        return await self._get_source_seed(str(id))

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        rows = await self.db.query(
            f"SELECT * FROM {SHADOW_DOC_STATUS_TABLE} "
            "WHERE storage_workspace_id = $1 AND doc_id = ANY($2)",
            [self.workspace, [str(i) for i in ids]],
            True,
        )
        by_id = {row["doc_id"]: self._row_to_doc(row) for row in rows or []}
        result = []
        for doc_id in ids:
            doc = by_id.get(str(doc_id)) or await self._get_source_seed(str(doc_id))
            if doc:
                result.append(doc)
        return result

    async def filter_keys(self, keys: set[str]) -> set[str]:
        if not keys:
            return set()
        rows = await self.db.query(
            f"SELECT doc_id FROM {SHADOW_DOC_STATUS_TABLE} "
            "WHERE storage_workspace_id = $1 AND doc_id = ANY($2)",
            [self.workspace, list(keys)],
            True,
        )
        existing = {row["doc_id"] for row in rows or []}
        return {key for key in keys if key not in existing}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        sql = f"""
            INSERT INTO {SHADOW_DOC_STATUS_TABLE}
            (storage_workspace_id, doc_id, status, content_summary, content_length,
             file_path, content_hash, chunks_count, chunks_list, error_msg,
             error_stage, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11,
                    $12::jsonb, $13, $14)
            ON CONFLICT (storage_workspace_id, doc_id) DO UPDATE SET
                status = EXCLUDED.status,
                content_summary = EXCLUDED.content_summary,
                content_length = EXCLUDED.content_length,
                file_path = EXCLUDED.file_path,
                content_hash = EXCLUDED.content_hash,
                chunks_count = EXCLUDED.chunks_count,
                chunks_list = EXCLUDED.chunks_list,
                error_msg = EXCLUDED.error_msg,
                error_stage = EXCLUDED.error_stage,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at
        """
        existing = {doc["id"]: doc for doc in await self.get_by_ids(list(data))}
        now = _utcnow()
        for doc_id, update in data.items():
            merged = {**existing.get(doc_id, {}), **update}
            await self.db.execute(
                sql,
                [
                    self.workspace,
                    doc_id,
                    _status_value(merged.get("status")),
                    merged.get("content_summary"),
                    merged.get("content_length"),
                    merged.get("file_path"),
                    merged.get("content_hash"),
                    merged.get("chunks_count"),
                    json.dumps(merged.get("chunks_list") or []),
                    merged.get("error_msg"),
                    merged.get("error_stage"),
                    json.dumps(merged.get("metadata") or {}),
                    _to_naive_utc(merged.get("created_at")) or now,
                    _to_naive_utc(merged.get("updated_at")) or now,
                ],
            )

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        await self.db.execute(
            f"DELETE FROM {SHADOW_DOC_STATUS_TABLE} "
            "WHERE storage_workspace_id = $1 AND doc_id = ANY($2)",
            [self.workspace, [str(i) for i in ids]],
        )

    async def is_empty(self) -> bool:
        row = await self.db.query(
            f"SELECT 1 AS present FROM {SHADOW_DOC_STATUS_TABLE} "
            "WHERE storage_workspace_id = $1 LIMIT 1",
            [self.workspace],
        )
        return row is None

    async def drop(self) -> dict[str, str]:
        try:
            await self.db.execute(
                f"DELETE FROM {SHADOW_DOC_STATUS_TABLE} WHERE storage_workspace_id = $1",
                [self.workspace],
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": str(e)}

    async def get_status_counts(self) -> dict[str, int]:
        rows = await self.db.query(
            f"SELECT status, COUNT(*) AS count FROM {SHADOW_DOC_STATUS_TABLE} "
            "WHERE storage_workspace_id = $1 GROUP BY status",
            [self.workspace],
            True,
        )
        return {row["status"]: int(row["count"]) for row in rows or []}

    async def get_all_status_counts(self) -> dict[str, int]:
        return await self.get_status_counts()

    async def get_status_counts_across_workspaces(self) -> dict[str, dict[str, int]]:
        return {self.workspace: await self.get_status_counts()}

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        rows = await self.db.query(
            f"SELECT * FROM {SHADOW_DOC_STATUS_TABLE} "
            "WHERE storage_workspace_id = $1 AND status = $2",
            [self.workspace, _status_value(status)],
            True,
        )
        return {
            row["doc_id"]: self._to_processing_status(self._row_to_doc(row))
            for row in rows or []
        }

    @staticmethod
    def _to_processing_status(doc: dict[str, Any]) -> DocProcessingStatus:
        return DocProcessingStatus(
            content_summary=doc.get("content_summary") or "",
            content_length=doc.get("content_length") or 0,
            file_path=doc.get("file_path") or "",
            status=doc["status"],
            created_at=doc.get("created_at") or "",
            updated_at=doc.get("updated_at") or "",
            content_hash=doc.get("content_hash"),
            chunks_count=doc.get("chunks_count"),
            chunks_list=doc.get("chunks_list") or [],
            error_msg=doc.get("error_msg"),
            error_stage=doc.get("error_stage"),
            metadata=doc.get("metadata") or {},
        )

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        page = max(page, 1)
        page_size = min(max(page_size, 10), 200)
        sort_column = {
            "created_at": "created_at",
            "updated_at": "updated_at",
            "id": "doc_id",
        }.get(sort_field, "updated_at")
        direction = "ASC" if sort_direction.lower() == "asc" else "DESC"
        params: list[Any] = [self.workspace]
        where = "storage_workspace_id = $1"
        if status_filter is not None:
            params.append(_status_value(status_filter))
            where += f" AND status = ${len(params)}"
        total_row = await self.db.query(
            f"SELECT COUNT(*) AS total FROM {SHADOW_DOC_STATUS_TABLE} WHERE {where}",
            params,
        )
        rows = await self.db.query(
            f"SELECT * FROM {SHADOW_DOC_STATUS_TABLE} WHERE {where} "
            f"ORDER BY {sort_column} {direction}, doc_id ASC "
            f"LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}",
            [*params, page_size, (page - 1) * page_size],
            True,
        )
        docs = [
            (row["doc_id"], self._to_processing_status(self._row_to_doc(row)))
            for row in rows or []
        ]
        return docs, int(total_row["total"]) if total_row else 0

    async def get_doc_by_file_path(self, file_path: str) -> dict[str, Any] | None:
        row = await self.db.query(
            f"SELECT * FROM {SHADOW_DOC_STATUS_TABLE} "
            "WHERE storage_workspace_id = $1 AND file_path = $2 LIMIT 1",
            [self.workspace, file_path],
        )
        return self._row_to_doc(row) if row else None

    async def check_duplicate_by_content_hash(
        self, content_hash: str, exclude_doc_id: str | None = None
    ) -> tuple[bool, str | None]:
        params: list[Any] = [self.workspace, content_hash]
        sql = (
            f"SELECT doc_id FROM {SHADOW_DOC_STATUS_TABLE} "
            "WHERE storage_workspace_id = $1 AND content_hash = $2"
        )
        if exclude_doc_id:
            params.append(str(exclude_doc_id))
            sql += " AND doc_id <> $3"
        row = await self.db.query(sql + " LIMIT 1", params)
        return (True, row["doc_id"]) if row else (False, None)

    async def get_processing_throughput(self) -> dict[str, float]:
        row = await self.db.query(
            f"SELECT COUNT(*) AS total FROM {SHADOW_DOC_STATUS_TABLE} "
            "WHERE storage_workspace_id = $1 AND status = 'PROCESSED' "
            "AND updated_at >= $2",
            [self.workspace, _utcnow() - datetime.timedelta(hours=24)],
        )
        total = int(row["total"]) if row else 0
        return {
            "docs_per_minute": total / (24 * 60),
            "docs_per_hour": total / 24,
            "total_24h": total,
        }




@final
@dataclass
class ShadowNullKVStorage(BaseKVStorage):
    """Discarding KV store for generation namespaces that must not touch shared rows.

    Used for the LLM response cache of a shadow generation: the shared cache table
    is keyed by a global ``id`` and references KMS chunk rows, so writing it from
    a shadow build could overwrite rows used by the active generation.
    """

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return []

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return set(keys)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        return None

    async def delete(self, ids: list[str]) -> None:
        return None

    async def is_empty(self) -> bool:
        return True

    async def index_done_callback(self) -> None:
        return None

    async def drop(self) -> dict[str, str]:
        return {"status": "success", "message": "data dropped"}


@final
class ReadOnlyKVView:
    """Read-only delegate over a source-workspace KV store (e.g. ``full_docs``).

    A shadow generation re-reads the source documents of the *logical* workspace;
    any write attempt is a bug and fails loudly instead of altering source data.
    """

    def __init__(self, inner: BaseKVStorage) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def initialize(self) -> None:
        await self._inner.initialize()

    async def finalize(self) -> None:
        await self._inner.finalize()

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        return await self._inner.get_by_id(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return await self._inner.get_by_ids(ids)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return await self._inner.filter_keys(keys)

    async def is_empty(self) -> bool:
        return await self._inner.is_empty()

    async def index_done_callback(self) -> None:
        return None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        raise PermissionError("Shadow generation must not write source documents")

    async def delete(self, ids: list[str]) -> None:
        raise PermissionError("Shadow generation must not delete source documents")

    async def drop(self) -> dict[str, str]:
        raise PermissionError("Shadow generation must not drop source documents")


__all__ = [
    "PGShadowDocStatusStorage",
    "PGShadowTextChunksStorage",
    "ReadOnlyKVView",
    "ShadowNullKVStorage",
    "ensure_shadow_tables",
]
