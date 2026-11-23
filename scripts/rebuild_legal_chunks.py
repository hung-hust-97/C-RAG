#!/usr/bin/env python3
"""Rebuild the legal chunk layer offline, for a collection-swap cutover.

Steps (each resumable, run inside a container built from the kafka-worker image):

  create-collection  clone the legal_chunks schema and indexes into legal_chunks_v2
  chunk              chunk the documents of shard i/N with the pipeline's own chunking
                     stage and stage the resulting chunk dicts in S3
  embed              embed staged chunks with the production embedding function
  import             bulk-import embedded batches into legal_chunks_v2 and wait for completion
  cutover-pg         swap PostgreSQL chunk rows to the rebuilt chunks (insert / detach phases)
  parity             check that rows built here equal the rows MilvusVectorDBStorage builds
  status             progress summary
  reembed-general    re-embed general documents whose vectors are missing from general_chunks
                     (run with PIPELINE_DOC_TYPE=general)

Staging layout in s3://$S3_BUCKET/rechunk/<tag>/: chunks/<shard>/<batch>.jsonl.gz,
vectors/<shard>/<batch>.npy (float32, same row order), imported/<shard>/<batch>.json.
Except for cutover-pg and reembed-general, no step writes PostgreSQL or the live collections.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import json
import os
import re
import sys
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import aioboto3  # noqa: E402
import asyncpg  # noqa: E402
import numpy as np  # noqa: E402
from pymilvus import MilvusClient, utility  # noqa: E402

from kafka_worker.helper.helpers import get_initialized_rag_instance  # noqa: E402
from lightrag.base import DocStatus  # noqa: E402
from lightrag.kg.milvus_impl import _to_unix_timestamp, normalize_legal_document_status  # noqa: E402
from lightrag.pipeline.models import ArtifactRef, PipelineStageEvent, StageName  # noqa: E402
from lightrag.pipeline.stages.chunking import run_chunking_stage  # noqa: E402

SOURCE_COLLECTION = "legal_chunks"
TARGET_COLLECTION = os.environ.get("REBUILD_TARGET_COLLECTION", "legal_chunks_v2")
# Same extra keys MilvusVectorDBStorage._bulk_upsert_impl copies into each row.
ROW_EXTRA_KEYS = (
    "entity_id", "chunk_id", "status", "effective_from", "effective_to", "legal_status",
    "legal_status_code", "clause_status", "effective_date", "status_date", "issuance_date",
)
MAX_DYN_SIZE = 60000
IMPORT_DONE, IMPORT_FAILED = 6, (7, 8)


# ---------------------------------------------------------------- helpers

def clean_for_embedding(text: str) -> str:
    """Same text MilvusVectorDBStorage embeds: tags replaced by spaces, whitespace collapsed."""
    if not isinstance(text, str):
        return text
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def _cap_dynamic_fields(meta: dict[str, Any]) -> None:
    """Mirror of the 60KB dynamic-field cap in MilvusVectorDBStorage._bulk_upsert_impl."""
    for _ in range(5):
        try:
            encoded_size = len(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
            if encoded_size <= MAX_DYN_SIZE:
                break
        except Exception:
            pass
        longest_key, max_len = None, -1
        for key, value in meta.items():
            if isinstance(value, str):
                value_len = len(value.encode("utf-8"))
                if value_len > max_len:
                    max_len, longest_key = value_len, key
        if longest_key is None or max_len < 500:
            for key, value in meta.items():
                if isinstance(value, (list, dict)):
                    meta[key] = json.dumps(value, ensure_ascii=False)[:2000]
            continue
        excess = encoded_size - MAX_DYN_SIZE
        target_len = max(100, max_len - excess - 500)
        meta[longest_key] = meta[longest_key].encode("utf-8")[:target_len].decode("utf-8", errors="ignore")


def build_milvus_row(chunk_id: str, chunk: dict, workspace_id: str, meta_fields, vector, now: int) -> dict:
    """Row for legal_chunks exactly as MilvusVectorDBStorage._bulk_upsert_impl builds it."""
    row: dict[str, Any] = {"id": chunk_id, "created_at": now, "workspace_id": workspace_id}
    meta = {k: v for k, v in chunk.items() if k in meta_fields or k in ROW_EXTRA_KEYS}
    _cap_dynamic_fields(meta)
    row.update(meta)
    row["vector"] = vector
    if not row.get("workspace_id"):
        row["workspace_id"] = workspace_id
    row.pop("doc_id", None)
    row["chunk_id"] = row.get("id")
    status = row.get("status") or row.get("clause_status") or row.get("legal_status_code") or row.get("legal_status")
    if not status:
        status = "UNKNOWN"
    elif isinstance(status, str) and status.strip():
        status = normalize_legal_document_status(status).get("legal_status_code") or status
    row["status"] = status
    row["effective_from"] = _to_unix_timestamp(row.get("effective_from") or row.get("effective_date"))
    row["effective_to"] = _to_unix_timestamp(row.get("effective_to") or row.get("status_date"))
    return row


class _CaptureTextChunks:
    def __init__(self):
        self.captured: dict[str, dict] = {}

    async def upsert(self, data):
        self.captured.update(data)


class _DocStatusProxy:
    """Real DB for metadata lookups; no status reads/writes (documents are PROCESSED)."""

    def __init__(self, real):
        self.db = getattr(real, "db", None)

    async def get_by_id(self, _doc_id):
        return None

    async def upsert(self, _data):
        return None


class _RagProxy:
    def __init__(self, real):
        self._real = real
        self.text_chunks = _CaptureTextChunks()
        self.doc_status = _DocStatusProxy(real.doc_status)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _NullArtifacts:
    async def write_jsonl(self, **_kw):
        return ArtifactRef(uri="memory://chunks", kind="chunks")

    async def write_manifest(self, **_kw):
        return ArtifactRef(uri="memory://manifest", kind="manifest")


class _CaptureContext:
    debug_sync = False

    def __init__(self):
        self.artifact_store = _NullArtifacts()
        self._rags: dict[str, Any] = {}
        self.last_proxy: _RagProxy | None = None

    async def get_rag(self, workspace_id, profile=None, **_kw):
        if workspace_id not in self._rags:
            self._rags[workspace_id] = await get_initialized_rag_instance(workspace_id, StageName.CHUNKING)
        self.last_proxy = _RagProxy(self._rags[workspace_id])
        return self.last_proxy


async def capture_chunks(context: _CaptureContext, doc: dict, tag: str) -> dict[str, dict]:
    template = doc.get("meta") or {}
    if isinstance(template, str):
        template = json.loads(template)
    metadata = {k: v for k, v in template.items() if k not in ("status", "input_artifact_uris", "input_manifest_uris")}
    metadata.update({
        "doc_type": "legal",
        "legal_document_id": str(doc["id"]),
        "content_path": doc["content_path"],
        "document_name": metadata.get("document_name") or doc["title"],
    })
    event = PipelineStageEvent(
        run_id=f"{doc['id']}-rebuild-{tag}", workspace_id=doc["workspace_id"], doc_id=str(doc["id"]),
        doc_type="legal", stage=StageName.CHUNKING, metadata=metadata,
    )
    await run_chunking_stage(event, context)
    return context.last_proxy.text_chunks.captured


def _s3():
    return aioboto3.Session().client(
        "s3", endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY") or os.environ.get("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY") or os.environ.get("S3_SECRET_ACCESS_KEY"),
    )


async def _list_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys = []
    async for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


async def _pg() -> asyncpg.Connection:
    return await asyncpg.connect(
        host=os.environ["POSTGRES_HOST"], port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ["POSTGRES_USER"], password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ["POSTGRES_DATABASE"],
    )


def _milvus() -> MilvusClient:
    return MilvusClient(uri=os.environ["MILVUS_URI"], db_name=os.environ.get("MILVUS_DB_NAME", "default"))


# ---------------------------------------------------------------- steps

def create_collection(args) -> None:
    client = _milvus()
    if client.has_collection(TARGET_COLLECTION):
        if not getattr(args, "recreate", False):
            print(f"{TARGET_COLLECTION} already exists")
            return
        row_count = int(client.get_collection_stats(TARGET_COLLECTION).get("row_count", 0))
        if row_count > 5000:
            print(f"refusing to drop {TARGET_COLLECTION}: row_count={row_count} > 5000")
            return
        client.drop_collection(TARGET_COLLECTION)
        print(f"dropped {TARGET_COLLECTION} (row_count={row_count})")
    desc = client.describe_collection(SOURCE_COLLECTION)
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=desc.get("enable_dynamic_field", True))
    for field in desc["fields"]:
        kwargs = dict(field.get("params") or {})
        for flag in ("is_primary", "is_partition_key", "nullable"):
            if field.get(flag):
                kwargs[flag] = True
        schema.add_field(field["name"], field["type"], **kwargs)
    create_kwargs = {}
    if desc.get("num_partitions"):
        create_kwargs["num_partitions"] = desc["num_partitions"]
    if desc.get("properties"):
        create_kwargs["properties"] = desc["properties"]
    client.create_collection(TARGET_COLLECTION, schema=schema, **create_kwargs)
    index_params = client.prepare_index_params()
    for name in client.list_indexes(SOURCE_COLLECTION):
        idx = client.describe_index(SOURCE_COLLECTION, name)
        params = {k: v for k, v in idx.items() if k not in (
            "index_type", "metric_type", "field_name", "index_name", "total_rows", "indexed_rows",
            "pending_index_rows", "state")}
        add = {"field_name": idx["field_name"], "index_type": idx["index_type"], "index_name": name, "params": params}
        if idx.get("metric_type"):
            add["metric_type"] = idx["metric_type"]
        index_params.add_index(**add)
    client.create_index(TARGET_COLLECTION, index_params)
    print(f"created {TARGET_COLLECTION}: {[f['name'] for f in desc['fields']]} indexes={client.list_indexes(TARGET_COLLECTION)}")


async def chunk(args) -> None:
    state_dir = Path(args.state_dir) / args.tag / f"chunk-{args.shard:03d}"
    state_dir.mkdir(parents=True, exist_ok=True)
    done_file = state_dir / "done_ids.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    conn = await _pg()
    try:
        docs = await conn.fetch(
            """
            SELECT ld.id, ld.workspace_id::text AS workspace_id, ld.title, ld.content_path, ld.status,
                   s.event_json->'metadata' AS meta
            FROM tbl_legal_document ld
            LEFT JOIN lightrag_pipeline_stage_runs s
              ON s.run_id = ld.id::text AND s.stage = 'chunking' AND s.shard_index = 0
            WHERE ld.content_path IS NOT NULL AND ld.workspace_id IS NOT NULL
              AND ld.process_status IN ('PROCESSED', 'CHUNKING', 'FAILED')
              AND ld.id % $1 = $2
            ORDER BY (ld.status = 'VALID') DESC, ld.id
            """,
            args.shards, args.shard,
        )
    finally:
        await conn.close()
    failed_file = state_dir / "failed.tsv"
    if args.retry_failed:
        if not failed_file.exists():
            print(f"shard={args.shard}/{args.shards} no failed.tsv to retry", flush=True)
            return
        retry_ids = {
            line.split("\t", 1)[0].strip()
            for line in failed_file.read_text().splitlines()
            if line.strip()
        }
        failed_file.rename(state_dir / f"failed.tsv.retried-{int(time.time())}")
        todo = [dict(d) for d in docs if str(d["id"]) in retry_ids][: args.limit or None]
    else:
        todo = [dict(d) for d in docs if str(d["id"]) not in done][: args.limit or None]
    print(f"shard={args.shard}/{args.shards} docs={len(docs)} done={len(done)} todo={len(todo)}", flush=True)
    context = _CaptureContext()
    bucket, prefix = os.environ["S3_BUCKET"], f"rechunk/{args.tag}"
    async with _s3() as s3:
        for start in range(0, len(todo), args.batch_docs):
            batch, lines, failed, t0 = todo[start: start + args.batch_docs], [], [], time.time()
            for doc in batch:
                try:
                    chunks = await capture_chunks(context, doc, args.tag)
                    milvus_workspace = context._rags[doc["workspace_id"]].chunks_vdb.workspace
                    lines.append(json.dumps({"doc_id": doc["id"], "workspace_id": doc["workspace_id"],
                                             "milvus_workspace": milvus_workspace, "status": doc["status"],
                                             "chunks": chunks}, ensure_ascii=False, default=str))
                except Exception as exc:  # keep going; failures are listed for a retry pass
                    failed.append(f"{doc['id']}\t{type(exc).__name__}: {str(exc)[:200]}")
            key = f"{prefix}/chunks/{args.shard:03d}/{batch[0]['id']:08d}-{uuid.uuid4().hex[:8]}.jsonl.gz"
            if lines:
                await s3.put_object(Bucket=bucket, Key=key, Body=gzip.compress("\n".join(lines).encode("utf-8")))
            with done_file.open("a") as fh:
                fh.write("".join(f"{d['id']}\n" for d in batch))
            if failed:
                with (state_dir / "failed.tsv").open("a") as fh:
                    fh.write("\n".join(failed) + "\n")
            print(f"shard={args.shard} batch_docs={len(batch)} ok={len(lines)} failed={len(failed)} "
                  f"secs={time.time() - t0:.0f} progress={start + len(batch)}/{len(todo)}", flush=True)


async def embed(args) -> None:
    bucket, prefix = os.environ["S3_BUCKET"], f"rechunk/{args.tag}"
    rag = await get_initialized_rag_instance(args.embed_workspace, StageName.CHUNK_EMBEDDING)
    sem = asyncio.Semaphore(args.concurrency)

    async def embed_batch(texts):
        async with sem:
            for attempt in range(5):
                try:
                    return np.asarray(await rag.embedding_func(texts), dtype=np.float32)
                except Exception as exc:
                    if attempt == 4:
                        raise
                    print(f"embed retry {attempt + 1}: {exc}", flush=True)
                    await asyncio.sleep(10 * (attempt + 1))

    async with _s3() as s3:
        while True:
            chunk_keys = sorted(await _list_keys(s3, bucket, f"{prefix}/chunks/"))
            done = {k.replace("/vectors/", "/chunks/").replace(".npy", ".jsonl.gz")
                    for k in await _list_keys(s3, bucket, f"{prefix}/vectors/")}
            pending = [k for k in chunk_keys if k not in done
                       and zlib.crc32(k.encode()) % args.workers == args.worker]
            if not pending:
                if args.follow:
                    await asyncio.sleep(60)
                    continue
                break
            for key in pending:
                t0 = time.time()
                body = await (await s3.get_object(Bucket=bucket, Key=key))["Body"].read()
                texts = []
                for line in gzip.decompress(body).decode("utf-8").split("\n"):
                    texts.extend(clean_for_embedding(c["content"]) for c in json.loads(line)["chunks"].values())
                parts = await asyncio.gather(*[embed_batch(texts[i: i + args.batch_size])
                                               for i in range(0, len(texts), args.batch_size)])
                vectors = np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)
                buf = io.BytesIO()
                np.save(buf, vectors)
                await s3.put_object(Bucket=bucket, Key=key.replace("/chunks/", "/vectors/").replace(".jsonl.gz", ".npy"),
                                    Body=buf.getvalue())
                dt = time.time() - t0
                print(f"embedded {key.rsplit('/', 1)[-1]} chunks={len(texts)} secs={dt:.1f} "
                      f"chunks_per_s={len(texts) / max(dt, 1e-6):.1f}", flush=True)


async def import_batches(args) -> None:
    bucket, prefix = os.environ["S3_BUCKET"], f"rechunk/{args.tag}"
    import_bucket = os.environ.get("MILVUS_BULK_INSERT_BUCKET", "milvus-vectors")
    rag = await get_initialized_rag_instance(args.embed_workspace, StageName.CHUNK_EMBEDDING)
    vdb = rag.chunks_vdb
    meta_fields = set(vdb.meta_fields)
    alias = f"rebuild_{uuid.uuid4().hex[:8]}"
    vdb._connect_bulk_insert_alias(alias)
    imported_docs_file = Path(args.state_dir) / args.tag / "imported_docs.txt"
    imported_docs_file.parent.mkdir(parents=True, exist_ok=True)
    imported_docs = set(imported_docs_file.read_text().split()) if imported_docs_file.exists() else set()
    async with _s3() as s3:
        while True:
            vector_keys = sorted(await _list_keys(s3, bucket, f"{prefix}/vectors/"))
            imported = {k.replace("/imported/", "/vectors/").replace(".json", ".npy")
                        for k in await _list_keys(s3, bucket, f"{prefix}/imported/")}
            pending = [k for k in vector_keys if k not in imported][: args.limit or None]
            if not pending:
                if args.follow:
                    await asyncio.sleep(120)
                    continue
                break
            print(f"import pending={len(pending)} into {TARGET_COLLECTION}", flush=True)
            for vkey in pending:
                t0 = time.time()
                ckey = vkey.replace("/vectors/", "/chunks/").replace(".npy", ".jsonl.gz")
                vectors = np.load(io.BytesIO(await (await s3.get_object(Bucket=bucket, Key=vkey))["Body"].read()))
                body = await (await s3.get_object(Bucket=bucket, Key=ckey))["Body"].read()
                rows, batch_doc_ids, now, i = [], [], int(time.time()), 0
                seen_in_batch: set[str] = set()
                for line in gzip.decompress(body).decode("utf-8").split("\n"):
                    if not line:
                        continue
                    doc = json.loads(line)
                    doc_id_str = str(doc["doc_id"])
                    doc_chunks = doc["chunks"]
                    if doc_id_str in imported_docs or doc_id_str in seen_in_batch:
                        i += len(doc_chunks)
                        continue
                    seen_in_batch.add(doc_id_str)
                    batch_doc_ids.append(doc_id_str)
                    for chunk_id, chunk_data in doc_chunks.items():
                        rows.append(build_milvus_row(chunk_id, chunk_data, doc["milvus_workspace"], meta_fields,
                                                     vectors[i].tolist(), now))
                        i += 1
                if i != len(vectors):
                    raise RuntimeError(f"vector/chunk count mismatch for {vkey}: {len(vectors)} vs {i}")
                if not rows:
                    await s3.put_object(
                        Bucket=bucket,
                        Key=vkey.replace("/vectors/", "/imported/").replace(".npy", ".json"),
                        Body=json.dumps({"task_id": None, "rows": 0}).encode(),
                    )
                    print(f"imported {vkey.rsplit('/', 1)[-1]} rows=0/0 (skipped) secs={time.time() - t0:.0f}",
                          flush=True)
                    continue
                object_key = f"rechunk_import/{args.tag}/{uuid.uuid4().hex}.json"
                await s3.put_object(Bucket=import_bucket, Key=object_key, Body=json.dumps({"rows": rows}).encode("utf-8"))
                task_id = await asyncio.to_thread(vdb._do_bulk_insert, TARGET_COLLECTION, [object_key], alias)
                while True:
                    state = await asyncio.to_thread(utility.get_bulk_insert_state, task_id=task_id, using=alias)
                    if state.state in (IMPORT_DONE, *IMPORT_FAILED):
                        break
                    await asyncio.sleep(5)
                await s3.delete_object(Bucket=import_bucket, Key=object_key)
                if state.state != IMPORT_DONE:
                    print(f"IMPORT FAILED {vkey} task={task_id} reason={state.failed_reason}", flush=True)
                    continue
                imported_docs.update(batch_doc_ids)
                with imported_docs_file.open("a") as fh:
                    fh.write("".join(f"{doc_id}\n" for doc_id in batch_doc_ids))
                await s3.put_object(Bucket=bucket, Key=vkey.replace("/vectors/", "/imported/").replace(".npy", ".json"),
                                    Body=json.dumps({"task_id": task_id, "rows": state.row_count}).encode())
                print(f"imported {vkey.rsplit('/', 1)[-1]} rows={state.row_count}/{len(rows)} secs={time.time() - t0:.0f}",
                      flush=True)
            if not args.follow:
                break


async def cutover_pg(args) -> None:
    """Swap PostgreSQL chunk rows to the rebuilt chunks, per document.

    phase insert: upsert the new chunk rows (the pipeline's own text_chunks.upsert); run
                  before the Milvus collection swap so every new vector already has its row.
    phase detach: set legal_document_id = NULL on the document's other (old) rows; run right
                  after the swap. Detached rows drop out of KMS search (it joins on
                  legal_document_id) but keep their ids, content and KG links, so no
                  tbl_chunk_entity/tbl_chunk_relation rows are deleted and no entity turns orphan.
    """
    bucket, prefix = os.environ["S3_BUCKET"], f"rechunk/{args.tag}"
    state_dir = Path(args.state_dir) / args.tag / f"cutover-{args.phase}-{args.shard:03d}"
    state_dir.mkdir(parents=True, exist_ok=True)
    done_file = state_dir / "done_files.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    context = _CaptureContext()
    pool = await asyncpg.create_pool(
        host=os.environ["POSTGRES_HOST"], port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ["POSTGRES_USER"], password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ["POSTGRES_DATABASE"], min_size=1, max_size=args.concurrency)
    sem, rag_lock, detach_log_lock = asyncio.Semaphore(args.concurrency), asyncio.Lock(), asyncio.Lock()
    detach_log_file = state_dir.parent / f"detached-{args.shard:03d}.jsonl"
    totals = {"docs": 0, "rows": 0, "detached": 0}

    async def one_doc(doc: dict) -> None:
        async with sem:
            if args.phase == "insert":
                if args.apply:
                    async with rag_lock:
                        if doc["workspace_id"] not in context._rags:
                            await context.get_rag(doc["workspace_id"])
                    await context._rags[doc["workspace_id"]].text_chunks.upsert(doc["chunks"])
                totals["rows"] += len(doc["chunks"])
            else:
                keep = [chunk_id.removeprefix("chunk-") for chunk_id in doc["chunks"]]
                sql_where = "legal_document_id = $1 AND NOT (id = ANY($2::uuid[]))"
                async with pool.acquire() as conn:
                    if args.apply:
                        detached_rows = await conn.fetch(
                            f"SELECT id::text FROM tbl_lightrag_document_chunk WHERE {sql_where}",
                            int(doc["doc_id"]), keep)
                        detached_ids = [r["id"] for r in detached_rows]
                        if detached_ids:
                            async with detach_log_lock:
                                with detach_log_file.open("a") as fh:
                                    fh.write(json.dumps({"doc_id": doc["doc_id"], "detached_ids": detached_ids}) + "\n")
                        result = await conn.execute(
                            f"UPDATE tbl_lightrag_document_chunk SET legal_document_id = NULL, "
                            f"update_time = NOW() WHERE {sql_where}", int(doc["doc_id"]), keep)
                        totals["detached"] += int(result.split()[-1])
                    else:
                        totals["detached"] += await conn.fetchval(
                            f"SELECT count(*) FROM tbl_lightrag_document_chunk WHERE {sql_where}",
                            int(doc["doc_id"]), keep)
            totals["docs"] += 1

    try:
        async with _s3() as s3:
            keys = sorted(k for k in await _list_keys(s3, bucket, f"{prefix}/imported/")
                          if zlib.crc32(k.encode()) % args.shards == args.shard)
            chunk_keys = [k.replace("/imported/", "/chunks/").replace(".json", ".jsonl.gz") for k in keys]
            pending = [k for k in chunk_keys if k not in done][: args.limit or None]
            print(f"cutover-pg phase={args.phase} apply={args.apply} shard={args.shard}/{args.shards} "
                  f"imported_batches={len(chunk_keys)} pending={len(pending)}", flush=True)
            for key in pending:
                body = await (await s3.get_object(Bucket=bucket, Key=key))["Body"].read()
                docs = [json.loads(line) for line in gzip.decompress(body).decode("utf-8").split("\n") if line]
                await asyncio.gather(*[one_doc(doc) for doc in docs])
                if args.apply:
                    with done_file.open("a") as fh:
                        fh.write(key + "\n")
                print(f"{args.phase} {key.rsplit('/', 1)[-1]} totals={totals}", flush=True)
    finally:
        await pool.close()


async def reembed_general(args) -> None:
    """Re-embed general documents whose chunk vectors are missing from general_chunks.

    Chunks are rebuilt with the pipeline's own chunking stage in capture mode (nothing is
    written to PostgreSQL or S3), must match the chunk ids already in
    tbl_lightrag_document_chunk, and are written with rag.chunks_vdb.upsert exactly like the
    chunk_embedding stage. Documents left FAILED although every stage finished are set PROCESSED.
    """
    if os.environ.get("PIPELINE_DOC_TYPE") != "general":
        raise SystemExit("reembed-general must run with PIPELINE_DOC_TYPE=general")
    conn = await _pg()
    try:
        docs = [dict(d) for d in await conn.fetch(
            """
            SELECT d.document_id::text AS doc_id, d.status::text AS status, d.document_name, s.event_json,
                   ARRAY(SELECT c.id::text FROM tbl_lightrag_document_chunk c
                         WHERE c.document_id = d.document_id) AS pg_ids,
                   (SELECT bool_and(r.status = 'done') FROM lightrag_pipeline_stage_runs r
                     WHERE r.run_id = d.document_id::text) AS all_stages_done
            FROM tbl_document d
            LEFT JOIN lightrag_pipeline_stage_runs s
              ON s.run_id = d.document_id::text AND s.stage = 'chunking' AND s.shard_index = 0
            WHERE EXISTS (SELECT 1 FROM tbl_lightrag_document_chunk c WHERE c.document_id = d.document_id)
            ORDER BY d.update_time
            """)]
    finally:
        await conn.close()
    client = _milvus()

    def milvus_ids(ids) -> set[str]:
        if not ids:
            return set()
        expr = "id in [" + ",".join(f'"{i}"' for i in ids) + "]"
        rows = client.query("general_chunks", filter=expr, output_fields=["id"], consistency_level="Strong")
        return {r["id"] for r in rows}

    context = _CaptureContext()
    totals = dict(docs=len(docs), skipped=0, mismatched=0, to_write=0, written=0, verified=0, status_fixed=0)
    for doc in docs:
        pg_ids = set(doc["pg_ids"] or [])
        missing = len(pg_ids) - len(milvus_ids([f"chunk-{i}" for i in pg_ids]))
        if missing == 0 and doc["status"] != "FAILED":
            totals["skipped"] += 1
            continue
        if not doc["event_json"]:
            print(f"NO TEMPLATE doc={doc['doc_id']} name={doc['document_name']!r}", flush=True)
            totals["mismatched"] += 1
            continue
        template = doc["event_json"] if isinstance(doc["event_json"], dict) else json.loads(doc["event_json"])
        metadata = {k: v for k, v in (template.get("metadata") or {}).items()
                    if k not in ("status", "input_artifact_uris", "input_manifest_uris")}
        workspace_id = template["workspace_id"]
        event = PipelineStageEvent(
            run_id=f"{doc['doc_id']}-reembed", workspace_id=workspace_id, doc_id=doc["doc_id"],
            doc_type="general", stage=StageName.CHUNKING, metadata=metadata,
        )
        await run_chunking_stage(event, context)
        chunks = context.last_proxy.text_chunks.captured
        new_ids = {chunk_id.removeprefix("chunk-") for chunk_id in chunks}
        if new_ids != pg_ids:
            print(f"ID MISMATCH doc={doc['doc_id']} name={doc['document_name']!r} pg={len(pg_ids)} "
                  f"rebuilt={len(new_ids)} common={len(pg_ids & new_ids)}", flush=True)
            totals["mismatched"] += 1
            continue
        rag = context._rags[workspace_id]
        if rag.chunks_vdb.final_namespace != "general_chunks":
            raise SystemExit(f"refusing to write: chunks_vdb targets {rag.chunks_vdb.final_namespace}")
        totals["to_write"] += len(chunks)
        print(f"doc={doc['doc_id']} status={doc['status']} chunks={len(chunks)} missing={missing} "
              f"all_done={doc['all_stages_done']} name={doc['document_name']!r}", flush=True)
        if not args.apply:
            continue
        await rag.chunks_vdb.upsert(chunks)
        await rag.chunks_vdb.index_done_callback()
        totals["written"] += len(chunks)
        got = milvus_ids(list(chunks))
        if len(got) != len(chunks):
            print(f"VERIFY FAILED doc={doc['doc_id']} {len(got)}/{len(chunks)}", flush=True)
            continue
        totals["verified"] += 1
        if doc["status"] == "FAILED" and doc["all_stages_done"]:
            await rag.doc_status.upsert({doc["doc_id"]: {
                "status": DocStatus.PROCESSED, "error_msg": "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }})
            totals["status_fixed"] += 1
    print("TOTALS", totals, flush=True)


class _StopBeforeImport(Exception):
    pass


async def parity(args) -> None:
    """Build rows both ways for a few documents and compare (S3 upload intercepted)."""
    from unittest.mock import patch

    conn = await _pg()
    docs = [dict(d) for d in await conn.fetch(
        "SELECT id, workspace_id::text AS workspace_id, title, content_path, status, NULL::jsonb AS meta "
        "FROM tbl_legal_document WHERE process_status='PROCESSED' AND content_path IS NOT NULL "
        "AND workspace_id IS NOT NULL ORDER BY random() LIMIT $1", args.limit or 3)]
    await conn.close()
    context = _CaptureContext()
    mismatches = 0
    for doc in docs:
        chunks = await capture_chunks(context, doc, "parity")
        vdb = context._rags[doc["workspace_id"]].chunks_vdb
        fake_vectors = np.random.rand(len(chunks), 4).astype(np.float32)
        captured = {}

        class _FakeS3:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def put_object(self, Bucket, Key, Body):
                captured["rows"] = json.loads(Body)["rows"]
                raise _StopBeforeImport()

        async def _fake_embed(batch, **_kw):
            start = captured.setdefault("offset", 0)
            captured["offset"] = start + len(batch)
            return fake_vectors[start: start + len(batch)]

        with patch("aioboto3.Session") as session, patch.object(vdb, "embedding_func", _fake_embed), \
                patch.dict(os.environ, {"S3_ENDPOINT_URL": "x", "S3_ACCESS_KEY": "x", "S3_SECRET_KEY": "x"}):
            session.return_value.client.return_value = _FakeS3()
            try:
                await vdb._bulk_upsert_impl(chunks)
            except _StopBeforeImport:
                pass
        theirs = {r["id"]: r for r in captured["rows"]}
        now = next(iter(theirs.values()))["created_at"] if theirs else 0
        for i, (chunk_id, chunk_data) in enumerate(chunks.items()):
            ours = build_milvus_row(chunk_id, chunk_data, vdb.workspace, set(vdb.meta_fields),
                                    fake_vectors[i].tolist(), now)
            if ours != theirs.get(chunk_id):
                mismatches += 1
                diff = {k for k in set(ours) | set(theirs.get(chunk_id, {})) if ours.get(k) != theirs.get(chunk_id, {}).get(k)}
                print(f"MISMATCH doc={doc['id']} chunk={chunk_id} keys={sorted(diff)}")
        print(f"doc={doc['id']} chunks={len(chunks)} compared")
    print("PARITY OK" if mismatches == 0 else f"PARITY FAILED mismatches={mismatches}")


async def status(args) -> None:
    bucket, prefix = os.environ["S3_BUCKET"], f"rechunk/{args.tag}"
    async with _s3() as s3:
        counts = {part: len(await _list_keys(s3, bucket, f"{prefix}/{part}/")) for part in ("chunks", "vectors", "imported")}
    state = Path(args.state_dir) / args.tag
    done_docs = sum(len(p.read_text().split()) for p in state.glob("chunk-*/done_ids.txt"))
    failed = sum(len(p.read_text().splitlines()) for p in state.glob("chunk-*/failed.tsv"))
    print(f"batches={counts} docs_chunked={done_docs} docs_failed={failed}")
    client = _milvus()
    if client.has_collection(TARGET_COLLECTION):
        print(TARGET_COLLECTION, client.get_collection_stats(TARGET_COLLECTION))


async def skip_legal_kg(args) -> None:
    where_sql = """
        WHERE stage IN ('entity_extraction','kg_merge','kg_embedding')
          AND status IN ('pending','running')
          AND workspace_id IN (SELECT DISTINCT workspace_id::text FROM tbl_legal_document WHERE workspace_id IS NOT NULL)
    """
    conn = await _pg()
    try:
        if not args.apply:
            rows = await conn.fetch(
                f"SELECT stage, status, count(*) FROM lightrag_pipeline_stage_runs {where_sql} GROUP BY 1,2 ORDER BY 1,2"
            )
            for r in rows:
                print(f"stage={r['stage']} status={r['status']} count={r['count']}", flush=True)
            print(f"DRY RUN total={sum(r['count'] for r in rows)}", flush=True)
            return
        result = await conn.execute(
            f"""
            UPDATE lightrag_pipeline_stage_runs
            SET status='failed', updated_at=NOW(),
                error_json='{{"error_type":"LegalKgDisabled","message":"skipped: legal KG disabled (recovery 2026-11)"}}'::jsonb
            {where_sql}
            """
        )
        print(f"UPDATED rows={result.split()[-1]}", flush=True)
    finally:
        await conn.close()


async def mark_processed(args) -> None:
    bucket, prefix = os.environ["S3_BUCKET"], f"rechunk/{args.tag}"
    doc_ids: set[int] = set()
    async with _s3() as s3:
        imported_keys = sorted(await _list_keys(s3, bucket, f"{prefix}/imported/"))
        for ikey in imported_keys:
            ckey = ikey.replace("/imported/", "/chunks/").replace(".json", ".jsonl.gz")
            body = await (await s3.get_object(Bucket=bucket, Key=ckey))["Body"].read()
            for line in gzip.decompress(body).decode("utf-8").split("\n"):
                if line:
                    doc_ids.add(int(json.loads(line)["doc_id"]))
    sorted_ids = sorted(doc_ids)
    print(f"imported_batches={len(imported_keys)} imported_docs={len(sorted_ids)} apply={args.apply}", flush=True)
    if not sorted_ids:
        print("TOTAL updated=0" if args.apply else "DRY RUN status_counts={} would_change=0", flush=True)
        return
    conn = await _pg()
    try:
        if not args.apply:
            counts: dict[str, int] = {}
            would_change = 0
            for start in range(0, len(sorted_ids), 5000):
                batch = sorted_ids[start: start + 5000]
                for r in await conn.fetch(
                    "SELECT process_status::text AS status, count(*) FROM tbl_legal_document "
                    "WHERE id = ANY($1::int[]) GROUP BY 1",
                    batch,
                ):
                    counts[r["status"]] = counts.get(r["status"], 0) + r["count"]
                would_change += await conn.fetchval(
                    "SELECT count(*) FROM tbl_legal_document d "
                    "WHERE d.id = ANY($1::int[]) AND d.process_status IN ('CHUNKING','FAILED') "
                    "  AND EXISTS (SELECT 1 FROM tbl_lightrag_document_chunk c WHERE c.legal_document_id = d.id)",
                    batch,
                )
            print(f"DRY RUN status_counts={counts} would_change={would_change}", flush=True)
            return
        total_updated = 0
        for start in range(0, len(sorted_ids), 5000):
            batch = sorted_ids[start: start + 5000]
            result = await conn.execute(
                """
                UPDATE tbl_legal_document d SET process_status='PROCESSED', updated_at=NOW()
                WHERE d.id = ANY($1::int[]) AND d.process_status IN ('CHUNKING','FAILED')
                  AND EXISTS (SELECT 1 FROM tbl_lightrag_document_chunk c WHERE c.legal_document_id = d.id)
                """,
                batch,
            )
            total_updated += int(result.split()[-1])
        print(f"TOTAL updated={total_updated}", flush=True)
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["create-collection", "chunk", "embed", "import", "cutover-pg",
                                         "parity", "status", "reembed-general", "skip-legal-kg", "mark-processed"])
    parser.add_argument("--recreate", action="store_true", help="create-collection: drop existing if <= 5000 rows")
    parser.add_argument("--retry-failed", action="store_true", help="chunk: retry doc ids from failed.tsv")
    parser.add_argument("--phase", choices=["insert", "detach"], default="insert", help="cutover-pg phase")
    parser.add_argument("--apply", action="store_true", help="cutover-pg: write (default is a dry run)")
    parser.add_argument("--tag", default="20261123")
    parser.add_argument("--state-dir", default="/app/data/rag_storage/rechunk_state")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch-docs", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32, help="embed: texts per request")
    parser.add_argument("--concurrency", type=int, default=4, help="embed: concurrent requests")
    parser.add_argument("--follow", action="store_true", help="embed: keep polling for new chunk batches")
    parser.add_argument("--worker", type=int, default=0, help="embed: this worker's index")
    parser.add_argument("--workers", type=int, default=1, help="embed: number of embed workers")
    parser.add_argument("--embed-workspace", default=os.environ.get("REBUILD_EMBED_WORKSPACE", ""))
    args = parser.parse_args()
    if args.step == "create-collection":
        create_collection(args)
        return
    asyncio.run({"chunk": chunk, "embed": embed, "import": import_batches, "cutover-pg": cutover_pg,
                 "parity": parity, "status": status, "reembed-general": reembed_general,
                 "skip-legal-kg": skip_legal_kg, "mark-processed": mark_processed}[args.step](args))


if __name__ == "__main__":
    main()
