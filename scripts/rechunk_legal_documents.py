#!/usr/bin/env python3
"""Re-run legal documents through the pipeline from the chunking stage.

Each document gets a fresh run ``<doc_id>-rechunk-<tag>`` so that events of the
original run still queued in Kafka keep hitting their own (finished) stage rows.
The chunking event carries ``force_rechunk`` so already PROCESSED documents are
not skipped; chunking then detaches the document's stale rows in
tbl_lightrag_document_chunk and removes their Milvus vectors (see ``_detach_stale_legal_chunks``).

Run inside a legal worker container, for example:

    python /app/scripts/rechunk_legal_documents.py --doc-ids 393779,901623 --apply
    python /app/scripts/rechunk_legal_documents.py --all --after-id 0 --limit 5000 --apply

Without --apply it only prints what it would publish. With --max-lag (default
20000) it pauses while the legal chunking + entity_extraction topics have more
unconsumed messages than that, so the backlog never outgrows Kafka retention.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import asyncpg  # noqa: E402
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer  # noqa: E402
from aiokafka.admin import AIOKafkaAdminClient  # noqa: E402

from kafka_worker.config import (  # noqa: E402
    KAFKA_BOOTSTRAP_SERVERS,
    PIPELINE_LEGAL_CHUNKING_TOPIC,
)
from kafka_worker.partitioned_publish import publish_pipeline_event  # noqa: E402
from lightrag.pipeline.models import PipelineStageEvent, StageName  # noqa: E402
from lightrag.pipeline.partitioning import create_partition_allocator_from_env  # noqa: E402

THROTTLE_TOPICS = (
    PIPELINE_LEGAL_CHUNKING_TOPIC,
    "lightrag.pipeline.legal.entity_extraction",
)


def build_rechunk_event(
    doc_id: str, workspace_id: str, template: dict | None, tag: str
) -> PipelineStageEvent:
    metadata = dict((template or {}).get("metadata") or {})
    for key in ("input_artifact_uris", "input_manifest_uris", "status"):
        metadata.pop(key, None)
    metadata.update(
        {"doc_type": "legal", "legal_document_id": str(doc_id), "force_rechunk": True}
    )
    return PipelineStageEvent(
        run_id=f"{doc_id}-rechunk-{tag}",
        workspace_id=str((template or {}).get("workspace_id") or workspace_id),
        doc_id=str(doc_id),
        doc_type="legal",
        stage=StageName.CHUNKING,
        metadata=metadata,
    )


async def _connect_pg() -> asyncpg.Connection:
    return await asyncpg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        database=os.environ["POSTGRES_DATABASE"],
    )


async def _select_doc_ids(conn: asyncpg.Connection, args: argparse.Namespace) -> list[int]:
    if args.doc_ids:
        return [int(x) for x in args.doc_ids.split(",") if x.strip()]
    if args.ids_file:
        return [int(x) for x in Path(args.ids_file).read_text().split() if x.strip()]
    rows = await conn.fetch(
        "SELECT id FROM tbl_legal_document WHERE process_status = 'PROCESSED' AND id > $1 "
        "ORDER BY id LIMIT $2",
        args.after_id,
        args.limit,
    )
    return [row["id"] for row in rows]


async def _load_targets(conn: asyncpg.Connection, doc_ids: list[int]) -> list[tuple[int, str, dict | None]]:
    rows = await conn.fetch(
        """
        SELECT ld.id, ld.workspace_id::text AS workspace_id, s.event_json
        FROM tbl_legal_document ld
        LEFT JOIN lightrag_pipeline_stage_runs s
          ON s.run_id = ld.id::text AND s.stage = 'chunking' AND s.shard_index = 0
        WHERE ld.id = ANY($1::int[])
        ORDER BY ld.id
        """,
        doc_ids,
    )
    targets = []
    for row in rows:
        template = row["event_json"]
        if isinstance(template, str):
            template = json.loads(template)
        targets.append((row["id"], row["workspace_id"], template))
    return targets


async def _topic_lag(admin: AIOKafkaAdminClient, consumer: AIOKafkaConsumer) -> int:
    total = 0
    for group_id, _protocol in await admin.list_consumer_groups():
        offsets = await admin.list_consumer_group_offsets(group_id)
        tracked = {tp: meta for tp, meta in offsets.items() if tp.topic in THROTTLE_TOPICS}
        if not tracked:
            continue
        ends = await consumer.end_offsets(list(tracked))
        total += sum(max(ends[tp] - meta.offset, 0) for tp, meta in tracked.items())
    return total


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--doc-ids", help="comma separated tbl_legal_document ids")
    source.add_argument("--ids-file", help="file with one tbl_legal_document id per line")
    source.add_argument("--all", action="store_true", help="every PROCESSED legal document")
    parser.add_argument("--after-id", type=int, default=0, help="--all: resume after this id")
    parser.add_argument("--limit", type=int, default=1000, help="--all: documents per invocation")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-lag", type=int, default=20000, help="0 disables throttling")
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    conn = await _connect_pg()
    try:
        targets = await _load_targets(conn, await _select_doc_ids(conn, args))
    finally:
        await conn.close()
    print(f"targets={len(targets)} apply={args.apply} tag={args.tag}")
    if not args.apply:
        for doc_id, workspace_id, template in targets[:20]:
            event = build_rechunk_event(str(doc_id), workspace_id, template, args.tag)
            print(f"would publish run_id={event.run_id} workspace_id={event.workspace_id} "
                  f"template={'yes' if template else 'no'}")
        return

    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    admin = AIOKafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    consumer = AIOKafkaConsumer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    allocator = create_partition_allocator_from_env()
    await producer.start()
    await admin.start()
    await consumer.start()
    try:
        for start in range(0, len(targets), args.batch_size):
            while args.max_lag and (lag := await _topic_lag(admin, consumer)) > args.max_lag:
                print(f"lag={lag} > {args.max_lag}; waiting 60s")
                await asyncio.sleep(60)
            batch = targets[start : start + args.batch_size]
            for doc_id, workspace_id, template in batch:
                event = build_rechunk_event(str(doc_id), workspace_id, template, args.tag)
                await publish_pipeline_event(
                    producer, allocator, PIPELINE_LEGAL_CHUNKING_TOPIC, event
                )
            print(f"published {start + len(batch)}/{len(targets)} last_id={batch[-1][0]}", flush=True)
    finally:
        await consumer.stop()
        await admin.close()
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
