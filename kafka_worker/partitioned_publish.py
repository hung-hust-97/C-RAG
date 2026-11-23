from __future__ import annotations

import zlib

from aiokafka import AIOKafkaProducer

from lightrag.pipeline.models import PipelineStageEvent, StageName
from lightrag.pipeline.partitioning import PipelinePartitionAllocator
from lightrag.utils import logger

# kg_merge serialises graph writes with an in-process keyed lock, which cannot
# protect two replicas merging the same entity. Pinning a workspace to one
# partition keeps all merges of that workspace inside a single consumer.
_WORKSPACE_PINNED_STAGES = {StageName.KG_MERGE}


async def _partition_count(producer: AIOKafkaProducer, topic: str) -> int:
    partitions = await producer.partitions_for(topic)
    if not partitions:
        raise RuntimeError(f"Kafka topic has no available partitions: {topic}")
    return len(partitions)


def workspace_partition(workspace_id: str, partition_count: int) -> int:
    return zlib.crc32(str(workspace_id).encode("utf-8")) % partition_count


async def publish_pipeline_event(
    producer: AIOKafkaProducer,
    allocator: PipelinePartitionAllocator,
    topic: str,
    event: PipelineStageEvent,
) -> int:
    partition_count = await _partition_count(producer, topic)
    if event.stage_name in _WORKSPACE_PINNED_STAGES:
        partition = workspace_partition(event.workspace_id, partition_count)
    else:
        partition = await allocator.allocate(topic, partition_count)
    await producer.send_and_wait(
        topic,
        event.model_dump_json(by_alias=True).encode("utf-8"),
        key=event.partition_key().encode("utf-8"),
        partition=partition,
    )
    logger.info(
        "[Pipeline] Published event topic=%s partition=%s/%s key=%s: %s",
        topic,
        partition,
        partition_count,
        event.partition_key(),
        event.log_context(),
    )
    return partition
