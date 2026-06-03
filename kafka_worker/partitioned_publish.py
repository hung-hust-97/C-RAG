from __future__ import annotations

from aiokafka import AIOKafkaProducer

from lightrag.pipeline.models import PipelineStageEvent
from lightrag.pipeline.partitioning import PipelinePartitionAllocator
from lightrag.utils import logger


async def _partition_count(producer: AIOKafkaProducer, topic: str) -> int:
    partitions = await producer.partitions_for(topic)
    if not partitions:
        raise RuntimeError(f"Kafka topic has no available partitions: {topic}")
    return len(partitions)


async def publish_pipeline_event(
    producer: AIOKafkaProducer,
    allocator: PipelinePartitionAllocator,
    topic: str,
    event: PipelineStageEvent,
) -> int:
    partition_count = await _partition_count(producer, topic)
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
