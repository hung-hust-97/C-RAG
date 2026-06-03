from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import traceback
from aiohttp import web

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.errors import IllegalStateError

from kafka_worker.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    PIPELINE_CONSUMER_GROUP_PREFIX,
    PIPELINE_DEBUG_SYNC,
    PIPELINE_FAILED_TOPIC,
    PIPELINE_WORKER_ID,
    PIPELINE_GENERAL_EXTRACTION_TOPIC, PIPELINE_LEGAL_EXTRACTION_TOPIC,
    PIPELINE_GENERAL_CHUNKING_TOPIC, PIPELINE_LEGAL_CHUNKING_TOPIC,
    PIPELINE_GENERAL_CHUNK_EMBEDDING_TOPIC, PIPELINE_LEGAL_CHUNK_EMBEDDING_TOPIC,
    PIPELINE_GENERAL_ENTITY_EXTRACTION_TOPIC, PIPELINE_LEGAL_ENTITY_EXTRACTION_TOPIC,
    PIPELINE_GENERAL_KG_MERGE_TOPIC, PIPELINE_LEGAL_KG_MERGE_TOPIC,
    PIPELINE_GENERAL_KG_EMBEDDING_TOPIC, PIPELINE_LEGAL_KG_EMBEDDING_TOPIC,
    PIPELINE_GENERAL_FINALIZE_TOPIC, PIPELINE_LEGAL_FINALIZE_TOPIC,
    KMS_EXTRACTED_TOPIC,
    KMS_PROCESSED_TOPIC,
    KMS_FAILED_TOPIC,
    KMS_EXTRACTED_LEGAL_TOPIC,
    KMS_PROCESSED_LEGAL_TOPIC,
    KMS_FAILED_LEGAL_TOPIC,
)
from kafka_worker.models import KmsEvent
from lightrag.base import DocStatus
from kafka_worker.helper.helpers import (
    close_cached_rag_instances,
    get_initialized_rag_instance,
)
from lightrag.kg.postgres_impl import ClientManager
from lightrag.pipeline.artifacts import create_artifact_store_from_env
from lightrag.pipeline.coordinator import create_coordinator_from_env
from lightrag.pipeline.models import PipelineStageEvent, StageName, utc_now_iso
from lightrag.pipeline.partitioning import create_partition_allocator_from_env
from lightrag.pipeline.runner import StageExecutionContext, StageRunner
from lightrag.pipeline.stages import default_stage_handlers
from lightrag.utils import logger
from kafka_worker.partitioned_publish import publish_pipeline_event


STAGE_TOPICS = {
    StageName.EXTRACTION: {
        "general": PIPELINE_GENERAL_EXTRACTION_TOPIC,
        "legal": PIPELINE_LEGAL_EXTRACTION_TOPIC,
    },
    StageName.CHUNKING: {
        "general": PIPELINE_GENERAL_CHUNKING_TOPIC,
        "legal": PIPELINE_LEGAL_CHUNKING_TOPIC,
    },
    StageName.CHUNK_EMBEDDING: {
        "general": PIPELINE_GENERAL_CHUNK_EMBEDDING_TOPIC,
        "legal": PIPELINE_LEGAL_CHUNK_EMBEDDING_TOPIC,
    },
    StageName.ENTITY_EXTRACTION: {
        "general": PIPELINE_GENERAL_ENTITY_EXTRACTION_TOPIC,
        "legal": PIPELINE_LEGAL_ENTITY_EXTRACTION_TOPIC,
    },
    StageName.KG_MERGE: {
        "general": PIPELINE_GENERAL_KG_MERGE_TOPIC,
        "legal": PIPELINE_LEGAL_KG_MERGE_TOPIC,
    },
    StageName.KG_EMBEDDING: {
        "general": PIPELINE_GENERAL_KG_EMBEDDING_TOPIC,
        "legal": PIPELINE_LEGAL_KG_EMBEDDING_TOPIC,
    },
    StageName.FINALIZE: {
        "general": PIPELINE_GENERAL_FINALIZE_TOPIC,
        "legal": PIPELINE_LEGAL_FINALIZE_TOPIC,
    },
}


def _canonical_doc_type(event: PipelineStageEvent) -> str:
    raw_doc_type = str(getattr(event, "doc_type", "general") or "general").lower()
    metadata_doc_type = str(event.metadata.get("doc_type", "") or "").lower()
    if (
        raw_doc_type == "legal"
        or metadata_doc_type == "legal"
        or event.metadata.get("legal_document_id") is not None
    ):
        return "legal"
    return "general"


def _normalize_event_doc_type(event: PipelineStageEvent) -> PipelineStageEvent:
    doc_type = _canonical_doc_type(event)
    if event.doc_type == doc_type and event.metadata.get("doc_type") == doc_type:
        return event

    normalized = event.model_copy(deep=True)
    normalized.doc_type = doc_type
    normalized.metadata = {
        **normalized.metadata,
        "doc_type": doc_type,
    }
    return normalized


def _consumer_group(stage: StageName) -> str:
    return f"{PIPELINE_CONSUMER_GROUP_PREFIX}_{stage.value}"


PARALLEL_MESSAGE_STAGES = {
    StageName.CHUNKING,
    StageName.CHUNK_EMBEDDING,
    StageName.KG_EMBEDDING,
    StageName.FINALIZE,
}

DEFAULT_MAX_RECORDS_BY_STAGE = {
    StageName.EXTRACTION: 1,
    StageName.CHUNKING: 4,
    StageName.CHUNK_EMBEDDING: 8,
    StageName.ENTITY_EXTRACTION: 1,
    StageName.KG_MERGE: 1,
    StageName.KG_EMBEDDING: 8,
    StageName.FINALIZE: 4,
}


def _stage_env_name(prefix: str, stage: StageName) -> str:
    return f"{prefix}_{stage.value.upper()}"


def _consumer_max_records(stage: StageName) -> int:
    env_name = _stage_env_name("PIPELINE_CONSUMER_MAX_RECORDS", stage)
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return DEFAULT_MAX_RECORDS_BY_STAGE[stage]
    try:
        return max(1, int(raw_value))
    except ValueError:
        logger.warning(
            "[Pipeline] Invalid consumer max records %s=%r, using default=%s",
            env_name,
            raw_value,
            DEFAULT_MAX_RECORDS_BY_STAGE[stage],
        )
        return DEFAULT_MAX_RECORDS_BY_STAGE[stage]


def _parse_stage() -> StageName:
    raw_stage = os.getenv("PIPELINE_WORKER_STAGE", StageName.CHUNKING.value)
    try:
        return StageName(raw_stage)
    except ValueError as exc:
        allowed = ", ".join(stage.value for stage in StageName)
        raise ValueError(
            f"Invalid PIPELINE_WORKER_STAGE={raw_stage}. Use one of: {allowed}"
        ) from exc


def _create_consumer(stage: StageName) -> AIOKafkaConsumer:
    auto_offset_reset = os.getenv("PIPELINE_AUTO_OFFSET_RESET", "earliest")
    topics = list(STAGE_TOPICS[stage].values())
    return AIOKafkaConsumer(
        *topics,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=_consumer_group(stage),
        enable_auto_commit=False,
        auto_offset_reset=auto_offset_reset,
        max_poll_interval_ms=int(os.getenv("PIPELINE_MAX_POLL_INTERVAL_MS", "3600000")),
    )


def _worker_context(stage: StageName | None = None) -> dict[str, str | int | list[str]]:
    raw_stage = os.getenv("PIPELINE_WORKER_STAGE", StageName.CHUNKING.value)
    selected_stage = stage
    if selected_stage is None:
        with contextlib.suppress(ValueError):
            selected_stage = StageName(raw_stage)

    topics = []
    consumer_group = "unknown"
    if selected_stage is not None:
        topics = list(STAGE_TOPICS[selected_stage].values())
        consumer_group = _consumer_group(selected_stage)

    return {
        "stage": selected_stage.value if selected_stage is not None else raw_stage,
        "worker_id": PIPELINE_WORKER_ID,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "topics": topics,
        "consumer_group": consumer_group,
        "auto_offset_reset": os.getenv("PIPELINE_AUTO_OFFSET_RESET", "earliest"),
    }


def _format_worker_context(context: dict[str, str | int | list[str]]) -> str:
    return " ".join(f"{key}={value}" for key, value in context.items())


async def _publish_event(
    producer: AIOKafkaProducer,
    partition_allocator,
    event: PipelineStageEvent,
) -> None:
    event = _normalize_event_doc_type(event)
    doc_type = _canonical_doc_type(event)
    topic = STAGE_TOPICS[event.stage_name][doc_type]
    await publish_pipeline_event(producer, partition_allocator, topic, event)


async def _publish_kms_response(
    producer: AIOKafkaProducer, pipeline_event: PipelineStageEvent, topic: str, status: DocStatus, message: str = ""
) -> None:
    try:
        kms_event = KmsEvent.model_validate(pipeline_event.metadata)
        kms_event.status = status
        kms_event.message = message
        payload = kms_event.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
        await producer.send_and_wait(topic, payload)
        
        doc_id_to_log = kms_event.document_id or kms_event.legal_document_id or pipeline_event.doc_id
        if status == DocStatus.FAILED:
            logger.error(f"[Egress] Published {status.value} to {topic} for doc {doc_id_to_log} | Reason: {message}")
            
            # Ghi riêng ra file log
            log_dir = os.environ.get("LIGHTRAG_LOG_DIR", "/app/data/logs")
            os.makedirs(log_dir, exist_ok=True)
            failed_log_path = os.path.join(log_dir, "failed_documents.log")
            try:
                from datetime import datetime, timezone
                timestamp = datetime.now(timezone.utc).isoformat()
                stage_val = pipeline_event.stage_name.value if pipeline_event.stage_name else "unknown"
                with open(failed_log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] DOC_ID: {doc_id_to_log} | TOPIC: {topic} | STAGE: {stage_val} | ERROR: {message}\n")
            except Exception as log_err:
                logger.error(f"[Egress] Failed to write to {failed_log_path}: {log_err}")
        else:
            logger.info(f"[Egress] Published {status.value} to {topic} for doc {doc_id_to_log}")
    except Exception as e:
        logger.error(f"[Egress] Failed to publish {status.value} to {topic}: {e}")

async def _publish_failure(
    producer: AIOKafkaProducer,
    partition_allocator,
    event: PipelineStageEvent,
) -> None:
    event = _normalize_event_doc_type(event)
    await publish_pipeline_event(
        producer,
        partition_allocator,
        PIPELINE_FAILED_TOPIC,
        event,
    )
    error_msg = event.metadata.get("error", {}).get("message", "Pipeline failed")
    
    doc_type = _canonical_doc_type(event)
    topic = KMS_FAILED_LEGAL_TOPIC if doc_type == "legal" else KMS_FAILED_TOPIC
    await _publish_kms_response(producer, event, topic, DocStatus.FAILED, error_msg)


async def _handle_message(
    msg,
    stage: StageName,
    runner: StageRunner,
    producer: AIOKafkaProducer,
) -> bool:
    try:
        event = PipelineStageEvent.model_validate_json(msg.value.decode("utf-8"))
    except Exception:
        logger.exception("[Pipeline] Failed to parse message from topic=%s", msg.topic)
        return True

    if event.stage_name != stage:
        logger.error(
            "[Pipeline] Stage mismatch. worker_stage=%s event=%s",
            stage.value,
            event.log_context(),
        )
        return True

    outcome = await runner.run(event)
    
    if outcome.status == "done":
        event = _normalize_event_doc_type(event)
        doc_type = _canonical_doc_type(event)
        if event.stage_name == StageName.EXTRACTION:
            topic = KMS_EXTRACTED_LEGAL_TOPIC if doc_type == "legal" else KMS_EXTRACTED_TOPIC
            await _publish_kms_response(producer, event, topic, DocStatus.EXTRACTED)
        elif event.stage_name == StageName.FINALIZE:
            topic = KMS_PROCESSED_LEGAL_TOPIC if doc_type == "legal" else KMS_PROCESSED_TOPIC
            await _publish_kms_response(producer, event, topic, DocStatus.PROCESSED)
            
    return outcome.status in {
        "done",
        "already_completed",
        "failed",
        "failed_attempt_recorded",
    }


async def _handle_message_safely(
    msg,
    stage: StageName,
    runner: StageRunner,
    producer: AIOKafkaProducer,
) -> bool:
    try:
        return await _handle_message(msg, stage, runner, producer)
    except Exception:
        logger.error(
            "[Pipeline] Unhandled worker error topic=%s offset=%s\n%s",
            msg.topic,
            msg.offset,
            traceback.format_exc(),
        )
        return False


async def _commit_partition_offset(
    consumer: AIOKafkaConsumer,
    tp,
    offset: int,
) -> bool:
    commit_tp = TopicPartition(tp.topic, tp.partition)
    if commit_tp not in consumer.assignment():
        logger.warning(
            "[Pipeline] Skipping commit because partition is no longer assigned "
            "topic=%s partition=%s offset=%s",
            tp.topic,
            tp.partition,
            offset,
        )
        return False
    try:
        await consumer.commit({commit_tp: offset})
        return True
    except IllegalStateError:
        logger.warning(
            "[Pipeline] Commit skipped after Kafka rebalance "
            "topic=%s partition=%s offset=%s",
            tp.topic,
            tp.partition,
            offset,
        )
        return False


def _seek_partition_offset(
    consumer: AIOKafkaConsumer,
    tp,
    offset: int,
) -> bool:
    if tp not in consumer.assignment():
        logger.warning(
            "[Pipeline] Skipping seek because partition is no longer assigned "
            "topic=%s partition=%s offset=%s",
            tp.topic,
            tp.partition,
            offset,
        )
        return False
    try:
        consumer.seek(tp, offset)
        return True
    except IllegalStateError:
        logger.warning(
            "[Pipeline] Seek skipped after Kafka rebalance "
            "topic=%s partition=%s offset=%s",
            tp.topic,
            tp.partition,
            offset,
        )
        return False


async def _process_partition_sequentially(
    consumer: AIOKafkaConsumer,
    tp,
    messages,
    stage: StageName,
    runner: StageRunner,
    producer: AIOKafkaProducer,
) -> None:
    for msg in messages:
        should_commit = await _handle_message_safely(msg, stage, runner, producer)
        if should_commit:
            await _commit_partition_offset(consumer, tp, msg.offset + 1)
            continue
        if _seek_partition_offset(consumer, tp, msg.offset):
            await asyncio.sleep(2)
        break


async def _process_partition_in_parallel(
    consumer: AIOKafkaConsumer,
    tp,
    messages,
    stage: StageName,
    runner: StageRunner,
    producer: AIOKafkaProducer,
) -> None:
    results = await asyncio.gather(
        *(
            _handle_message_safely(msg, stage, runner, producer)
            for msg in messages
        )
    )
    if all(results):
        await _commit_partition_offset(consumer, tp, messages[-1].offset + 1)
        return

    failed_offsets = [
        msg.offset for msg, should_commit in zip(messages, results) if not should_commit
    ]
    if failed_offsets and _seek_partition_offset(consumer, tp, min(failed_offsets)):
        await asyncio.sleep(2)


async def _consume_loop(
    consumer: AIOKafkaConsumer,
    stage: StageName,
    runner: StageRunner,
    producer: AIOKafkaProducer,
    shutdown_event: asyncio.Event,
) -> None:
    max_records = _consumer_max_records(stage)
    parallel_messages = stage in PARALLEL_MESSAGE_STAGES and max_records > 1
    logger.info(
        "[Pipeline] Consumer configured stage=%s max_records=%s parallel_messages=%s",
        stage.value,
        max_records,
        parallel_messages,
    )
    while not shutdown_event.is_set():
        try:
            result = await consumer.getmany(timeout_ms=1000, max_records=max_records)
            if not result:
                continue

            for tp, messages in result.items():
                if parallel_messages:
                    await _process_partition_in_parallel(
                        consumer, tp, messages, stage, runner, producer
                    )
                else:
                    await _process_partition_sequentially(
                        consumer, tp, messages, stage, runner, producer
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("[Pipeline] Consumer loop error\n%s", traceback.format_exc())
            await asyncio.sleep(2)


async def health_check(request):
    return web.json_response({"status": "ok", "worker_id": PIPELINE_WORKER_ID})

async def _start_health_server(shutdown_event: asyncio.Event):
    app = web.Application()
    app.router.add_get('/health', health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()
    logger.info("[Pipeline] Health check server started on port 8080")
    
    await shutdown_event.wait()
    await runner.cleanup()


async def _run_worker_tasks(
    consumer: AIOKafkaConsumer,
    producer: AIOKafkaProducer,
    stage: StageName,
    runner: StageRunner,
    shutdown_event: asyncio.Event,
    started_at: str,
) -> None:
    tasks = [
        asyncio.create_task(
            _consume_loop(consumer, stage, runner, producer, shutdown_event),
            name="pipeline-consume-loop",
        ),
        asyncio.create_task(
            _start_health_server(shutdown_event),
            name="pipeline-health-server",
        ),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

        first_exception: BaseException | None = None
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None and first_exception is None:
                first_exception = exc

        shutdown_event.set()
        if first_exception is not None:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise first_exception

        if pending:
            await asyncio.gather(*pending)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def main() -> None:
    stage = _parse_stage()
    consumer = _create_consumer(stage)
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    partition_allocator = create_partition_allocator_from_env()
    artifact_store = create_artifact_store_from_env()
    coordinator = create_coordinator_from_env()
    context = StageExecutionContext(
        artifact_store=artifact_store,
        coordinator=coordinator,
        rag_factory=get_initialized_rag_instance,
        debug_sync=PIPELINE_DEBUG_SYNC,
    )
    runner = StageRunner(
        handlers=default_stage_handlers(),
        context=context,
        event_publisher=lambda event: _publish_event(
            producer, partition_allocator, event
        ),
        failure_publisher=lambda event: _publish_failure(
            producer, partition_allocator, event
        ),
        raise_stage_errors=False,
    )

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info(
            "[Pipeline] Received shutdown signal %s",
            _format_worker_context(_worker_context(stage)),
        )
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda _sig, _frame: _signal_handler())

    await asyncio.gather(consumer.start(), producer.start())
    started_at = utc_now_iso()
    logger.info(
        "[Pipeline] Worker started %s debug_sync=%s",
        _format_worker_context(_worker_context(stage)),
        PIPELINE_DEBUG_SYNC,
    )
    try:
        await _run_worker_tasks(
            consumer,
            producer,
            stage,
            runner,
            shutdown_event,
            started_at,
        )
    finally:
        logger.info(
            "[Pipeline] Shutting down stage worker %s",
            _format_worker_context(_worker_context(stage)),
        )
        await asyncio.gather(
            consumer.stop(),
            producer.stop(),
            return_exceptions=True,
        )
        await asyncio.gather(
            close_cached_rag_instances(),
            coordinator.finalize(),
            partition_allocator.finalize(),
            return_exceptions=True,
        )
        await ClientManager.force_close_shared_pool()


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info(
            "[Pipeline] Worker interrupted %s",
            _format_worker_context(_worker_context()),
        )
        raise
    except BaseException:
        logger.exception(
            "[Pipeline] WORKER_CRASH %s",
            _format_worker_context(_worker_context()),
        )
        raise


if __name__ == "__main__":
    run()
