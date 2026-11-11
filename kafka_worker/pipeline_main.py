from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import time
import traceback
from aiohttp import web

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, TopicPartition
from aiokafka.errors import CommitFailedError, IllegalStateError

from kafka_worker.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    PIPELINE_CONSUMER_GROUP_PREFIX,
    PIPELINE_DEBUG_SYNC,
    PIPELINE_FAILED_TOPIC,
    PIPELINE_WORKER_ID,
    PIPELINE_GENERAL_EXTRACTION_TOPIC,
    PIPELINE_LEGAL_EXTRACTION_TOPIC,
    PIPELINE_GENERAL_CHUNKING_TOPIC,
    PIPELINE_LEGAL_CHUNKING_TOPIC,
    PIPELINE_GENERAL_CHUNK_EMBEDDING_TOPIC,
    PIPELINE_LEGAL_CHUNK_EMBEDDING_TOPIC,
    PIPELINE_GENERAL_ENTITY_EXTRACTION_TOPIC,
    PIPELINE_LEGAL_ENTITY_EXTRACTION_TOPIC,
    PIPELINE_GENERAL_KG_MERGE_TOPIC,
    PIPELINE_LEGAL_KG_MERGE_TOPIC,
    PIPELINE_GENERAL_KG_EMBEDDING_TOPIC,
    PIPELINE_LEGAL_KG_EMBEDDING_TOPIC,
    PIPELINE_GENERAL_FINALIZE_TOPIC,
    PIPELINE_LEGAL_FINALIZE_TOPIC,
    KMS_EXTRACTED_TOPIC,
    KMS_FAILED_TOPIC,
    KMS_EXTRACTED_LEGAL_TOPIC,
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


def _stage_env_name(prefix: str, stage: StageName) -> str:
    return f"{prefix}_{stage.value.upper()}"


def _stage_str_env(prefix: str, stage: StageName, default: str = "") -> str:
    stage_value = os.getenv(_stage_env_name(prefix, stage))
    if stage_value is not None and stage_value.strip():
        return stage_value.strip()
    value = os.getenv(prefix)
    if value is not None and value.strip():
        return value.strip()
    return default


def _consumer_group(stage: StageName) -> str:
    stage_group = _stage_str_env("PIPELINE_CONSUMER_GROUP", stage)
    if stage_group:
        return stage_group
    return f"{PIPELINE_CONSUMER_GROUP_PREFIX}_{stage.value}"


PARALLEL_MESSAGE_STAGES = {
    StageName.EXTRACTION,
    StageName.CHUNKING,
    StageName.CHUNK_EMBEDDING,
    StageName.ENTITY_EXTRACTION,
    StageName.KG_EMBEDDING,
    StageName.FINALIZE,
}

DEFAULT_MAX_RECORDS_BY_STAGE = {
    StageName.EXTRACTION: 3,
    StageName.CHUNKING: 4,
    StageName.CHUNK_EMBEDDING: 4,
    StageName.ENTITY_EXTRACTION: 4,
    StageName.KG_MERGE: 2,
    StageName.KG_EMBEDDING: 4,
    StageName.FINALIZE: 8,
}

DEFAULT_CONSUMER_TIMEOUTS_BY_STAGE = {
    stage: {
        "max_poll_interval_ms": 7200000,
        "session_timeout_ms": 120000,
        "heartbeat_interval_ms": 30000,
        "rebalance_timeout_ms": 120000,
        "request_timeout_ms": 120000,
        "retry_backoff_ms": 1000,
    }
    for stage in StageName
}
DEFAULT_CONSUMER_TIMEOUTS_BY_STAGE[StageName.CHUNKING] = {
    **DEFAULT_CONSUMER_TIMEOUTS_BY_STAGE[StageName.CHUNKING],
    "session_timeout_ms": 300000,
    "rebalance_timeout_ms": 300000,
    "request_timeout_ms": 300000,
}
DEFAULT_CONSUMER_TIMEOUTS_BY_STAGE[StageName.CHUNK_EMBEDDING] = {
    **DEFAULT_CONSUMER_TIMEOUTS_BY_STAGE[StageName.CHUNK_EMBEDDING],
    "session_timeout_ms": 900000,
    "rebalance_timeout_ms": 900000,
    "request_timeout_ms": 900000,
}


def _stage_int_env(
    prefix: str,
    stage: StageName,
    default: int,
    *,
    minimum: int = 1,
) -> int:
    stage_env_name = _stage_env_name(prefix, stage)
    raw_value = os.getenv(stage_env_name)
    env_name = stage_env_name
    if raw_value is None:
        raw_value = os.getenv(prefix)
        env_name = prefix
    if raw_value is None:
        return default

    try:
        return max(minimum, int(raw_value))
    except ValueError:
        logger.warning(
            "[Pipeline] Invalid integer config %s=%r, using default=%s",
            env_name,
            raw_value,
            default,
        )
        return default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return max(minimum, int(raw_value))
    except ValueError:
        logger.warning(
            "[Pipeline] Invalid integer config %s=%r, using default=%s",
            name,
            raw_value,
            default,
        )
        return default


def _consumer_timeout_config(stage: StageName) -> dict[str, int]:
    defaults = DEFAULT_CONSUMER_TIMEOUTS_BY_STAGE[stage]
    config = {
        key: _stage_int_env(
            f"PIPELINE_{key.upper()}",
            stage,
            default,
        )
        for key, default in defaults.items()
    }

    if config["heartbeat_interval_ms"] >= config["session_timeout_ms"]:
        adjusted_heartbeat_ms = max(1000, config["session_timeout_ms"] // 3)
        if adjusted_heartbeat_ms >= config["session_timeout_ms"]:
            adjusted_heartbeat_ms = max(1, config["session_timeout_ms"] - 1)
        logger.warning(
            "[Pipeline] heartbeat_interval_ms=%s must be lower than "
            "session_timeout_ms=%s for stage=%s; using heartbeat_interval_ms=%s",
            config["heartbeat_interval_ms"],
            config["session_timeout_ms"],
            stage.value,
            adjusted_heartbeat_ms,
        )
        config["heartbeat_interval_ms"] = adjusted_heartbeat_ms

    if config["rebalance_timeout_ms"] < config["session_timeout_ms"]:
        logger.warning(
            "[Pipeline] rebalance_timeout_ms=%s is lower than session_timeout_ms=%s "
            "for stage=%s; using session_timeout_ms",
            config["rebalance_timeout_ms"],
            config["session_timeout_ms"],
            stage.value,
        )
        config["rebalance_timeout_ms"] = config["session_timeout_ms"]

    if config["request_timeout_ms"] < config["session_timeout_ms"]:
        logger.warning(
            "[Pipeline] request_timeout_ms=%s is lower than session_timeout_ms=%s "
            "for stage=%s; using session_timeout_ms",
            config["request_timeout_ms"],
            config["session_timeout_ms"],
            stage.value,
        )
        config["request_timeout_ms"] = config["session_timeout_ms"]

    return config


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
    auto_offset_reset = _stage_str_env(
        "PIPELINE_AUTO_OFFSET_RESET",
        stage,
        default="earliest",
    )
    doc_type = os.environ.get("PIPELINE_DOC_TYPE", "all").lower()
    if doc_type == "all":
        topics = list(STAGE_TOPICS[stage].values())
    elif doc_type in STAGE_TOPICS[stage]:
        topics = [STAGE_TOPICS[stage][doc_type]]
    else:
        raise ValueError(f"Invalid PIPELINE_DOC_TYPE: {doc_type}")
    timeout_config = _consumer_timeout_config(stage)
    max_poll_records = _consumer_max_records(stage)
    consumer_group = _consumer_group(stage)

    return AIOKafkaConsumer(
        *topics,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=consumer_group,
        client_id=f"{PIPELINE_WORKER_ID}-{stage.value}",
        enable_auto_commit=False,
        auto_offset_reset=auto_offset_reset,
        max_poll_interval_ms=timeout_config["max_poll_interval_ms"],
        rebalance_timeout_ms=timeout_config["rebalance_timeout_ms"],
        session_timeout_ms=timeout_config["session_timeout_ms"],
        heartbeat_interval_ms=timeout_config["heartbeat_interval_ms"],
        max_poll_records=max_poll_records,
        request_timeout_ms=timeout_config["request_timeout_ms"],
        retry_backoff_ms=timeout_config["retry_backoff_ms"],
    )


def _kafka_start_timeout_seconds(stage: StageName) -> int:
    return _stage_int_env(
        "PIPELINE_KAFKA_START_TIMEOUT_SECONDS",
        stage,
        default=120,
        minimum=1,
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
        doc_type = os.environ.get("PIPELINE_DOC_TYPE", "all").lower()
        if doc_type == "all":
            topics = list(STAGE_TOPICS[selected_stage].values())
        elif doc_type in STAGE_TOPICS[selected_stage]:
            topics = [STAGE_TOPICS[selected_stage][doc_type]]
        consumer_group = _consumer_group(selected_stage)

    context: dict[str, str | int | list[str]] = {
        "stage": selected_stage.value if selected_stage is not None else raw_stage,
        "worker_id": PIPELINE_WORKER_ID,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "topics": topics,
        "consumer_group": consumer_group,
        "auto_offset_reset": _stage_str_env(
            "PIPELINE_AUTO_OFFSET_RESET",
            selected_stage,
            default="earliest",
        )
        if selected_stage is not None
        else os.getenv("PIPELINE_AUTO_OFFSET_RESET", "earliest"),
    }
    if selected_stage is not None:
        try:
            timeout_config = _consumer_timeout_config(selected_stage)
        except Exception as exc:
            context["consumer_config_error"] = str(exc)
        else:
            context.update(timeout_config)
            context["max_poll_records"] = _consumer_max_records(selected_stage)

    return context


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


async def _relay_pipeline_outbox(
    coordinator,
    producer: AIOKafkaProducer,
    partition_allocator,
    shutdown_event: asyncio.Event,
) -> None:
    interval_seconds = max(
        1, _env_int("PIPELINE_OUTBOX_POLL_INTERVAL_SECONDS", 10)
    )
    while not shutdown_event.is_set():
        try:
            await coordinator.publish_outbox(
                lambda event: _publish_event(producer, partition_allocator, event)
            )
        except Exception:
            logger.exception("[Pipeline] Outbox relay iteration failed")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


async def _publish_kms_response(
    producer: AIOKafkaProducer,
    pipeline_event: PipelineStageEvent,
    topic: str,
    status: DocStatus,
    message: str = "",
) -> None:
    try:
        kms_event = KmsEvent.model_validate(pipeline_event.metadata)
        kms_event.status = status
        kms_event.message = message
        if not kms_event.content_path and "content_path" in pipeline_event.metadata:
            kms_event.content_path = pipeline_event.metadata["content_path"]
        if not kms_event.content_hash and "content_hash" in pipeline_event.metadata:
            kms_event.content_hash = pipeline_event.metadata["content_hash"]
        if not kms_event.content_length and "content_length" in pipeline_event.metadata:
            kms_event.content_length = pipeline_event.metadata["content_length"]
        if not kms_event.content_summary and "content_summary" in pipeline_event.metadata:
            kms_event.content_summary = pipeline_event.metadata["content_summary"]

        doc_id_to_log = (
            kms_event.document_id
            or kms_event.legal_document_id
            or pipeline_event.doc_id
        )
        payload = kms_event.model_dump_json(by_alias=True, exclude_none=True).encode(
            "utf-8"
        )
        key = str(doc_id_to_log).encode("utf-8") if doc_id_to_log else None
        await producer.send_and_wait(topic, payload, key=key)
        if status == DocStatus.FAILED:
            logger.error(
                f"[Egress] Published {status.value} to {topic} for doc {doc_id_to_log} | Reason: {message}"
            )

            # Ghi riêng ra file log
            log_dir = os.environ.get("LIGHTRAG_LOG_DIR", "/app/data/logs")
            os.makedirs(log_dir, exist_ok=True)
            failed_log_path = os.path.join(log_dir, "failed_documents.log")
            try:
                from datetime import datetime, timezone

                timestamp = datetime.now(timezone.utc).isoformat()
                stage_val = (
                    pipeline_event.stage_name.value
                    if pipeline_event.stage_name
                    else "unknown"
                )
                with open(failed_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"[{timestamp}] DOC_ID: {doc_id_to_log} | TOPIC: {topic} | STAGE: {stage_val} | ERROR: {message}\n"
                    )
            except Exception as log_err:
                logger.error(
                    f"[Egress] Failed to write to {failed_log_path}: {log_err}"
                )
        else:
            logger.info(
                f"[Egress] Published {status.value} to {topic} for doc {doc_id_to_log}"
            )
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
            is_duplicate = any(
                str(getattr(ne, "status", "")).upper() == "DUPLICATED"
                for ne in (outcome.next_events or [])
            ) or "duplicate" in str(getattr(outcome, "message", "")).lower()
            if not is_duplicate:
                topic = (
                    KMS_EXTRACTED_LEGAL_TOPIC
                    if doc_type == "legal"
                    else KMS_EXTRACTED_TOPIC
                )
                await _publish_kms_response(producer, event, topic, DocStatus.EXTRACTED)

    return outcome.status in {
        "done",
        "already_completed",
        "already_running",
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
    except (CommitFailedError, IllegalStateError) as exc:
        logger.warning(
            "[Pipeline] Commit skipped after Kafka rebalance "
            "topic=%s partition=%s offset=%s error=%s",
            tp.topic,
            tp.partition,
            offset,
            exc,
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
        *(_handle_message_safely(msg, stage, runner, producer) for msg in messages)
    )
    if all(results):
        await _commit_partition_offset(consumer, tp, messages[-1].offset + 1)
        return

    failed_offsets = [
        msg.offset for msg, should_commit in zip(messages, results) if not should_commit
    ]
    if failed_offsets and _seek_partition_offset(consumer, tp, min(failed_offsets)):
        await asyncio.sleep(2)


class GeneralPriorityGate:
    """Decides when legal partitions must be paused so general documents go first.

    Legal consumption is paused while general records keep arriving and for
    ``hold_seconds`` after the last one, then resumed so legal still progresses
    whenever general traffic is idle.
    """

    def __init__(self, hold_seconds: float, clock=time.monotonic) -> None:
        self.hold_seconds = max(0.0, hold_seconds)
        self._clock = clock
        self._last_general_at: float | None = None

    def observe(self, has_general: bool) -> None:
        if has_general:
            self._last_general_at = self._clock()

    def should_pause_legal(self) -> bool:
        if self._last_general_at is None:
            return False
        return self._clock() - self._last_general_at < self.hold_seconds


def _create_priority_gate(stage: StageName) -> GeneralPriorityGate | None:
    """Priority applies only to a worker that consumes both doc types."""
    enabled = os.getenv("PIPELINE_GENERAL_PRIORITY_ENABLED", "true").lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None
    if os.environ.get("PIPELINE_DOC_TYPE", "all").lower() != "all":
        return None
    hold_ms = _stage_int_env(
        "PIPELINE_GENERAL_PRIORITY_HOLD_MS", stage, 3000, minimum=0
    )
    return GeneralPriorityGate(hold_seconds=hold_ms / 1000)


def _general_first(result, legal_topic: str):
    """Order polled partitions so general partitions are handled before legal."""
    return sorted(result.items(), key=lambda item: item[0].topic == legal_topic)


def _apply_legal_pause(
    consumer: AIOKafkaConsumer, legal_topic: str, paused: set, pause: bool
) -> set:
    """Pause or resume the legal partitions; returns the currently paused set."""
    legal_tps = {tp for tp in consumer.assignment() if tp.topic == legal_topic}
    if pause:
        newly_paused = legal_tps - paused
        if newly_paused:
            consumer.pause(*newly_paused)
            logger.info(
                "[Pipeline] Pausing legal partitions for general priority: %s",
                sorted(str(tp) for tp in newly_paused),
            )
        return paused | newly_paused
    to_resume = paused & legal_tps
    if to_resume:
        consumer.resume(*to_resume)
        logger.info("[Pipeline] Resuming legal partitions after general idle")
    return set()


async def _consume_loop(
    consumer: AIOKafkaConsumer,
    stage: StageName,
    runner: StageRunner,
    producer: AIOKafkaProducer,
    shutdown_event: asyncio.Event,
) -> None:
    max_records = _consumer_max_records(stage)
    parallel_messages = stage in PARALLEL_MESSAGE_STAGES and max_records > 1
    timeout_config = _consumer_timeout_config(stage)
    logger.info(
        "[Pipeline] Consumer configured stage=%s group_id=%s max_records=%s "
        "parallel_messages=%s max_poll_interval_ms=%s session_timeout_ms=%s "
        "heartbeat_interval_ms=%s rebalance_timeout_ms=%s request_timeout_ms=%s",
        stage.value,
        _consumer_group(stage),
        max_records,
        parallel_messages,
        timeout_config["max_poll_interval_ms"],
        timeout_config["session_timeout_ms"],
        timeout_config["heartbeat_interval_ms"],
        timeout_config["rebalance_timeout_ms"],
        timeout_config["request_timeout_ms"],
    )
    gate = _create_priority_gate(stage)
    legal_topic = STAGE_TOPICS[stage]["legal"]
    legal_paused: set = set()
    while not shutdown_event.is_set():
        try:
            if gate is not None:
                legal_paused = _apply_legal_pause(
                    consumer, legal_topic, legal_paused, gate.should_pause_legal()
                )
            result = await consumer.getmany(timeout_ms=1000, max_records=max_records)
            if gate is not None:
                gate.observe(
                    has_general=any(
                        tp.topic != legal_topic and messages
                        for tp, messages in result.items()
                    )
                )
            if not result:
                continue

            for tp, messages in _general_first(result, legal_topic):
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
    app.router.add_get("/health", health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
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
        partition_allocator=partition_allocator,
        producer=producer,
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

    kafka_start_timeout_seconds = _kafka_start_timeout_seconds(stage)
    logger.info(
        "[Pipeline] Starting Kafka clients %s kafka_start_timeout_seconds=%s",
        _format_worker_context(_worker_context(stage)),
        kafka_start_timeout_seconds,
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(consumer.start(), producer.start()),
            timeout=kafka_start_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(
            "[Pipeline] Kafka clients failed to start within %ss %s",
            kafka_start_timeout_seconds,
            _format_worker_context(_worker_context(stage)),
        )
        raise
    started_at = utc_now_iso()
    outbox_task = asyncio.create_task(
        _relay_pipeline_outbox(
            coordinator, producer, partition_allocator, shutdown_event
        ),
        name=f"pipeline-outbox-relay-{stage.value}",
    )
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
        shutdown_event.set()
        outbox_task.cancel()
        await asyncio.gather(outbox_task, return_exceptions=True)
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
