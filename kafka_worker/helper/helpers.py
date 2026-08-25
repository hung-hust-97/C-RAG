import json
import copy
import asyncio
import os
from aiokafka import AIOKafkaProducer
from lightrag.lightrag import LightRAG
from lightrag.utils import logger
from kafka_worker.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_GROUP_ID
from kafka_worker.models import KmsEvent

_config_cache = None
_rag_instances: dict[str, LightRAG] = {}
_rag_init_locks: dict[str, asyncio.Lock] = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _is_transient_rag_init_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "timeout",
            "timed out",
            "connection",
            "connection refused",
            "connection reset",
            "connection aborted",
            "server unavailable",
            "service unavailable",
            "channel",
            "grpc",
            "milvus",
            "neo4j",
            "failed to initialize",
            "temporarily unavailable",
        )
    )


async def _initialize_rag_with_retry(rag: LightRAG, profile_name: str | None) -> None:
    attempts = max(1, _env_int("LIGHTRAG_RAG_INIT_RETRY_ATTEMPTS", 4))
    delay = max(1, _env_int("LIGHTRAG_RAG_INIT_RETRY_DELAY_SECONDS", 3))
    max_delay = max(delay, _env_int("LIGHTRAG_RAG_INIT_RETRY_MAX_DELAY_SECONDS", 30))

    for attempt in range(1, attempts + 1):
        try:
            await rag.initialize_storages(profile=profile_name)
            return
        except Exception as exc:
            if attempt >= attempts or not _is_transient_rag_init_error(exc):
                raise
            sleep_seconds = min(max_delay, delay * (2 ** (attempt - 1)))
            logger.warning(
                "[Kafka] Transient LightRAG init error profile=%s attempt=%s/%s sleep_seconds=%s error=%s",
                profile_name,
                attempt,
                attempts,
                sleep_seconds,
                exc,
            )
            try:
                await rag.finalize_storages()
            except Exception as close_exc:
                logger.debug(
                    "[Kafka] Ignored error while cleaning failed LightRAG init: %s",
                    close_exc,
                )
            await asyncio.sleep(sleep_seconds)


def validate_config() -> None:
    """Kiểm tra xem các biến môi trường bắt buộc cho Kafka đã được thiết lập chưa."""
    required = {
        "KAFKA_BOOTSTRAP_SERVERS": KAFKA_BOOTSTRAP_SERVERS,
        "KAFKA_GROUP_ID": KAFKA_GROUP_ID,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Please set these environment variables before running the worker."
        )


def _get_rag_instance(workspace_id: str) -> LightRAG:
    """Khởi tạo một instance LightRAG hoàn chỉnh cho một workspace."""
    global _config_cache
    from lightrag.api.lightrag_factory import LLMConfigCache, build_rag_instance
    from lightrag.api.config import global_args

    if _config_cache is None:
        _config_cache = LLMConfigCache(global_args)

    copied_args = copy.deepcopy(global_args)
    copied_args.working_dir = f"./rag_storage/{workspace_id}"

    return build_rag_instance(workspace_id, copied_args, _config_cache)


def _normalize_profile(profile: object | None) -> str | None:
    value = getattr(profile, "value", profile)
    return str(value) if value else None


async def get_initialized_rag_instance(
    workspace_id: str, profile: object | None = None
) -> LightRAG:
    """Return a cached, initialized LightRAG instance for the worker process."""
    profile_name = _normalize_profile(profile)
    cache_key = f"{workspace_id}_{profile_name}" if profile_name else workspace_id
    if cache_key not in _rag_init_locks:
        _rag_init_locks[cache_key] = asyncio.Lock()

    async with _rag_init_locks[cache_key]:
        rag = _rag_instances.get(cache_key)
        if rag is None:
            rag = _get_rag_instance(workspace_id)
            await _initialize_rag_with_retry(rag, profile_name)
            _rag_instances[cache_key] = rag
            logger.info(
                f"[Kafka] Initialized cached LightRAG for workspace {workspace_id} (profile: {profile_name})"
            )
        return rag


async def close_cached_rag_instances() -> None:
    """Finalize cached LightRAG instances during worker shutdown."""
    for workspace_id, rag in list(_rag_instances.items()):
        try:
            await rag.finalize_storages()
            logger.info(
                f"[Kafka] Finalized cached LightRAG for workspace {workspace_id}"
            )
        except Exception as e:
            logger.warning(
                f"[Kafka] Failed to finalize LightRAG for workspace {workspace_id}: {e}"
            )
    _rag_instances.clear()


async def _publish_event(
    producer: AIOKafkaProducer,
    topic: str,
    event: KmsEvent,
) -> None:
    """Gửi Event (Pydantic object) dạng JSON lên topic tương ứng của Kafka."""
    payload = json.dumps(event.model_dump(by_alias=True)).encode("utf-8")
    await producer.send_and_wait(topic, payload)
    logger.info(
        f"[Kafka] Pushed status {event.status.value} to topic {topic} for {event.document_id}"
    )
