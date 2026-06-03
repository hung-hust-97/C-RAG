import logging
import time
import asyncio
import hashlib
import os
import re
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class RedisDistributedSemaphore:
    _ACQUIRE_SCRIPT = """
local key = KEYS[1]
local token = ARGV[1]
local now_ms = tonumber(ARGV[2])
local lease_ms = tonumber(ARGV[3])
local limit = tonumber(ARGV[4])

redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms)
if redis.call('ZCARD', key) < limit then
    redis.call('ZADD', key, now_ms + lease_ms, token)
    redis.call('PEXPIRE', key, lease_ms)
    return 1
end
return 0
"""
    _REFRESH_SCRIPT = """
local key = KEYS[1]
local token = ARGV[1]
local now_ms = tonumber(ARGV[2])
local lease_ms = tonumber(ARGV[3])
local current_score = redis.call('ZSCORE', key, token)

if current_score and tonumber(current_score) > now_ms then
    redis.call('ZADD', key, now_ms + lease_ms, token)
    redis.call('PEXPIRE', key, lease_ms)
    return 1
end
return 0
"""

    def __init__(
        self,
        *,
        redis_url: str,
        key: str,
        limit: int,
        lease_seconds: float,
        acquire_timeout_seconds: float,
        required: bool,
    ) -> None:
        self.redis_url = redis_url
        self.key = key
        self.limit = max(1, int(limit))
        self.lease_ms = int(max(1.0, lease_seconds) * 1000)
        self.acquire_timeout_seconds = max(0.1, acquire_timeout_seconds)
        self.refresh_interval_seconds = max(
            1.0, min(30.0, (self.lease_ms / 1000.0) / 3.0)
        )
        self.required = required
        self._client = None

    async def _redis(self):
        if self._client is None:
            from redis.asyncio import Redis

            self._client = Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    async def acquire(self) -> str | None:
        token = str(uuid.uuid4())
        deadline = time.monotonic() + self.acquire_timeout_seconds
        while True:
            try:
                client = await self._redis()
                acquired = await client.eval(
                    self._ACQUIRE_SCRIPT,
                    1,
                    self.key,
                    token,
                    int(time.time() * 1000),
                    self.lease_ms,
                    self.limit,
                )
                if int(acquired) == 1:
                    return token
            except Exception as exc:
                if self.required:
                    raise RuntimeError(
                        f"Failed to acquire distributed LLM slot {self.key}: {exc}"
                    ) from exc
                logger.warning(
                    "[LoadBalancer] Distributed limiter unavailable for %s: %s. "
                    "Continuing with local limiter only.",
                    self.key,
                    exc,
                )
                return None

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for distributed LLM slot {self.key} "
                    f"(limit={self.limit})"
                )
            await asyncio.sleep(0.05)

    async def release(self, token: str | None) -> None:
        if token is None:
            return
        try:
            client = await self._redis()
            await client.zrem(self.key, token)
        except Exception as exc:
            logger.warning(
                "[LoadBalancer] Failed to release distributed LLM slot %s: %s",
                self.key,
                exc,
            )

    async def _refresh_loop(self, token: str) -> None:
        while True:
            await asyncio.sleep(self.refresh_interval_seconds)
            try:
                client = await self._redis()
                refreshed = await client.eval(
                    self._REFRESH_SCRIPT,
                    1,
                    self.key,
                    token,
                    int(time.time() * 1000),
                    self.lease_ms,
                )
                if int(refreshed) != 1:
                    logger.warning(
                        "[LoadBalancer] Distributed LLM slot lease expired before "
                        "refresh key=%s",
                        self.key,
                    )
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "[LoadBalancer] Failed to refresh distributed LLM slot %s: %s",
                    self.key,
                    exc,
                )

    @asynccontextmanager
    async def slot(self):
        token = await self.acquire()
        refresh_task = (
            asyncio.create_task(self._refresh_loop(token)) if token is not None else None
        )
        try:
            yield
        finally:
            if refresh_task is not None:
                refresh_task.cancel()
                with suppress(asyncio.CancelledError):
                    await refresh_task
            await self.release(token)


@asynccontextmanager
async def _noop_async_context():
    yield


class RedisDistributedNodeHealth:
    def __init__(
        self,
        *,
        redis_url: str,
        key: str,
        unhealthy_ttl_seconds: float,
        recovery_probe_seconds: float,
    ) -> None:
        self.redis_url = redis_url
        self.key = key
        self.failed_key = f"{key}:last_failed"
        self.probe_key = f"{key}:probe"
        self.unhealthy_ttl_seconds = max(1.0, unhealthy_ttl_seconds)
        self.recovery_probe_seconds = max(1.0, recovery_probe_seconds)
        self._client = None

    async def _redis(self):
        if self._client is None:
            from redis.asyncio import Redis

            self._client = Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    async def is_unhealthy(self) -> bool:
        try:
            client = await self._redis()
            return await client.exists(self.key) == 1
        except Exception as exc:
            logger.warning(
                "[LoadBalancer] Distributed health check unavailable for %s: %s",
                self.key,
                exc,
            )
            return False

    async def mark_unhealthy(self, reason: str) -> None:
        try:
            client = await self._redis()
            await client.set(self.key, reason[:500], ex=int(self.unhealthy_ttl_seconds))
            await client.set(
                self.failed_key,
                reason[:500],
                ex=int(max(self.unhealthy_ttl_seconds * 10, 3600)),
            )
        except Exception as exc:
            logger.warning(
                "[LoadBalancer] Failed to mark distributed unhealthy %s: %s",
                self.key,
                exc,
            )

    async def had_recent_failure(self) -> bool:
        try:
            client = await self._redis()
            return await client.exists(self.failed_key) == 1
        except Exception as exc:
            logger.warning(
                "[LoadBalancer] Distributed failure check unavailable for %s: %s",
                self.key,
                exc,
            )
            return False

    async def acquire_recovery_probe(self) -> bool:
        try:
            client = await self._redis()
            return bool(
                await client.set(
                    self.probe_key,
                    str(uuid.uuid4()),
                    ex=int(self.recovery_probe_seconds),
                    nx=True,
                )
            )
        except Exception as exc:
            logger.warning(
                "[LoadBalancer] Distributed recovery probe unavailable for %s: %s",
                self.key,
                exc,
            )
            return False

    async def clear_failure(self) -> None:
        try:
            client = await self._redis()
            await client.delete(self.key, self.failed_key, self.probe_key)
        except Exception as exc:
            logger.warning(
                "[LoadBalancer] Failed to clear distributed health state %s: %s",
                self.key,
                exc,
            )


@dataclass
class LLMNode:
    host: str
    api_key: Optional[str]
    model: str
    semaphore: asyncio.Semaphore
    distributed_semaphore: RedisDistributedSemaphore | None = None
    scoped_distributed_semaphores: dict[str, RedisDistributedSemaphore] | None = None
    distributed_health: RedisDistributedNodeHealth | None = None
    is_healthy: bool = True
    last_failed: float = 0.0


class LLMLoadBalancer:
    """
    An advanced Round-Robin Load Balancer for LLM endpoints.
    Distributes LLM requests across multiple hosts, handles health checking,
    fallback retries on alternative nodes, and prevents overloading remaining nodes
    by enforcing per-node concurrency limits.
    """

    def __init__(self, hosts: str, api_keys: Optional[str], models: str, max_async: int = 100):
        host_list = [h.strip() for h in hosts.split(",") if h.strip()] if hosts else []
        key_list = [k.strip() for k in api_keys.split(",") if k.strip()] if api_keys else []
        model_list = [m.strip() for m in models.split(",") if m.strip()] if models else []

        if not host_list:
            logger.warning("No hosts provided to LLMLoadBalancer")
            self.nodes = []
            self._last_index = 0
            self.cooldown_seconds = 60.0
            self.allow_all_unhealthy_retry = False
            return

        self.nodes: List[LLMNode] = []
        num_nodes = len(host_list)
        self.cooldown_seconds = float(
            os.getenv("LLM_LOAD_BALANCER_NODE_COOLDOWN_SECONDS", "60")
        )
        self.recovery_probe_seconds = float(
            os.getenv("LLM_LOAD_BALANCER_RECOVERY_PROBE_SECONDS", "30")
        )
        self.allow_all_unhealthy_retry = os.getenv(
            "LLM_LOAD_BALANCER_ALLOW_ALL_UNHEALTHY_RETRY", "false"
        ).lower() in {"1", "true", "yes", "on"}

        configured_per_node = int(
            os.getenv("LLM_LOAD_BALANCER_MAX_CONCURRENT_PER_NODE", "60")
        )
        concurrency_per_node = min(
            configured_per_node,
            max(1, max_async // num_nodes) if num_nodes > 1 else max_async,
        )
        distributed_limit_enabled = os.getenv(
            "LLM_LOAD_BALANCER_DISTRIBUTED_LIMIT", "false"
        ).lower() in {"1", "true", "yes", "on"}
        distributed_required = os.getenv(
            "LLM_LOAD_BALANCER_DISTRIBUTED_LIMIT_REQUIRED", "true"
        ).lower() in {"1", "true", "yes", "on"}
        distributed_health_enabled = os.getenv(
            "LLM_LOAD_BALANCER_DISTRIBUTED_HEALTH",
            "true" if distributed_limit_enabled else "false",
        ).lower() in {"1", "true", "yes", "on"}
        redis_url = os.getenv(
            "LLM_LOAD_BALANCER_REDIS_URL",
            os.getenv("REDIS_URI", "redis://localhost:6379"),
        )
        lease_seconds = float(os.getenv("LLM_LOAD_BALANCER_LEASE_SECONDS", "900"))
        acquire_timeout_seconds = float(
            os.getenv("LLM_LOAD_BALANCER_ACQUIRE_TIMEOUT_SECONDS", "600")
        )
        key_prefix = os.getenv("LLM_LOAD_BALANCER_REDIS_KEY_PREFIX", "lightrag:llm")
        scope_limits = self._parse_scope_limits(
            os.getenv(
                "LLM_LOAD_BALANCER_SCOPE_LIMITS",
                "extract:36,summary:8,query:5,default:45",
            ),
            default_limit=concurrency_per_node,
        )
        logger.info(
            f"Initializing LLMLoadBalancer with {num_nodes} nodes. "
            f"Total max_async: {max_async}, Concurrency per node limit: {concurrency_per_node}, "
            f"distributed_limit={distributed_limit_enabled}, scope_limits={scope_limits}"
        )

        for i in range(num_nodes):
            host = host_list[i]
            # Re-use the last available key/model if the lists are shorter than hosts
            key = key_list[i] if i < len(key_list) else (key_list[-1] if key_list else None)
            model = model_list[i] if i < len(model_list) else (model_list[-1] if model_list else "default")
            distributed_semaphore = None
            scoped_distributed_semaphores = None
            distributed_health = None
            if distributed_limit_enabled:
                host_hash = hashlib.sha1(host.encode("utf-8")).hexdigest()[:16]
                distributed_semaphore = RedisDistributedSemaphore(
                    redis_url=redis_url,
                    key=f"{key_prefix}:node:{host_hash}",
                    limit=concurrency_per_node,
                    lease_seconds=lease_seconds,
                    acquire_timeout_seconds=acquire_timeout_seconds,
                    required=distributed_required,
                )
                if scope_limits:
                    scoped_distributed_semaphores = {
                        scope: RedisDistributedSemaphore(
                            redis_url=redis_url,
                            key=f"{key_prefix}:scope:{scope}:node:{host_hash}",
                            limit=min(limit, concurrency_per_node),
                            lease_seconds=lease_seconds,
                            acquire_timeout_seconds=acquire_timeout_seconds,
                            required=distributed_required,
                        )
                        for scope, limit in scope_limits.items()
                    }
                if distributed_health_enabled:
                    distributed_health = RedisDistributedNodeHealth(
                        redis_url=redis_url,
                        key=f"{key_prefix}:unhealthy:{host_hash}",
                        unhealthy_ttl_seconds=self.cooldown_seconds,
                        recovery_probe_seconds=self.recovery_probe_seconds,
                    )

            self.nodes.append(
                LLMNode(
                    host=host,
                    api_key=key,
                    model=model,
                    semaphore=asyncio.Semaphore(concurrency_per_node),
                    distributed_semaphore=distributed_semaphore,
                    scoped_distributed_semaphores=scoped_distributed_semaphores,
                    distributed_health=distributed_health,
                )
            )

        self._last_index = 0

    @staticmethod
    def _normalize_scope(scope: str | None) -> str:
        normalized = re.sub(r"[^a-z0-9_]+", "_", (scope or "default").lower()).strip("_")
        return normalized or "default"

    @classmethod
    def _parse_scope_limits(
        cls, raw_limits: str | None, *, default_limit: int
    ) -> dict[str, int]:
        if raw_limits is None or not raw_limits.strip():
            return {}

        limits: dict[str, int] = {}
        for item in raw_limits.split(","):
            if not item.strip() or ":" not in item:
                continue
            raw_scope, raw_limit = item.split(":", 1)
            scope = cls._normalize_scope(raw_scope)
            try:
                limit = int(raw_limit.strip())
            except ValueError:
                logger.warning(
                    "[LoadBalancer] Invalid scope limit %r in LLM_LOAD_BALANCER_SCOPE_LIMITS",
                    item,
                )
                continue
            if limit <= 0:
                continue
            limits[scope] = min(limit, default_limit)
        return limits

    @staticmethod
    def _scoped_slot_context(node: LLMNode, scope: str | None):
        semaphores = node.scoped_distributed_semaphores
        if not semaphores:
            return _noop_async_context()
        normalized_scope = LLMLoadBalancer._normalize_scope(scope)
        semaphore = semaphores.get(normalized_scope) or semaphores.get("default")
        if semaphore is None:
            return _noop_async_context()
        return semaphore.slot()

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        exc = LLMLoadBalancer._unwrap_retry_error(exc)
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            try:
                return int(status_code)
            except (TypeError, ValueError):
                pass
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            try:
                return int(status_code)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _unwrap_retry_error(exc: Exception) -> Exception:
        last_attempt = getattr(exc, "last_attempt", None)
        if last_attempt is None:
            return exc
        try:
            final_exception = last_attempt.exception()
        except Exception:
            return exc
        if isinstance(final_exception, Exception):
            return final_exception
        return exc

    @classmethod
    def _retryable_node_error(cls, exc: Exception) -> Exception | None:
        exc = cls._unwrap_retry_error(exc)
        status_code = cls._status_code(exc)
        if status_code is not None:
            return exc if status_code == 429 or status_code >= 500 else None

        error_name = type(exc).__name__.lower()
        error_text = str(exc).lower()
        retryable_tokens = (
            "apiconnection",
            "connection",
            "connect",
            "timeout",
            "network",
            "serverdisconnected",
            "remoteprotocol",
            "readerror",
            "writeerror",
            "pooltimeout",
        )
        if any(token in error_name for token in retryable_tokens):
            return exc
        if any(
            token in error_text
            for token in (
                "connection refused",
                "connection reset",
                "connect call failed",
                "server disconnected",
                "temporarily unavailable",
                "timed out",
                "timeout",
                "network is unreachable",
                "no route to host",
                "eof",
            )
        ):
            return exc
        return None

    async def _mark_unhealthy(self, node: LLMNode, exc: Exception) -> None:
        node.is_healthy = False
        node.last_failed = time.time()
        if node.distributed_health is not None:
            await node.distributed_health.mark_unhealthy(
                f"{type(exc).__name__}: {exc}"
            )
        logger.error(
            "[LoadBalancer] LLM node %s is unhealthy after %s: %s. "
            "Failing over to another node for %.1fs.",
            node.host,
            type(exc).__name__,
            exc,
            self.cooldown_seconds,
        )

    async def _mark_healthy(self, node: LLMNode) -> None:
        node.is_healthy = True
        node.last_failed = 0.0
        if node.distributed_health is not None:
            await node.distributed_health.clear_failure()

    async def _is_available(self, node: LLMNode, now: float) -> bool:
        if not node.is_healthy:
            if now - node.last_failed > self.cooldown_seconds:
                node.is_healthy = True
                logger.info(
                    "[LoadBalancer] Node %s cooldown expired. Marked as HEALTHY again.",
                    node.host,
                )
            else:
                return False
        if node.distributed_health is not None:
            if await node.distributed_health.is_unhealthy():
                return False
            if await node.distributed_health.had_recent_failure():
                if not await node.distributed_health.acquire_recovery_probe():
                    return False
                logger.info(
                    "[LoadBalancer] Allowing one half-open recovery probe for node %s.",
                    node.host,
                )
        return True

    async def _is_still_allowed_before_send(self, node: LLMNode, now: float) -> bool:
        if not node.is_healthy and now - node.last_failed <= self.cooldown_seconds:
            return False
        if node.distributed_health is not None and await node.distributed_health.is_unhealthy():
            return False
        return True

    def next_node(self) -> Optional[LLMNode]:
        """Legacy compatibility function. Returns the next LLM node without health checks."""
        if not self.nodes:
            return None
        node = self.nodes[self._last_index % len(self.nodes)]
        self._last_index = (self._last_index + 1) % len(self.nodes)
        return node

    async def execute_with_fallback(self, func, *args, scope: str | None = None, **kwargs) -> Any:
        """
        Executes an LLM function by selecting a healthy node and acquiring its semaphore.
        If the selected node fails (connection error, timeout, server 5xx error), 
        it automatically marks the node as unhealthy, releases its slots, and retries 
        the request on another healthy node.
        """
        if not self.nodes:
            raise ValueError("No LLM nodes available in load balancer")

        now = time.time()

        # 1. Get available healthy nodes. Local state is per process, while
        # optional Redis health state is shared by all worker/API processes.
        healthy_nodes = [
            node for node in self.nodes if await self._is_available(node, now)
        ]
        
        # Disaster recovery is opt-in. By default, do not route to known bad nodes.
        if not healthy_nodes:
            if self.allow_all_unhealthy_retry:
                logger.warning(
                    "[LoadBalancer] All LLM nodes are unhealthy. "
                    "LLM_LOAD_BALANCER_ALLOW_ALL_UNHEALTHY_RETRY=true, retrying all nodes."
                )
                healthy_nodes = self.nodes
            else:
                raise RuntimeError(
                    "All LLM nodes are unhealthy; refusing to route requests to known bad nodes"
                )

        attempts = len(healthy_nodes)
        last_exception = None

        # Reserve the starting node before any await point so concurrent callers do
        # not all read the same index and stampede one vLLM endpoint.
        start_index = self._last_index % len(healthy_nodes)
        self._last_index = (self._last_index + 1) % len(healthy_nodes)

        # 3. Try available nodes in round-robin order
        for attempt in range(attempts):
            idx = (start_index + attempt) % len(healthy_nodes)
            node = healthy_nodes[idx]

            logger.debug(
                f"[LoadBalancer] Routing request to node {node.host} "
                f"(Attempt {attempt + 1}/{attempts}, Scope: {self._normalize_scope(scope)}, "
                f"Semaphore available: {node.semaphore._value})"
            )

            try:
                async with self._scoped_slot_context(node, scope):
                    # Acquire local and optional distributed semaphores to enforce per-node concurrency.
                    async with node.semaphore:
                        slot_context = (
                            node.distributed_semaphore.slot()
                            if node.distributed_semaphore is not None
                            else _noop_async_context()
                        )
                        async with slot_context:
                            if not await self._is_still_allowed_before_send(node, time.time()):
                                logger.info(
                                    "[LoadBalancer] Node %s became unhealthy before send; skipping.",
                                    node.host,
                                )
                                continue
                            try:
                                # Execute the LLM request
                                result = await func(node)
                                await self._mark_healthy(node)
                                return result
                            except Exception as e:
                                retryable_error = self._retryable_node_error(e)
                                if retryable_error is None:
                                    logger.error(
                                        "[LoadBalancer] LLM request failed on %s with "
                                        "non-retryable %s: %s",
                                        node.host,
                                        type(e).__name__,
                                        e,
                                    )
                                    raise

                                await self._mark_unhealthy(node, retryable_error)
                                last_exception = e
            except TimeoutError as e:
                logger.warning(
                    "[LoadBalancer] Timed out waiting for limiter slot on %s. "
                    "Trying another node if available.",
                    node.host,
                )
                last_exception = e

        # If all attempts failed, raise the last exception
        if last_exception:
            raise last_exception
        raise ValueError("LLM execution failed on all nodes in load balancer")
