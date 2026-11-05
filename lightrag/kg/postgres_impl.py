import asyncio
import json
import os
import uuid
import datetime
import traceback
from datetime import timezone
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar, Union, final
import configparser
import ssl
import asyncpg  # type: ignore
from asyncpg import Pool  # type: ignore
from dotenv import load_dotenv


from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from ..base import (
    BaseKVStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
)
from ..namespace import NameSpace, is_namespace
from ..utils import logger
from ..kg.shared_storage import get_data_init_lock

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

T = TypeVar("T")


def to_uuid(val: Any) -> uuid.UUID:
    """Safely convert any ID (even with prefix like 'chunk-', 'doc-', etc.) to standard uuid.UUID."""
    if isinstance(val, uuid.UUID):
        return val
    if not isinstance(val, str):
        return uuid.UUID(str(val))

    orig_val = val
    # Extract the last part if colon is present (e.g. default:extract:hash)
    if ":" in val:
        val = val.split(":")[-1]

    # Strip common core LightRAG string prefixes
    for prefix in ["chunk-", "ent-", "rel-", "doc-", "node-", "edge-"]:
        if val.startswith(prefix):
            val = val[len(prefix) :]
            break
    clean_val = val.replace("-", "")
    if len(clean_val) == 32:
        try:
            return uuid.UUID(clean_val)
        except ValueError:
            pass
    # Fallback to deterministic UUIDv5 using namespace DNS and the original string value
    return uuid.uuid5(uuid.NAMESPACE_DNS, str(orig_val))


class PostgreSQLDB:
    def __init__(self, config: dict[str, Any], **kwargs: Any):
        self.host = config["host"]
        self.port = config["port"]
        self.user = config["user"]
        self.password = config["password"]
        self.database = config["database"]
        self.workspace = config["workspace"]
        self.max = int(config["max_connections"])
        self.increment = 1
        self.pool: Pool | None = None
        self._pool_loop: asyncio.AbstractEventLoop | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._health_check_lock: asyncio.Lock | None = None
        self._health_check_lock_loop: asyncio.AbstractEventLoop | None = None

        # SSL configuration
        self.ssl_mode = config.get("ssl_mode")
        self.ssl_cert = config.get("ssl_cert")
        self.ssl_key = config.get("ssl_key")
        self.ssl_root_cert = config.get("ssl_root_cert")

        # Server settings
        self.server_settings = config.get("server_settings")

        # Statement LRU cache size (keep as-is, allow None for optional configuration)
        self.statement_cache_size = config.get("statement_cache_size")

        # Validate required connection parameters
        if not self.user or not self.password or not self.database:
            missing_params = []
            if not self.user:
                missing_params.append("user")
            if not self.password:
                missing_params.append("password")
            if not self.database:
                missing_params.append("database")
            raise ValueError(
                f"Missing required database parameters: {', '.join(missing_params)}"
            )

        # Validate host and port
        if not self.host:
            raise ValueError("Missing database host")

        try:
            self.port = int(self.port)
            if self.port <= 0 or self.port > 65535:
                raise ValueError(f"Invalid port number: {self.port}")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid port number: {self.port}")

        # Guard concurrent pool resets - initialized lazily to avoid loop affinity issues
        self._pool_reconnect_lock: asyncio.Lock | None = None

        self._transient_exceptions = (
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
            OSError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.TooManyConnectionsError,
            asyncpg.exceptions.CannotConnectNowError,
            asyncpg.exceptions.PostgresConnectionError,
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.ConnectionFailureError,
        )

        # Connection retry configuration
        self.connection_retry_attempts = config["connection_retry_attempts"]
        self.connection_retry_backoff = config["connection_retry_backoff"]
        self.connection_retry_backoff_max = max(
            self.connection_retry_backoff,
            config["connection_retry_backoff_max"],
        )
        self.pool_close_timeout = config["pool_close_timeout"]

        # Connection pool idle timeout (0 = disabled)
        self.max_inactive_connection_lifetime = config.get(
            "max_inactive_connection_lifetime", 300.0
        )

        logger.info(
            "PostgreSQL, Retry config: attempts=%s, backoff=%.1fs, backoff_max=%.1fs, pool_close_timeout=%.1fs",
            self.connection_retry_attempts,
            self.connection_retry_backoff,
            self.connection_retry_backoff_max,
            self.pool_close_timeout,
        )
        self._registered_workspaces: dict[str, str] = {}

    async def ensure_workspace_metadata(
        self, workspace_id: str, workspace_name: str | None = None
    ) -> None:
        """Upsert workspace metadata row for workspace discovery and display."""
        normalized_workspace_id = (workspace_id or "").strip()
        if not normalized_workspace_id:
            return

        # Check if workspace already exists in tbl_workspace
        # If it does, we do NOT want to overwrite its existing nice name with a fallback workspace_id
        check_sql = "SELECT name FROM tbl_workspace WHERE id = $1"
        try:
            row = await self.query(check_sql, [to_uuid(normalized_workspace_id)])
            if row and isinstance(row, dict) and row.get("name"):
                existing_name = row["name"]
                if not workspace_name or workspace_name == existing_name:
                    # Update local cache and return early to preserve the nice name in the database
                    self._registered_workspaces[normalized_workspace_id] = existing_name
                    return
        except Exception as e:
            logger.debug(f"Failed to check existing workspace metadata: {e}")

        normalized_workspace_name = (workspace_name or normalized_workspace_id).strip()
        if not normalized_workspace_name:
            normalized_workspace_name = normalized_workspace_id

        cached_name = self._registered_workspaces.get(normalized_workspace_id)
        if cached_name == normalized_workspace_name:
            return

        # Using tbl_workspace and snake_case columns to match Java schema
        upsert_workspace_sql = """
        INSERT INTO tbl_workspace (id, name, update_time)
        VALUES ($1, $2, CURRENT_TIMESTAMP)
        ON CONFLICT (id)
        DO UPDATE SET
            name = EXCLUDED.name,
            update_time = CURRENT_TIMESTAMP
        """

        await self.execute(
            upsert_workspace_sql,
            {
                "id": to_uuid(normalized_workspace_id),
                "name": normalized_workspace_name,
            },
        )
        self._registered_workspaces[normalized_workspace_id] = normalized_workspace_name

    def _create_ssl_context(self) -> ssl.SSLContext | None:
        """Create SSL context based on configuration parameters."""
        if not self.ssl_mode:
            return None

        ssl_mode = self.ssl_mode.lower()

        # For simple modes that don't require custom context
        if ssl_mode in ["disable", "allow", "prefer", "require"]:
            if ssl_mode == "disable":
                return None
            elif ssl_mode in ["require", "prefer", "allow"]:
                # Return None for simple SSL requirement, handled in initdb
                return None

        # For modes that require certificate verification
        if ssl_mode in ["verify-ca", "verify-full"]:
            try:
                context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

                # Configure certificate verification
                if ssl_mode == "verify-ca":
                    context.check_hostname = False
                elif ssl_mode == "verify-full":
                    context.check_hostname = True

                # Load root certificate if provided
                if self.ssl_root_cert:
                    if os.path.exists(self.ssl_root_cert):
                        context.load_verify_locations(cafile=self.ssl_root_cert)
                        logger.info(
                            f"PostgreSQL, Loaded SSL root certificate: {self.ssl_root_cert}"
                        )
                    else:
                        logger.warning(
                            f"PostgreSQL, SSL root certificate file not found: {self.ssl_root_cert}"
                        )

                # Load client certificate and key if provided
                if self.ssl_cert and self.ssl_key:
                    if os.path.exists(self.ssl_cert) and os.path.exists(self.ssl_key):
                        context.load_cert_chain(self.ssl_cert, self.ssl_key)
                        logger.info(
                            f"PostgreSQL, Loaded SSL client certificate: {self.ssl_cert}"
                        )
                    else:
                        logger.warning(
                            "PostgreSQL, SSL client certificate or key file not found"
                        )

                # Load certificate revocation list if provided
                if self.ssl_crl:
                    if os.path.exists(self.ssl_crl):
                        context.load_verify_locations(crlfile=self.ssl_crl)
                        logger.info(f"PostgreSQL, Loaded SSL CRL: {self.ssl_crl}")
                    else:
                        logger.warning(
                            f"PostgreSQL, SSL CRL file not found: {self.ssl_crl}"
                        )

                return context

            except Exception as e:
                logger.error(f"PostgreSQL, Failed to create SSL context: {e}")
                raise ValueError(f"SSL configuration error: {e}")

        # Unknown SSL mode
        logger.warning(f"PostgreSQL, Unknown SSL mode: {ssl_mode}, SSL disabled")
        return None

    async def initdb(self):
        logger.info(
            f"Initializing PostgreSQL connection to {self.host}:{self.port}/{self.database}"
        )

        # Prepare connection parameters
        connection_params = {
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "host": self.host,
            "port": self.port,
            "min_size": 1,
            "max_size": self.max,
        }

        # Add idle connection timeout if configured (0 = disabled)
        if (
            hasattr(self, "max_inactive_connection_lifetime")
            and self.max_inactive_connection_lifetime > 0
        ):
            connection_params["max_inactive_connection_lifetime"] = (
                self.max_inactive_connection_lifetime
            )
            logger.info(
                f"PostgreSQL, Idle connection timeout: {self.max_inactive_connection_lifetime}s"
            )

        # Only add statement_cache_size if it's configured
        if self.statement_cache_size is not None:
            connection_params["statement_cache_size"] = int(self.statement_cache_size)
            logger.info(
                f"PostgreSQL, statement LRU cache size set as: {self.statement_cache_size}"
            )

        # Add SSL configuration if provided
        ssl_context = self._create_ssl_context()
        if ssl_context is not None:
            connection_params["ssl"] = ssl_context
            logger.info("PostgreSQL, SSL configuration applied")
        elif self.ssl_mode:
            # Handle simple SSL modes without custom context
            if self.ssl_mode.lower() in ["require", "prefer"]:
                connection_params["ssl"] = True
            elif self.ssl_mode.lower() == "disable":
                connection_params["ssl"] = False
            logger.info(f"PostgreSQL, SSL mode set to: {self.ssl_mode}")

        # Add server settings if provided
        if self.server_settings:
            try:
                settings = {}
                pairs = self.server_settings.split("&")
                for pair in pairs:
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        settings[key] = value
                if settings:
                    connection_params["server_settings"] = settings
                    logger.info(f"PostgreSQL, Server settings applied: {settings}")
            except Exception as e:
                logger.warning(
                    f"PostgreSQL, Failed to parse server_settings: {self.server_settings}, error: {e}"
                )

        wait_strategy = (
            wait_exponential(
                multiplier=self.connection_retry_backoff,
                min=self.connection_retry_backoff,
                max=self.connection_retry_backoff_max,
            )
            if self.connection_retry_backoff > 0
            else wait_fixed(0)
        )

        async def _create_pool_once() -> None:
            pool = await asyncpg.create_pool(
                **connection_params,
            )
            self.pool = pool
            self._pool_loop = asyncio.get_event_loop()

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.connection_retry_attempts),
                retry=retry_if_exception_type(self._transient_exceptions),
                wait=wait_strategy,
                before_sleep=self._before_sleep,
                reraise=True,
            ):
                with attempt:
                    await _create_pool_once()

            ssl_status = "with SSL" if connection_params.get("ssl") else "without SSL"
            logger.info(
                f"PostgreSQL, Connected to database at {self.host}:{self.port}/{self.database} {ssl_status}"
            )
        except Exception as e:
            logger.error(
                f"PostgreSQL, Failed to connect database at {self.host}:{self.port}/{self.database}, Got:{e}"
            )
            raise

    def _get_health_check_lock(self) -> asyncio.Lock:
        """Get or create a loop-aware lock for pool health check."""
        current_loop = asyncio.get_event_loop()
        if (
            self._health_check_lock is None
            or self._health_check_lock_loop is not current_loop
            or self._health_check_lock_loop.is_closed()
        ):
            self._health_check_lock = asyncio.Lock()
            self._health_check_lock_loop = current_loop
        return self._health_check_lock

    def _get_reconnect_lock(self) -> asyncio.Lock:
        """Get or create a loop-aware lock for pool reconnection."""
        current_loop = asyncio.get_event_loop()
        if (
            self._pool_reconnect_lock is None
            or self._lock_loop is not current_loop
            or self._lock_loop.is_closed()
        ):
            self._pool_reconnect_lock = asyncio.Lock()
            self._lock_loop = current_loop
        return self._pool_reconnect_lock

    async def _ensure_pool(self) -> None:
        """Ensure the connection pool is initialised and healthy."""
        current_loop = asyncio.get_event_loop()

        # Check if loop changed (common in Celery prefork or multi-loop environments)
        if self.pool is not None and (
            self._pool_loop is not current_loop or self._pool_loop.is_closed()
        ):
            logger.info("PostgreSQL, Event loop changed or closed, reinitializing pool")
            self._pool_loop = None
            self.pool = None

        if self.pool is None:
            async with self._get_reconnect_lock():
                if self.pool is None:
                    await self.initdb()
        else:
            # Health check for existing pool
            try:
                async with self.pool.acquire() as conn:
                    await conn.execute("SELECT 1")
            except Exception as e:
                logger.warning(
                    f"PostgreSQL pool health check failed: {e}, reinitializing pool"
                )
                async with self._get_reconnect_lock():
                    await self._reset_pool()
                    await self.initdb()

    async def _reset_pool(self) -> None:
        if self.pool is not None:
            try:
                await asyncio.wait_for(
                    self.pool.close(), timeout=self.pool_close_timeout
                )
            except asyncio.TimeoutError:
                logger.error(
                    "PostgreSQL, Timed out closing connection pool after %.2fs",
                    self.pool_close_timeout,
                )
            except Exception as close_error:  # pragma: no cover - defensive logging
                logger.warning(
                    f"PostgreSQL, Failed to close existing connection pool cleanly: {close_error!r}"
                )
        self.pool = None

    async def _before_sleep(self, retry_state: RetryCallState) -> None:
        """Hook invoked by tenacity before sleeping between retries."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "PostgreSQL transient connection issue on attempt %s/%s: %r",
            retry_state.attempt_number,
            self.connection_retry_attempts,
            exc,
        )
        await self._reset_pool()

    async def _run_with_retry(
        self,
        operation: Callable[[asyncpg.Connection], Awaitable[T]],
    ) -> T:
        """
        Execute a database operation with automatic retry for transient failures.

        Args:
            operation: Async callable that receives an active connection.

        Returns:
            The result returned by the operation.

        Raises:
            Exception: Propagates the last error if all retry attempts fail or a non-transient error occurs.
        """
        wait_strategy = (
            wait_exponential(
                multiplier=self.connection_retry_backoff,
                min=self.connection_retry_backoff,
                max=self.connection_retry_backoff_max,
            )
            if self.connection_retry_backoff > 0
            else wait_fixed(0)
        )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.connection_retry_attempts),
            retry=retry_if_exception_type(self._transient_exceptions),
            wait=wait_strategy,
            before_sleep=self._before_sleep,
            reraise=True,
        ):
            with attempt:
                await self._ensure_pool()
                assert self.pool is not None
                async with self.pool.acquire() as connection:  # type: ignore[arg-type]
                    return await operation(connection)

    @staticmethod
    async def configure_vector_extension(connection: asyncpg.Connection) -> None:
        """Create VECTOR extension if it doesn't exist for vector similarity operations."""
        try:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")  # type: ignore
            logger.info("PostgreSQL, VECTOR extension enabled")
        except Exception as e:
            logger.warning(f"Could not create VECTOR extension: {e}")
            # Don't raise - let the system continue without vector extension

    @staticmethod
    async def configure_age_extension(connection: asyncpg.Connection) -> None:
        """Create AGE extension if it doesn't exist for graph operations."""
        try:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS AGE CASCADE")  # type: ignore
            logger.info("PostgreSQL, AGE extension enabled")
        except Exception as e:
            logger.warning(f"Could not create AGE extension: {e}")
            # Don't raise - let the system continue without AGE extension

    @staticmethod
    async def configure_age(connection: asyncpg.Connection, graph_name: str) -> None:
        """Set the Apache AGE environment and creates a graph if it does not exist.

        This method:
        - Sets the PostgreSQL `search_path` to include `ag_catalog`, ensuring that Apache AGE functions can be used without specifying the schema.
        - Attempts to create a new graph with the provided `graph_name` if it does not already exist.
        - Silently ignores errors related to the graph already existing.

        """
        try:
            await connection.execute(  # type: ignore
                'SET search_path = ag_catalog, "$user", public'
            )
            await connection.execute(  # type: ignore
                f"select create_graph('{graph_name}')"
            )
        except (
            asyncpg.exceptions.InvalidSchemaNameError,
            asyncpg.exceptions.UniqueViolationError,
        ):
            pass

    async def configure_vchordrq(self, connection: asyncpg.Connection) -> None:
        """Configure VCHORDRQ extension for vector similarity search.

        Raises:
            asyncpg.exceptions.UndefinedObjectError: If VCHORDRQ extension is not installed
            asyncpg.exceptions.InvalidParameterValueError: If parameter value is invalid

        Note:
            This method does not catch exceptions. Configuration errors will fail-fast,
            while transient connection errors will be retried by _run_with_retry.
        """
        # Handle probes parameter - only set if non-empty value is provided
        if self.vchordrq_probes and str(self.vchordrq_probes).strip():
            await connection.execute(f"SET vchordrq.probes TO '{self.vchordrq_probes}'")
            logger.debug(f"PostgreSQL, VCHORDRQ probes set to: {self.vchordrq_probes}")

        # Handle epsilon parameter independently - check for None to allow 0.0 as valid value
        if self.vchordrq_epsilon is not None:
            await connection.execute(f"SET vchordrq.epsilon TO {self.vchordrq_epsilon}")
            logger.debug(
                f"PostgreSQL, VCHORDRQ epsilon set to: {self.vchordrq_epsilon}"
            )

    async def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        multirows: bool = False,
    ) -> dict[str, Any] | None | list[dict[str, Any]]:
        async def _operation(connection: asyncpg.Connection) -> Any:
            prepared_params = tuple(params) if params else ()
            if prepared_params:
                rows = await connection.fetch(sql, *prepared_params)
            else:
                rows = await connection.fetch(sql)

            if multirows:
                if rows:
                    columns = [col for col in rows[0].keys()]
                    return [dict(zip(columns, row)) for row in rows]
                return []

            if rows:
                columns = rows[0].keys()
                return dict(zip(columns, rows[0]))
            return None

        try:
            return await self._run_with_retry(_operation)
        except Exception as e:
            logger.error(f"PostgreSQL database, error:{e}")
            raise

    async def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in PostgreSQL database

        Args:
            table_name: Name of the table to check

        Returns:
            bool: True if table exists, False otherwise
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = $1
            )
        """
        result = await self.query(query, [table_name.lower()])
        return result.get("exists", False) if result else False

    async def execute(
        self,
        sql: str,
        data: dict[str, Any] | list[Any] | tuple[Any, ...] | None = None,
        upsert: bool = False,
        ignore_if_exists: bool = False,
    ):
        async def _operation(connection: asyncpg.Connection) -> Any:
            if isinstance(data, dict):
                prepared_values = tuple(data.values())
            elif data:
                prepared_values = tuple(data)
            else:
                prepared_values = ()
            try:
                if not data:
                    return await connection.execute(sql)
                return await connection.execute(sql, *prepared_values)
            except (
                asyncpg.exceptions.UniqueViolationError,
                asyncpg.exceptions.DuplicateTableError,
                asyncpg.exceptions.DuplicateObjectError,
                asyncpg.exceptions.InvalidSchemaNameError,
            ) as e:
                if ignore_if_exists:
                    logger.debug("PostgreSQL, ignoring duplicate during execute: %r", e)
                    return None
                if upsert:
                    logger.info(
                        "PostgreSQL, duplicate detected but treated as upsert success: %r",
                        e,
                    )
                    return None
                raise

        try:
            await self._run_with_retry(_operation)
        except Exception as e:
            logger.error(f"PostgreSQL database,\nsql:{sql},\ndata:{data},\nerror:{e}")
            raise


class ClientManager:
    # Change to global singleton instead of per-event-loop
    _shared_instance: dict[str, Any] = {
        "db": None,
        "ref_count": 0,
        "lock": asyncio.Lock(),
    }

    @classmethod
    def _get_ctx(cls) -> int:
        # Always return 0 to force global sharing
        return 0

    @classmethod
    def _get_shared_lock(cls) -> asyncio.Lock:
        """Return the shared asyncio.Lock for synchronizing pool access."""
        return cls._shared_instance["lock"]

    @staticmethod
    def get_config() -> dict[str, Any]:
        config = configparser.ConfigParser()

        # Try to read config.ini, but don't fail if it doesn't exist
        try:
            config.read("config.ini", "utf-8")
        except Exception as e:
            logger.debug(f"Could not read config.ini: {e}")

        # Reduce default max_connections since it's now shared across all storage types
        default_max_connections = 30  # Reduced from 50

        # Helper function to get config with validation
        def get_config_value(
            env_key: str,
            config_section: str,
            config_key: str,
            fallback: Any,
            required: bool = False,
        ) -> Any:
            value = os.environ.get(env_key)
            if value is None:
                try:
                    value = config.get(config_section, config_key, fallback=fallback)
                except (configparser.NoSectionError, configparser.NoOptionError):
                    value = fallback

            # Validate required parameters
            if required and (
                value is None or (isinstance(value, str) and not value.strip())
            ):
                raise ValueError(
                    f"Required PostgreSQL parameter '{env_key}' is missing or empty"
                )

            return value

        return {
            "host": get_config_value("POSTGRES_HOST", "postgres", "host", "localhost"),
            "port": get_config_value("POSTGRES_PORT", "postgres", "port", 5432),
            "user": get_config_value(
                "POSTGRES_USER", "postgres", "user", None, required=True
            ),
            "password": get_config_value(
                "POSTGRES_PASSWORD", "postgres", "password", None, required=True
            ),
            "database": get_config_value(
                "POSTGRES_DATABASE", "postgres", "database", None, required=True
            ),
            "workspace": get_config_value(
                "POSTGRES_WORKSPACE", "postgres", "workspace", None
            ),
            "max_connections": get_config_value(
                "POSTGRES_MAX_CONNECTIONS",
                "postgres",
                "max_connections",
                default_max_connections,
            ),
            # SSL configuration
            "ssl_mode": get_config_value(
                "POSTGRES_SSL_MODE", "postgres", "ssl_mode", None
            ),
            "ssl_cert": get_config_value(
                "POSTGRES_SSL_CERT", "postgres", "ssl_cert", None
            ),
            "ssl_key": get_config_value(
                "POSTGRES_SSL_KEY", "postgres", "ssl_key", None
            ),
            "ssl_root_cert": get_config_value(
                "POSTGRES_SSL_ROOT_CERT", "postgres", "ssl_root_cert", None
            ),
            "ssl_crl": get_config_value(
                "POSTGRES_SSL_CRL", "postgres", "ssl_crl", None
            ),
            # Server settings for Supabase
            "server_settings": get_config_value(
                "POSTGRES_SERVER_SETTINGS", "postgres", "server_options", None
            ),
            "statement_cache_size": get_config_value(
                "POSTGRES_STATEMENT_CACHE_SIZE",
                "postgres",
                "statement_cache_size",
                None,
            ),
            # Connection retry configuration
            "connection_retry_attempts": min(
                100,
                int(
                    get_config_value(
                        "POSTGRES_CONNECTION_RETRIES",
                        "postgres",
                        "connection_retries",
                        10,
                    )
                ),
            ),
            "connection_retry_backoff": min(
                300.0,
                float(
                    get_config_value(
                        "POSTGRES_CONNECTION_RETRY_BACKOFF",
                        "postgres",
                        "connection_retry_backoff",
                        3.0,
                    )
                ),
            ),
            "connection_retry_backoff_max": min(
                600.0,
                float(
                    get_config_value(
                        "POSTGRES_CONNECTION_RETRY_BACKOFF_MAX",
                        "postgres",
                        "connection_retry_backoff_max",
                        30.0,
                    )
                ),
            ),
            "pool_close_timeout": min(
                30.0,
                float(
                    get_config_value(
                        "POSTGRES_POOL_CLOSE_TIMEOUT",
                        "postgres",
                        "pool_close_timeout",
                        5.0,
                    )
                ),
            ),
            # Connection pool idle timeout (0 = disabled)
            "max_inactive_connection_lifetime": max(
                0.0,
                float(
                    get_config_value(
                        "POSTGRES_MAX_INACTIVE_CONNECTION_LIFETIME",
                        "postgres",
                        "max_inactive_connection_lifetime",
                        300.0,
                    )
                ),
            ),
        }

    @classmethod
    async def get_client(cls) -> PostgreSQLDB:
        # Use global shared instance instead of per-event-loop
        async with cls._get_shared_lock():
            if cls._shared_instance["db"] is None:
                try:
                    config = ClientManager.get_config()

                    logger.info(
                        f"Creating shared PostgreSQL connection pool with {config['max_connections']} connections"
                    )
                    logger.info(
                        f"Connecting to PostgreSQL at {config['host']}:{config['port']}/{config['database']} as user '{config['user']}'"
                    )

                    db = PostgreSQLDB(config)
                    await db.initdb()
                    cls._shared_instance["db"] = db
                    cls._shared_instance["ref_count"] = 0

                    logger.info(
                        "Shared PostgreSQL connection pool created successfully"
                    )
                except Exception as e:
                    logger.error(f"Failed to create PostgreSQL connection pool: {e}")
                    # Reset the instance to allow retry
                    cls._shared_instance["db"] = None
                    raise

            cls._shared_instance["ref_count"] += 1
            logger.debug(
                f"PostgreSQL shared pool reference count: {cls._shared_instance['ref_count']}"
            )
            return cls._shared_instance["db"]

    @classmethod
    async def release_client(cls, db: PostgreSQLDB):
        # Use global shared instance
        async with cls._get_shared_lock():
            if db is not None:
                if db is cls._shared_instance["db"]:
                    cls._shared_instance["ref_count"] -= 1
                    logger.debug(
                        f"PostgreSQL shared pool reference count: {cls._shared_instance['ref_count']}"
                    )

                    # Don't close pool immediately, let it stay for reuse
                    # Only close when explicitly requested or during shutdown
                    if cls._shared_instance["ref_count"] <= 0:
                        logger.info("All references to shared PostgreSQL pool released")
                else:
                    # This shouldn't happen with shared pool, but handle gracefully
                    if db.pool:
                        await db.pool.close()
                        logger.warning("Closed non-shared PostgreSQL pool (unexpected)")

    @classmethod
    async def force_close_shared_pool(cls):
        """Force close the shared pool during shutdown."""
        async with cls._get_shared_lock():
            if cls._shared_instance["db"] is not None:
                if cls._shared_instance["db"].pool:
                    await cls._shared_instance["db"].pool.close()
                logger.info("Force closed shared PostgreSQL connection pool")
                cls._shared_instance["db"] = None
                cls._shared_instance["ref_count"] = 0

    @classmethod
    def get_pool_stats(cls) -> dict[str, Any]:
        """Get statistics about the shared connection pool."""
        if (
            cls._shared_instance["db"] is None
            or cls._shared_instance["db"].pool is None
        ):
            return {
                "status": "not_created",
                "ref_count": cls._shared_instance["ref_count"],
            }

        pool = cls._shared_instance["db"].pool
        return {
            "status": "active",
            "ref_count": cls._shared_instance["ref_count"],
            "size": pool.get_size(),
            "min_size": pool.get_min_size(),
            "max_size": pool.get_max_size(),
            "idle_size": pool.get_idle_size(),
        }

    @classmethod
    async def test_connection(cls) -> dict[str, Any]:
        """Test PostgreSQL connection with current configuration."""
        try:
            config = cls.get_config()

            # Test basic connection parameters
            test_conn = await asyncpg.connect(
                user=config["user"],
                password=config["password"],
                database=config["database"],
                host=config["host"],
                port=config["port"],
                timeout=10.0,  # 10 second timeout for test
            )

            # Test basic query
            result = await test_conn.fetchval("SELECT version()")
            await test_conn.close()

            return {
                "status": "success",
                "message": "Connection successful",
                "server_version": result,
                "config": {
                    "host": config["host"],
                    "port": config["port"],
                    "database": config["database"],
                    "user": config["user"],
                    "max_connections": config["max_connections"],
                },
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection failed: {str(e)}",
                "error_type": type(e).__name__,
            }


@final
@dataclass
class PGKVStorage(BaseKVStorage):
    db: PostgreSQLDB = field(default=None)

    def __post_init__(self):
        self._max_batch_size = 200  # DB batch size, independent of embedding batch size

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            explicit_storage_workspace = self.global_config.get(
                "_storage_workspace_explicit", False
            )
            # An explicit generation workspace must not be overridden by the
            # process-wide PostgreSQL workspace setting.
            if explicit_storage_workspace:
                self.workspace = self.workspace or "default"
            elif self.db.workspace:
                # Use PostgreSQLDB's workspace (highest priority)
                logger.info(
                    f"Using PG_WORKSPACE environment variable: '{self.db.workspace}' (overriding '{self.workspace}/{self.namespace}')"
                )
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

            try:
                await self.db.ensure_workspace_metadata(self.workspace)
            except Exception as e:
                logger.warning(
                    f"Failed to upsert workspace metadata for '{self.workspace}': {e}"
                )

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    ################ QUERY METHODS ################
    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get data by id."""
        if is_namespace(
            self.namespace, NameSpace.KV_STORE_FULL_ENTITIES
        ) or is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            return None

        if is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            from ..utils import parse_relation_chunk_key

            try:
                entity1, entity2 = parse_relation_chunk_key(id)
            except Exception:
                entity1, entity2 = id, id

            sql = SQL_TEMPLATES["get_by_id_relation_chunks"]
            params = {
                "workspace": to_uuid(self.workspace),
                "entity1": entity1,
                "entity2": entity2,
            }
            response = await self.db.query(sql, list(params.values()))
            if response:
                chunk_ids = response.get("chunk_ids")
                if chunk_ids is None:
                    return None
                if isinstance(chunk_ids, str):
                    try:
                        chunk_ids = json.loads(chunk_ids)
                    except json.JSONDecodeError:
                        chunk_ids = []
                if not chunk_ids:
                    return None
                response["chunk_ids"] = chunk_ids
            return response if response else None

        sql = SQL_TEMPLATES["get_by_id_" + self.namespace]
        target_id = id
        if (
            is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS)
            or is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS)
            or is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE)
        ):
            target_id = to_uuid(id)
        params = {"workspace": to_uuid(self.workspace), "id": target_id}
        response = await self.db.query(sql, list(params.values()))

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list
            llm_cache_list = response.get("llm_cache_list", [])
            if isinstance(llm_cache_list, str):
                try:
                    llm_cache_list = json.loads(llm_cache_list)
                except json.JSONDecodeError:
                    llm_cache_list = []
            response["llm_cache_list"] = llm_cache_list
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            response["create_time"] = create_time
            response["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if response and is_namespace(
            self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
        ):
            create_time = response.get("create_time", 0)
            update_time = response.get("update_time", 0)
            # Parse queryparam JSON string back to dict
            queryparam = response.get("queryparam")
            if isinstance(queryparam, str):
                try:
                    queryparam = json.loads(queryparam)
                except json.JSONDecodeError:
                    queryparam = None
            # Map field names for compatibility (mode field removed)
            response = {
                **response,
                "return": response.get("return_value", ""),
                "cache_type": response.get("cache_type").lower()
                if response.get("cache_type")
                else "extract",
                "original_prompt": response.get("original_prompt", ""),
                "chunk_id": response.get("chunk_id"),
                "queryparam": queryparam,
                "create_time": create_time,
                "update_time": create_time if update_time == 0 else update_time,
            }

        if response and is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            # Parse chunk_ids JSON string back to list
            chunk_ids = response.get("chunk_ids", [])
            if isinstance(chunk_ids, str):
                try:
                    chunk_ids = json.loads(chunk_ids)
                except json.JSONDecodeError:
                    chunk_ids = []
            response["chunk_ids"] = chunk_ids

        return response if response else None

    # Query by id
    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get data by ids"""
        if not ids:
            return []

        if is_namespace(
            self.namespace, NameSpace.KV_STORE_FULL_ENTITIES
        ) or is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            return []

        if is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            results = []
            for doc_id in ids:
                res = await self.get_by_id(doc_id)
                if res:
                    results.append(res)
            return results

        sql = SQL_TEMPLATES["get_by_ids_" + self.namespace]
        target_ids = ids
        if (
            is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS)
            or is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS)
            or is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE)
        ):
            target_ids = [to_uuid(x) for x in ids]
        params = {"workspace": to_uuid(self.workspace), "ids": target_ids}
        results = await self.db.query(sql, list(params.values()), multirows=True)

        def _order_results(
            rows: list[dict[str, Any]] | None,
        ) -> list[dict[str, Any] | None]:
            """Preserve the caller requested ordering for bulk id lookups."""
            if not rows:
                return [None for _ in ids]

            id_map: dict[str, dict[str, Any]] = {}
            for row in rows:
                if row is None:
                    continue
                row_id = row.get("id")
                if row_id is not None:
                    id_map[str(row_id)] = row

            ordered: list[dict[str, Any] | None] = []
            for requested_id in ids:
                key = str(requested_id)
                if key in id_map:
                    ordered.append(id_map[key])
                else:
                    try:
                        uuid_key = str(to_uuid(requested_id))
                        ordered.append(id_map.get(uuid_key))
                    except Exception:
                        ordered.append(None)
            return ordered

        if results and is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Parse llm_cache_list JSON string back to list for each result
            for result in results:
                llm_cache_list = result.get("llm_cache_list", [])
                if isinstance(llm_cache_list, str):
                    try:
                        llm_cache_list = json.loads(llm_cache_list)
                    except json.JSONDecodeError:
                        llm_cache_list = []
                result["llm_cache_list"] = llm_cache_list
                create_time = result.get("create_time", 0)
                update_time = result.get("update_time", 0)
                result["create_time"] = create_time
                result["update_time"] = create_time if update_time == 0 else update_time

        # Special handling for LLM cache to ensure compatibility with _get_cached_extraction_results
        if results and is_namespace(
            self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE
        ):
            processed_results = []
            for row in results:
                create_time = row.get("create_time", 0)
                update_time = row.get("update_time", 0)
                # Parse queryparam JSON string back to dict
                queryparam = row.get("queryparam")
                if isinstance(queryparam, str):
                    try:
                        queryparam = json.loads(queryparam)
                    except json.JSONDecodeError:
                        queryparam = None
                # Map field names for compatibility (mode field removed)
                processed_row = {
                    **row,
                    "return": row.get("return_value", ""),
                    "cache_type": row.get("cache_type").lower()
                    if row.get("cache_type")
                    else "extract",
                    "original_prompt": row.get("original_prompt", ""),
                    "chunk_id": row.get("chunk_id"),
                    "queryparam": queryparam,
                    "create_time": create_time,
                    "update_time": create_time if update_time == 0 else update_time,
                }
                processed_results.append(processed_row)
            return _order_results(processed_results)

        if results and is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            for row in results:
                chunk_ids = row.get("chunk_ids", [])
                if isinstance(chunk_ids, str):
                    try:
                        chunk_ids = json.loads(chunk_ids)
                    except json.JSONDecodeError:
                        chunk_ids = []
                row["chunk_ids"] = chunk_ids

        return _order_results(results)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter out duplicated content"""
        if not keys:
            return set()

        table_name = namespace_to_table_name(self.namespace)
        sql = f"SELECT id FROM {table_name} WHERE workspace_id=$1 AND id = ANY($2)"
        target_ids = list(keys)
        if (
            is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS)
            or is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS)
            or is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE)
        ):
            target_ids = [to_uuid(x) for x in keys]
        params = {"workspace": to_uuid(self.workspace), "ids": target_ids}
        try:
            res = await self.db.query(sql, list(params.values()), multirows=True)
            if res:
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            new_keys = set([s for s in keys if s not in exist_keys])
            return new_keys
        except Exception as e:
            logger.error(
                f"[{self.workspace}] PostgreSQL database,\nsql:{sql},\nparams:{params},\nerror:{e}"
            )
            raise

    ################ INSERT METHODS ################
    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        batch_values: list[tuple] = []
        upsert_sql = ""

        if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
            # Check if this is a legal document by looking at the first item
            is_legal = False
            if data:
                first_val = next(iter(data.values()))
                if "legal_document_id" in first_val and first_val["legal_document_id"]:
                    is_legal = True

            if is_legal:
                upsert_sql = """
                    INSERT INTO tbl_lightrag_document_chunk 
                    (legal_document_id, id, tokens, order_index, content, create_time, update_time)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (id) DO UPDATE SET 
                        content = EXCLUDED.content,
                        update_time = EXCLUDED.update_time,
                        tokens = EXCLUDED.tokens,
                        order_index = EXCLUDED.order_index
                """
            else:
                upsert_sql = SQL_TEMPLATES["upsert_text_chunk"]

            # Get current UTC time and convert to naive datetime for database storage
            current_time = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            for k, v in data.items():
                if is_legal:
                    # Tuple order: (document_id (int), id (uuid), tokens, order_index, content, create_time, update_time)
                    batch_values.append(
                        (
                            int(v["legal_document_id"]),
                            to_uuid(k),
                            v["tokens"],
                            v["chunk_order_index"],
                            v["content"],
                            current_time,
                            current_time,
                        )
                    )
                else:
                    # Tuple order: (document_id (uuid), id (uuid), tokens, order_index, content, create_time, update_time)
                    batch_values.append(
                        (
                            to_uuid(v["full_doc_id"]),
                            to_uuid(k),
                            v["tokens"],
                            v["chunk_order_index"],
                            v["content"],
                            current_time,
                            current_time,
                        )
                    )
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
            upsert_sql = SQL_TEMPLATES["upsert_doc_full"]
            for k, v in data.items():
                # Tuple order must match SQL: (id, content, doc_name, workspace)
                batch_values.append(
                    (
                        to_uuid(k),
                        v["content"],
                        v.get("file_path", ""),
                        to_uuid(self.workspace),
                    )
                )
        elif is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
            upsert_sql = SQL_TEMPLATES["upsert_llm_response_cache"]
            for k, v in data.items():
                # Tuple order must match SQL: (workspace, id, original_prompt, return_value,
                #   chunk_id, cache_type, queryparam)
                batch_values.append(
                    (
                        to_uuid(self.workspace),
                        to_uuid(k),
                        v["original_prompt"],
                        v["return"],
                        to_uuid(v.get("chunk_id")) if v.get("chunk_id") else None,
                        str(v.get("cache_type") or "extract").upper(),
                        json.dumps(v.get("queryparam"))
                        if v.get("queryparam")
                        else None,
                    )
                )
        elif (
            is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS)
            or is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS)
            or is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES)
            or is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS)
        ):
            return
        else:
            logger.error(f"Unknown namespace: {self.namespace}")
            raise ValueError(f"Unknown namespace: {self.namespace}")

        # upsert_sql is always set here; unknown namespace raises ValueError above
        if batch_values:
            # Split into sub-batches to prevent database overload
            for i in range(0, len(batch_values), self._max_batch_size):
                sub_batch = batch_values[i : i + self._max_batch_size]

                async def _batch_upsert(
                    connection: asyncpg.Connection,
                    _sql: str = upsert_sql,
                    _data: list[tuple] = sub_batch,
                ) -> None:
                    await connection.executemany(_sql, _data)

                await self.db._run_with_retry(_batch_upsert)

            num_batches = (
                len(batch_values) + self._max_batch_size - 1
            ) // self._max_batch_size
            logger.debug(
                f"[{self.workspace}] Batch upserted {len(batch_values)} records to {self.namespace} "
                f"in {num_batches} sub-batches"
            )

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def is_empty(self) -> bool:
        """Check if the storage is empty for the current workspace and namespace

        Returns:
            bool: True if storage is empty, False otherwise
        """
        if is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
            sql = """
                SELECT EXISTS(
                    SELECT 1 
                    FROM tbl_chunk_entity c
                    JOIN tbl_lightrag_entity e ON c.entity_id = e.id
                    WHERE e.workspace_id = $1
                    LIMIT 1
                ) as has_data
            """
        elif is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
            sql = """
                SELECT EXISTS(
                    SELECT 1 
                    FROM tbl_chunk_relation cr
                    JOIN tbl_lightrag_relation r ON cr.relation_id = r.id
                    WHERE r.workspace_id = $1
                    LIMIT 1
                ) as has_data
            """
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
            sql = "SELECT EXISTS(SELECT 1 FROM tbl_lightrag_entity WHERE workspace_id = $1 LIMIT 1) as has_data"
        elif is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
            sql = "SELECT EXISTS(SELECT 1 FROM tbl_lightrag_relation WHERE workspace_id = $1 LIMIT 1) as has_data"
        else:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                logger.error(
                    f"[{self.workspace}] Unknown namespace for is_empty check: {self.namespace}"
                )
                return True
            sql = f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE workspace_id=$1 LIMIT 1) as has_data"

        try:
            result = await self.db.query(sql, [to_uuid(self.workspace)])
            return not result.get("has_data", False) if result else True
        except Exception as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs."""
        if not ids:
            return

        logger.info(
            f"[{self.workspace}] Deleting {len(ids)} records from {self.namespace}"
        )

        try:
            # 1. KV_STORE_TEXT_CHUNKS (tbl_lightrag_document_chunk)
            if is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
                target_ids = [to_uuid(x) for x in ids]

                # Delete associated records in tbl_chunk_entity first
                await self.db.execute(
                    "DELETE FROM tbl_chunk_entity WHERE chunk_id = ANY($1)",
                    [target_ids],
                )

                # Delete associated records in tbl_chunk_relation
                await self.db.execute(
                    "DELETE FROM tbl_chunk_relation WHERE chunk_id = ANY($1)",
                    [target_ids],
                )

                # Delete associated records in tbl_lightrag_llm_cache
                await self.db.execute(
                    "DELETE FROM tbl_lightrag_llm_cache WHERE chunk_id = ANY($1)",
                    [target_ids],
                )

                # Delete from tbl_lightrag_document_chunk
                await self.db.execute(
                    "DELETE FROM tbl_lightrag_document_chunk WHERE id = ANY($1)",
                    [target_ids],
                )
                return

            # 2. KV_STORE_FULL_DOCS or DOC_STATUS (tbl_document)
            elif is_namespace(
                self.namespace, NameSpace.KV_STORE_FULL_DOCS
            ) or is_namespace(self.namespace, NameSpace.DOC_STATUS):
                target_ids = [to_uuid(x) for x in ids]

                # Delete chunk associations first because Cascade on tbl_lightrag_document_chunk
                # does not automatically cascade to tbl_chunk_entity/tbl_chunk_relation
                await self.db.execute(
                    """
                    DELETE FROM tbl_chunk_entity 
                    WHERE chunk_id IN (
                        SELECT id FROM tbl_lightrag_document_chunk WHERE document_id = ANY($1)
                    )
                    """,
                    [target_ids],
                )
                await self.db.execute(
                    """
                    DELETE FROM tbl_chunk_relation 
                    WHERE chunk_id IN (
                        SELECT id FROM tbl_lightrag_document_chunk WHERE document_id = ANY($1)
                    )
                    """,
                    [target_ids],
                )

                # Now we can delete from tbl_document, which will cascade delete tbl_lightrag_document_chunk
                # and tbl_lightrag_llm_cache
                await self.db.execute(
                    "DELETE FROM tbl_document WHERE workspace_id = $1 AND document_id = ANY($2)",
                    [to_uuid(self.workspace), target_ids],
                )
                return

            # 3. KV_STORE_LLM_RESPONSE_CACHE (tbl_lightrag_llm_cache)
            elif is_namespace(self.namespace, NameSpace.KV_STORE_LLM_RESPONSE_CACHE):
                target_ids = [to_uuid(x) for x in ids]
                await self.db.execute(
                    "DELETE FROM tbl_lightrag_llm_cache WHERE workspace_id = $1 AND id = ANY($2)",
                    [to_uuid(self.workspace), target_ids],
                )
                return

            # 4. KV_STORE_ENTITY_CHUNKS (chunk mapping to entity)
            elif is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
                await self.db.execute(
                    """
                    DELETE FROM tbl_chunk_entity 
                    WHERE entity_id IN (
                        SELECT id FROM tbl_lightrag_entity 
                        WHERE workspace_id = $1 AND name = ANY($2)
                    )
                    """,
                    [to_uuid(self.workspace), ids],
                )
                return

            # 5. KV_STORE_RELATION_CHUNKS (chunk mapping to relation)
            elif is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
                from ..utils import parse_relation_chunk_key

                for rel_id in ids:
                    try:
                        entity1, entity2 = parse_relation_chunk_key(rel_id)
                    except Exception:
                        entity1, entity2 = rel_id, rel_id

                    await self.db.execute(
                        """
                        DELETE FROM tbl_chunk_relation 
                        WHERE relation_id IN (
                            SELECT r.id 
                            FROM tbl_lightrag_relation r
                            JOIN tbl_lightrag_entity e1 ON r.source_entity_id = e1.id
                            JOIN tbl_lightrag_entity e2 ON r.target_entity_id = e2.id
                            WHERE r.workspace_id = $1 
                              AND (
                                (e1.name = $2 AND e2.name = $3) OR 
                                (e1.name = $3 AND e2.name = $2)
                              )
                        )
                        """,
                        [to_uuid(self.workspace), entity1, entity2],
                    )
                return

            # Fallback for any other namespaces
            else:
                table_name = namespace_to_table_name(self.namespace)
                if not table_name:
                    logger.error(
                        f"[{self.workspace}] Unknown namespace for deletion: {self.namespace}"
                    )
                    return

                await self.db.execute(
                    f"DELETE FROM {table_name} WHERE workspace_id = $1 AND id = ANY($2)",
                    [to_uuid(self.workspace), ids],
                )

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}",
                exc_info=True,
            )

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            workspace_uuid = to_uuid(self.workspace)

            if is_namespace(self.namespace, NameSpace.KV_STORE_ENTITY_CHUNKS):
                await self.db.execute(
                    """
                    DELETE FROM tbl_chunk_entity
                    WHERE entity_id IN (
                        SELECT id FROM tbl_lightrag_entity WHERE workspace_id = $1
                    )
                    """,
                    {"workspace": workspace_uuid},
                )
                return {"status": "success", "message": "data dropped"}

            if is_namespace(self.namespace, NameSpace.KV_STORE_RELATION_CHUNKS):
                await self.db.execute(
                    """
                    DELETE FROM tbl_chunk_relation
                    WHERE relation_id IN (
                        SELECT id FROM tbl_lightrag_relation WHERE workspace_id = $1
                    )
                    """,
                    {"workspace": workspace_uuid},
                )
                return {"status": "success", "message": "data dropped"}

            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_RELATIONS):
                await self.db.execute(
                    """
                    DELETE FROM tbl_chunk_relation
                    WHERE relation_id IN (
                        SELECT id FROM tbl_lightrag_relation WHERE workspace_id = $1
                    )
                    """,
                    {"workspace": workspace_uuid},
                )
                await self.db.execute(
                    "DELETE FROM tbl_lightrag_relation WHERE workspace_id = $1",
                    {"workspace": workspace_uuid},
                )
                return {"status": "success", "message": "data dropped"}

            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_ENTITIES):
                await self.db.execute(
                    """
                    DELETE FROM tbl_chunk_relation
                    WHERE relation_id IN (
                        SELECT id FROM tbl_lightrag_relation WHERE workspace_id = $1
                    )
                    """,
                    {"workspace": workspace_uuid},
                )
                await self.db.execute(
                    "DELETE FROM tbl_lightrag_relation WHERE workspace_id = $1",
                    {"workspace": workspace_uuid},
                )
                await self.db.execute(
                    """
                    DELETE FROM tbl_chunk_entity
                    WHERE entity_id IN (
                        SELECT id FROM tbl_lightrag_entity WHERE workspace_id = $1
                    )
                    """,
                    {"workspace": workspace_uuid},
                )
                await self.db.execute(
                    "DELETE FROM tbl_lightrag_entity WHERE workspace_id = $1",
                    {"workspace": workspace_uuid},
                )
                return {"status": "success", "message": "data dropped"}

            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specifiy_table_workspace"].format(
                table_name=table_name
            )
            await self.db.execute(drop_sql, {"workspace": workspace_uuid})
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}


@final
@dataclass
@final
@dataclass
class PGDocStatusStorage(DocStatusStorage):
    db: PostgreSQLDB = field(default=None)

    @staticmethod
    def _strip_document_prefix(id_str: str) -> str:
        clean_id = str(id_str)
        for prefix in ["chunk-", "ent-", "rel-", "doc-", "legal-", "node-", "edge-"]:
            if clean_id.startswith(prefix):
                return clean_id[len(prefix):]
        return clean_id

    @classmethod
    def _legal_document_id_from_key(cls, key: str) -> int | None:
        clean_id = cls._strip_document_prefix(str(key))
        if clean_id.isdigit():
            return int(clean_id)
        return None

    def _format_datetime_with_timezone(self, dt):
        """Convert datetime to ISO format string with timezone info"""
        if dt is None:
            return None
        # If no timezone info, assume it's UTC time (as stored in database)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # If datetime already has timezone info, keep it as is
        return dt.isoformat()

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            explicit_storage_workspace = self.global_config.get(
                "_storage_workspace_explicit", False
            )
            # An explicit generation workspace must not be overridden by the
            # process-wide PostgreSQL workspace setting.
            if explicit_storage_workspace:
                self.workspace = self.workspace or "default"
            elif self.db.workspace:
                # Use PostgreSQLDB's workspace (highest priority)
                logger.info(
                    f"Using PG_WORKSPACE environment variable: '{self.db.workspace}' (overriding '{self.workspace}/{self.namespace}')"
                )
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

            try:
                await self.db.ensure_workspace_metadata(self.workspace)
            except Exception as e:
                logger.warning(
                    f"Failed to upsert workspace metadata for '{self.workspace}': {e}"
                )

            # NOTE: Table creation is handled by PostgreSQLDB.initdb() during initialization
            # No need to create table here as it's already created in the TABLES dict

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter out duplicated content"""
        if not keys:
            return set()

        table_name = namespace_to_table_name(self.namespace)
        sql = f"SELECT id FROM {table_name} WHERE workspace_id=$1 AND id = ANY($2)"
        params = {"workspace": self.workspace, "ids": list(keys)}
        try:
            res = await self.db.query(sql, list(params.values()), multirows=True)
            if res:
                exist_keys = [key["id"] for key in res]
            else:
                exist_keys = []
            new_keys = set([s for s in keys if s not in exist_keys])
            # print(f"keys: {keys}")
            # print(f"new_keys: {new_keys}")
            return new_keys
        except Exception as e:
            logger.error(
                f"[{self.workspace}] PostgreSQL database,\nsql:{sql},\nparams:{params},\nerror:{e}"
            )
            raise

    async def get_by_id(self, id: str) -> Union[dict[str, Any], None]:
        id_str = str(id)
        # Check if it's an integer ID (for tbl_legal_document)
        is_int_id = False
        int_id = None
        # Strip common prefixes first
        clean_id = self._strip_document_prefix(id_str)
        if clean_id.isdigit():
            is_int_id = True
            int_id = int(clean_id)

        if is_int_id:
            sql = """
                SELECT 
                    workspace_id,
                    id::text AS document_id,
                    summary AS content_summary,
                    length(COALESCE(summary, '')) AS content_length,
                    COALESCE(process_status, 'EXTRACTED') AS status,
                    title AS document_name,
                    NULL::text AS content_hash,
                    NULL::text AS error_msg,
                    created_at AS create_time,
                    updated_at AS update_time
                FROM "tbl_legal_document"
                WHERE "workspace_id"=$1 AND "id"=$2
            """
            params = {"workspace": to_uuid(self.workspace), "id": int_id}
        else:
            sql = """
                SELECT 
                    workspace_id,
                    document_id::text AS document_id,
                    content_summary,
                    content_length,
                    status,
                    document_name,
                    content_hash,
                    error_msg,
                    create_time,
                    update_time
                FROM "tbl_document"
                WHERE "workspace_id"=$1 AND "document_id"=$2
            """
            params = {"workspace": to_uuid(self.workspace), "id": to_uuid(id_str)}

        result = await self.db.query(sql, list(params.values()), True)
        if result is None or result == []:
            return None
        else:
            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(result[0]["create_time"])
            updated_at = self._format_datetime_with_timezone(result[0]["update_time"])

            return dict(
                id=str(result[0]["document_id"]),
                content_length=result[0]["content_length"],
                content_summary=result[0]["content_summary"],
                status=result[0]["status"],
                created_at=created_at,
                updated_at=updated_at,
                file_path=result[0]["document_name"],
                metadata={},
                error_msg=result[0].get("error_msg"),
                content_hash=result[0].get("content_hash"),
            )

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get doc_chunks data by multiple IDs from both tbl_document and tbl_legal_document."""
        if not ids:
            return []

        int_ids = []
        uuid_ids = []
        id_to_type = {}

        for requested_id in ids:
            id_str = str(requested_id)
            clean_id = self._strip_document_prefix(id_str)
            if clean_id.isdigit():
                int_ids.append(int(clean_id))
                id_to_type[id_str] = ('int', clean_id)
            else:
                uuid_ids.append(to_uuid(id_str))
                id_to_type[id_str] = ('uuid', to_uuid(id_str))

        processed_map: dict[str, dict[str, Any]] = {}

        # 1. Query tbl_document
        if uuid_ids:
            sql_doc = """
                SELECT 
                    workspace_id,
                    document_id,
                    content_summary,
                    content_length,
                    status,
                    document_name,
                    content_hash,
                    error_msg,
                    create_time,
                    update_time
                FROM "tbl_document"
                WHERE "workspace_id"=$1 AND "document_id" = ANY($2)
            """
            params_doc = {"workspace": to_uuid(self.workspace), "ids": uuid_ids}
            results_doc = await self.db.query(sql_doc, list(params_doc.values()), True)
            if results_doc:
                for row in results_doc:
                    created_at = self._format_datetime_with_timezone(row["create_time"])
                    updated_at = self._format_datetime_with_timezone(row["update_time"])
                    doc_id_str = str(row.get("document_id"))
                    processed_map[doc_id_str] = {
                        "id": doc_id_str,
                        "content_length": row["content_length"],
                        "content_summary": row["content_summary"],
                        "status": row["status"],
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "file_path": row["document_name"],
                        "metadata": {},
                        "error_msg": row.get("error_msg"),
                        "content_hash": row.get("content_hash"),
                    }

        # 2. Query tbl_legal_document
        if int_ids:
            sql_legal = """
                SELECT 
                    workspace_id,
                    id,
                    summary AS content_summary,
                    length(COALESCE(summary, '')) AS content_length,
                    COALESCE(process_status, 'EXTRACTED') AS status,
                    title AS document_name,
                    NULL::text AS content_hash,
                    NULL::text AS error_msg,
                    created_at AS create_time,
                    updated_at AS update_time
                FROM "tbl_legal_document"
                WHERE "workspace_id"=$1 AND "id" = ANY($2)
            """
            params_legal = {"workspace": to_uuid(self.workspace), "ids": int_ids}
            results_legal = await self.db.query(sql_legal, list(params_legal.values()), True)
            if results_legal:
                for row in results_legal:
                    created_at = self._format_datetime_with_timezone(row["create_time"])
                    updated_at = self._format_datetime_with_timezone(row["update_time"])
                    doc_id_str = str(row.get("id"))
                    processed_map[doc_id_str] = {
                        "id": doc_id_str,
                        "content_length": row["content_length"],
                        "content_summary": row["content_summary"],
                        "status": row["status"],
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "file_path": row["document_name"],
                        "metadata": {},
                        "error_msg": row.get("error_msg"),
                        "content_hash": row.get("content_hash"),
                    }

        ordered_results: list[dict[str, Any] | None] = []
        for requested_id in ids:
            id_str = str(requested_id)
            res_item = processed_map.get(id_str)
            if not res_item:
                info = id_to_type.get(id_str)
                if info:
                    res_item = processed_map.get(str(info[1]))
            ordered_results.append(res_item)

        return ordered_results

    async def get_doc_by_file_path(self, file_path: str) -> Union[dict[str, Any], None]:
        """Get document by file path from both tbl_document and tbl_legal_document

        Args:
            file_path: The file path to search for

        Returns:
            Union[dict[str, Any], None]: Document data if found, None otherwise
        """
        # Try tbl_document first
        sql_doc = 'select * from "tbl_document" where "workspace_id"=$1 and "document_name"=$2'
        params_doc = {"workspace": to_uuid(self.workspace), "file_path": file_path}
        result_doc = await self.db.query(sql_doc, list(params_doc.values()), True)

        if result_doc and result_doc != []:
            created_at = self._format_datetime_with_timezone(result_doc[0]["create_time"])
            updated_at = self._format_datetime_with_timezone(result_doc[0]["update_time"])
            return dict(
                id=str(result_doc[0]["document_id"]),
                content_length=result_doc[0]["content_length"],
                content_summary=result_doc[0]["content_summary"],
                status=result_doc[0]["status"],
                created_at=created_at,
                updated_at=updated_at,
                file_path=result_doc[0]["document_name"],
                metadata={},
                error_msg=result_doc[0].get("error_msg"),
                content_hash=result_doc[0].get("content_hash"),
            )

        # Try tbl_legal_document second
        sql_legal = """
            SELECT 
                workspace_id,
                id,
                summary AS content_summary,
                length(COALESCE(summary, '')) AS content_length,
                COALESCE(process_status, 'EXTRACTED') AS status,
                title AS document_name,
                NULL::text AS content_hash,
                NULL::text AS error_msg,
                created_at AS create_time,
                updated_at AS update_time
            FROM "tbl_legal_document"
            WHERE "workspace_id"=$1 AND "title"=$2
        """
        params_legal = {"workspace": to_uuid(self.workspace), "title": file_path}
        result_legal = await self.db.query(sql_legal, list(params_legal.values()), True)

        if result_legal and result_legal != []:
            created_at = self._format_datetime_with_timezone(result_legal[0]["create_time"])
            updated_at = self._format_datetime_with_timezone(result_legal[0]["update_time"])
            return dict(
                id=str(result_legal[0]["id"]),
                content_length=result_legal[0]["content_length"],
                content_summary=result_legal[0]["content_summary"],
                status=result_legal[0]["status"],
                created_at=created_at,
                updated_at=updated_at,
                file_path=result_legal[0]["document_name"],
                metadata={},
                error_msg=result_legal[0].get("error_msg"),
                content_hash=result_legal[0].get("content_hash"),
            )

        return None

    async def get_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status from both tbl_document and tbl_legal_document"""
        sql = """
            SELECT status, SUM(count) as count FROM (
                SELECT status, COUNT(1) as count
                FROM "tbl_document"
                WHERE "workspace_id"=$1
                GROUP BY status
                UNION ALL
                SELECT 
                    COALESCE(process_status, 'EXTRACTED') AS status,
                    COUNT(1) as count
                FROM "tbl_legal_document"
                WHERE "workspace_id"=$1
                GROUP BY COALESCE(process_status, 'EXTRACTED')
            ) t
            GROUP BY status
        """
        params = {"workspace": to_uuid(self.workspace)}
        result = await self.db.query(sql, list(params.values()), True)
        counts = {}
        for doc in result:
            counts[doc["status"]] = int(doc["count"])
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        """all documents with a specific status from both tables"""
        sql = """
            SELECT 
                workspace_id,
                document_id::text AS document_id,
                content_summary,
                content_length,
                status,
                document_name,
                content_hash,
                error_msg,
                create_time,
                update_time
            FROM "tbl_document"
            WHERE "workspace_id"=$1 AND status=$2
            UNION ALL
            SELECT 
                workspace_id,
                id::text AS document_id,
                summary AS content_summary,
                length(COALESCE(summary, '')) AS content_length,
                COALESCE(process_status, 'EXTRACTED') AS status,
                title AS document_name,
                NULL::text AS content_hash,
                NULL::text AS error_msg,
                created_at AS create_time,
                updated_at AS update_time
            FROM "tbl_legal_document"
            WHERE "workspace_id"=$1 AND COALESCE(process_status, 'EXTRACTED') = $2
        """
        params = {"workspace": to_uuid(self.workspace), "status": status.value}
        result = await self.db.query(sql, list(params.values()), True)

        docs_by_status = {}
        for element in result:
            created_at = self._format_datetime_with_timezone(element["create_time"])
            updated_at = self._format_datetime_with_timezone(element["update_time"])

            docs_by_status[str(element["document_id"])] = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=created_at,
                updated_at=updated_at,
                file_path=element["document_name"],
                metadata={},
                error_msg=element.get("error_msg"),
                content_hash=element.get("content_hash"),
            )

        return docs_by_status

    async def get_docs_paginated(
        self,
        status_filter: DocStatus | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ) -> tuple[list[tuple[str, DocProcessingStatus]], int]:
        """Get documents with pagination support from both tbl_document and tbl_legal_document

        Args:
            status_filter: Filter by document status, None for all statuses
            page: Page number (1-based)
            page_size: Number of documents per page (10-200)
            sort_field: Field to sort by ('created_at', 'updated_at', 'id')
            sort_direction: Sort direction ('asc' or 'desc')

        Returns:
            Tuple of (list of (doc_id, DocProcessingStatus) tuples, total_count)
        """
        # Validate parameters
        if page < 1:
            page = 1
        if page_size < 10:
            page_size = 10
        elif page_size > 200:
            page_size = 200

        # Whitelist validation for sort_field to prevent SQL injection
        field_mapping = {
            "created_at": "create_time",
            "updated_at": "update_time",
            "id": "document_id",
            "file_path": "document_name",
        }
        db_sort_field = field_mapping.get(sort_field, "update_time")

        # Whitelist validation for sort_direction to prevent SQL injection
        if sort_direction.lower() not in ["asc", "desc"]:
            sort_direction = "desc"
        else:
            sort_direction = sort_direction.lower()

        # Calculate offset
        offset = (page - 1) * page_size

        # Build parameterized query components
        params = {"workspace": to_uuid(self.workspace)}
        param_count = 1

        # Build WHERE clauses with parameterized queries pushed down into each table
        if status_filter is not None:
            param_count += 1
            doc_where = 'WHERE "workspace_id"=$1 AND status=$2'
            legal_where = 'WHERE "workspace_id"=$1 AND COALESCE(process_status, \'EXTRACTED\')=$2'
            params["status"] = status_filter.value
        else:
            doc_where = 'WHERE "workspace_id"=$1'
            legal_where = 'WHERE "workspace_id"=$1'

        # Build ORDER BY clause using validated whitelist values
        order_clause = f"ORDER BY {db_sort_field} {sort_direction.upper()}"

        # Unified document CTE with pushed-down filters to leverage workspace index
        unified_sql_base = f"""
            WITH all_docs AS (
                SELECT 
                    workspace_id,
                    document_id::text AS document_id,
                    content_summary,
                    content_length,
                    status,
                    document_name,
                    content_hash,
                    error_msg,
                    create_time,
                    update_time
                FROM "tbl_document"
                {doc_where}
                UNION ALL
                SELECT 
                    workspace_id,
                    id::text AS document_id,
                    summary AS content_summary,
                    length(COALESCE(summary, '')) AS content_length,
                    COALESCE(process_status, 'EXTRACTED') AS status,
                    title AS document_name,
                    NULL::text AS content_hash,
                    NULL::text AS error_msg,
                    created_at AS create_time,
                    updated_at AS update_time
                FROM "tbl_legal_document"
                {legal_where}
            )
        """

        # Query for total count
        count_sql = f"""
            {unified_sql_base}
            SELECT COUNT(*) as total FROM all_docs
        """
        count_result = await self.db.query(count_sql, list(params.values()))
        total_count = count_result["total"] if count_result else 0

        # Query for paginated data with parameterized LIMIT and OFFSET
        data_sql = f"""
            {unified_sql_base}
            SELECT * FROM all_docs
            {order_clause}
            LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """
        params["limit"] = page_size
        params["offset"] = offset

        result = await self.db.query(data_sql, list(params.values()), True)

        # Convert to (doc_id, DocProcessingStatus) tuples
        documents = []
        for element in result:
            doc_id = str(element["document_id"])

            # Convert datetime objects to ISO format strings with timezone info
            created_at = self._format_datetime_with_timezone(element["create_time"])
            updated_at = self._format_datetime_with_timezone(element["update_time"])

            doc_status = DocProcessingStatus(
                content_summary=element["content_summary"],
                content_length=element["content_length"],
                status=element["status"],
                created_at=created_at,
                updated_at=updated_at,
                file_path=element["document_name"],
                metadata={},
                error_msg=element.get("error_msg"),
                content_hash=element.get("content_hash"),
            )
            documents.append((doc_id, doc_status))

        return documents, total_count

    async def get_all_status_counts(self) -> dict[str, int]:
        """Get counts of documents in each status for all documents across both tables

        Returns:
            Dictionary mapping status names to counts, including 'all' field
        """
        try:
            logger.debug(f"[{self.workspace}] Getting status counts...")
            sql = """
                SELECT status, SUM(count) as count FROM (
                    SELECT status, COUNT(*) as count
                    FROM "tbl_document"
                    WHERE "workspace_id"=$1
                    GROUP BY status
                    UNION ALL
                    SELECT 
                        COALESCE(process_status, 'EXTRACTED') AS status,
                        COUNT(*) as count
                    FROM "tbl_legal_document"
                    WHERE "workspace_id"=$1
                    GROUP BY COALESCE(process_status, 'EXTRACTED')
                ) t
                GROUP BY status
            """
            params = {"workspace": to_uuid(self.workspace)}

            # Add timeout to prevent hanging
            result = await asyncio.wait_for(
                self.db.query(sql, list(params.values()), True),
                timeout=10.0,  # 10 second timeout
            )

            counts = {}
            total_count = 0
            for row in result:
                counts[row["status"]] = int(row["count"])
                total_count += int(row["count"])

            # Add 'all' field with total count
            counts["all"] = total_count

            logger.debug(f"[{self.workspace}] Status counts retrieved: {counts}")
            return counts

        except asyncio.TimeoutError:
            logger.error(f"[{self.workspace}] Timeout getting status counts after 10s")
            # Return empty counts on timeout
            return {"all": 0}
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting status counts: {str(e)}")
            logger.error(traceback.format_exc())
            # Return empty counts on error
            return {"all": 0}

    async def get_status_counts_across_workspaces(self) -> dict[str, dict[str, int]]:
        """Get counts of documents in each status across all workspaces for both tables

        Returns:
            Dictionary mapping workspace IDs to status count dictionaries
        """
        try:
            logger.debug("Getting status counts across all workspaces...")
            sql = """
                SELECT workspace_id, status, SUM(count) as count FROM (
                    SELECT workspace_id, status, COUNT(*) as count
                    FROM "tbl_document"
                    GROUP BY workspace_id, status
                    UNION ALL
                    SELECT 
                        workspace_id,
                        COALESCE(process_status, 'EXTRACTED') AS status,
                        COUNT(*) as count
                    FROM "tbl_legal_document"
                    GROUP BY workspace_id, COALESCE(process_status, 'EXTRACTED')
                ) t
                GROUP BY workspace_id, status
            """

            # Add timeout to prevent hanging
            result = await asyncio.wait_for(
                self.db.query(sql, [], True),
                timeout=15.0,  # 15 second timeout for cross-workspace query
            )

            workspaces_counts = {}
            for row in result:
                ws = str(row["workspace_id"])
                status = row["status"]
                count = int(row["count"])

                if ws not in workspaces_counts:
                    workspaces_counts[ws] = {"all": 0}

                workspaces_counts[ws][status] = count
                workspaces_counts[ws]["all"] += count

            logger.debug(
                f"Status counts across workspaces retrieved: {len(workspaces_counts)} workspaces"
            )
            return workspaces_counts

        except asyncio.TimeoutError:
            logger.error("Timeout getting status counts across workspaces after 15s")
            # Return empty dict on timeout
            return {}
        except Exception as e:
            logger.error(f"Error getting status counts across workspaces: {str(e)}")
            logger.error(traceback.format_exc())
            # Return empty dict on error
            return {}

    async def get_processing_throughput(self) -> dict[str, float]:
        """Get recent processed-document throughput for the current workspace."""
        try:
            sql = """
                WITH completed_docs AS (
                    SELECT update_time AS completed_at
                    FROM "tbl_document"
                    WHERE workspace_id=$1 AND status='PROCESSED'
                    UNION ALL
                    SELECT updated_at AS completed_at
                    FROM "tbl_legal_document"
                    WHERE workspace_id=$1
                      AND COALESCE(process_status, 'EXTRACTED')='PROCESSED'
                )
                SELECT
                    COUNT(*) FILTER (
                        WHERE completed_at >= NOW() - INTERVAL '1 minute'
                    )::float AS docs_per_minute,
                    COUNT(*) FILTER (
                        WHERE completed_at >= NOW() - INTERVAL '1 hour'
                    )::float AS docs_per_hour,
                    COUNT(*) FILTER (
                        WHERE completed_at >= NOW() - INTERVAL '24 hours'
                    )::float AS total_24h
                FROM completed_docs
            """
            result = await asyncio.wait_for(
                self.db.query(sql, [to_uuid(self.workspace)]),
                timeout=10.0,
            )
            if not result:
                return {
                    "docs_per_minute": 0.0,
                    "docs_per_hour": 0.0,
                    "total_24h": 0.0,
                }

            return {
                "docs_per_minute": float(result.get("docs_per_minute") or 0.0),
                "docs_per_hour": float(result.get("docs_per_hour") or 0.0),
                "total_24h": float(result.get("total_24h") or 0.0),
            }
        except asyncio.TimeoutError:
            logger.error(
                f"[{self.workspace}] Timeout getting processing throughput after 10s"
            )
            return {
                "docs_per_minute": 0.0,
                "docs_per_hour": 0.0,
                "total_24h": 0.0,
            }
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error getting processing throughput: {e}"
            )
            return {
                "docs_per_minute": 0.0,
                "docs_per_hour": 0.0,
                "total_24h": 0.0,
            }

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    async def is_empty(self) -> bool:
        """Check if the storage is empty for the current workspace and namespace

        Returns:
            bool: True if storage is empty, False otherwise
        """
        table_name = namespace_to_table_name(self.namespace)
        if not table_name:
            logger.error(
                f"[{self.workspace}] Unknown namespace for is_empty check: {self.namespace}"
            )
            return True

        if table_name == "tbl_document":
            sql = """
                SELECT EXISTS(
                    SELECT 1 FROM "tbl_document" WHERE workspace_id=$1 LIMIT 1
                    UNION ALL
                    SELECT 1 FROM "tbl_legal_document" WHERE workspace_id=$1 LIMIT 1
                ) as has_data
            """
        else:
            sql = f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE workspace_id=$1 LIMIT 1) as has_data"

        try:
            result = await self.db.query(sql, [to_uuid(self.workspace)])
            return not result.get("has_data", False) if result else True
        except Exception as e:
            logger.error(f"[{self.workspace}] Error checking if storage is empty: {e}")
            return True

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs."""
        if not ids:
            return

        logger.info(
            f"[{self.workspace}] Deleting {len(ids)} documents from PGDocStatusStorage"
        )
        target_ids = [to_uuid(x) for x in ids]

        try:
            # Delete chunk associations first because Cascade on tbl_lightrag_document_chunk
            # does not automatically cascade to tbl_chunk_entity/tbl_chunk_relation
            await self.db.execute(
                """
                DELETE FROM tbl_chunk_entity 
                WHERE chunk_id IN (
                    SELECT id FROM tbl_lightrag_document_chunk WHERE document_id = ANY($1)
                )
                """,
                [target_ids],
            )
            await self.db.execute(
                """
                DELETE FROM tbl_chunk_relation 
                WHERE chunk_id IN (
                    SELECT id FROM tbl_lightrag_document_chunk WHERE document_id = ANY($1)
                )
                """,
                [target_ids],
            )

            # Now delete from tbl_document, cascading to chunks and cache
            await self.db.execute(
                "DELETE FROM tbl_document WHERE workspace_id = $1 AND document_id = ANY($2)",
                [to_uuid(self.workspace), target_ids],
            )
            logger.debug(
                f"[{self.workspace}] Successfully deleted {len(ids)} records from {self.namespace}"
            )
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting records from {self.namespace}: {e}",
                exc_info=True,
            )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Update or insert document status

        Args:
            data: dictionary of document IDs and their status data
        """
        logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        def parse_datetime(dt_str):
            """Parse datetime and ensure it's stored as UTC time in database"""
            if dt_str is None:
                return None
            if isinstance(dt_str, (datetime.date, datetime.datetime)):
                # If it's a datetime object
                if isinstance(dt_str, datetime.datetime):
                    # If no timezone info, assume it's UTC
                    if dt_str.tzinfo is None:
                        dt_str = dt_str.replace(tzinfo=timezone.utc)
                    # Convert to UTC and remove timezone info for storage
                    return dt_str.astimezone(timezone.utc).replace(tzinfo=None)
                return dt_str
            try:
                # Process ISO format string with timezone
                dt = datetime.datetime.fromisoformat(dt_str)
                # If no timezone info, assume it's UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                # Convert to UTC and remove timezone info for storage
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            except (ValueError, TypeError):
                logger.warning(
                    f"[{self.workspace}] Unable to parse datetime string: {dt_str}"
                )
                return None

        document_updates: dict[str, dict[str, Any]] = {}
        legal_updates: dict[int, dict[str, Any]] = {}
        for doc_id, update_data in data.items():
            legal_document_id = self._legal_document_id_from_key(doc_id)
            if legal_document_id is None:
                document_updates[doc_id] = update_data
            else:
                legal_updates[legal_document_id] = update_data

        for legal_document_id, update_data in legal_updates.items():
            updated_at = parse_datetime(update_data.get("updated_at"))
            if updated_at is None:
                updated_at = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
            status = update_data.get("status")
            if hasattr(status, "value"):
                status = status.value
            status = status or "UNKNOWN"
            await self.db.execute(
                """
                UPDATE "tbl_legal_document"
                SET process_status=$3,
                    updated_at=$4
                WHERE "workspace_id"=$1 AND "id"=$2
                """,
                [to_uuid(self.workspace), legal_document_id, str(status), updated_at],
            )

        data = document_updates
        if not data:
            return

        # Changed to UPDATE since C-RAG only updates existing rows created by KMS
        sql = """UPDATE "tbl_document" SET
                 "content_summary" = COALESCE($3, "content_summary"),
                 "content_length" = COALESCE($4, "content_length"),
                 status = $5,
                 "document_name" = COALESCE($6, "document_name"),
                 "content_hash" = COALESCE($7, "content_hash"),
                 "error_msg" = $8,
                 "create_time" = COALESCE($9, "create_time"),
                 "update_time" = $10
                 WHERE "workspace_id" = $1 AND "document_id" = $2"""

        # Merge partial updates with existing records to avoid KeyError
        existing_rows = await self.get_by_ids(list(data.keys()))
        existing_map = (
            {row["id"]: row for row in existing_rows if row} if existing_rows else {}
        )

        for k, update_data in data.items():
            v = {
                "content_summary": "",
                "content_length": 0,
                "status": "UNKNOWN",
                "file_path": "",
                "chunks_count": -1,
                "chunks_list": [],
                "content_hash": "",
                "metadata": {},
                "error_msg": "",
            }
            if k in existing_map:
                v.update(existing_map[k])
            v.update(update_data)

            # Remove timezone information, store utc time in db
            created_at = parse_datetime(v.get("created_at"))
            updated_at = parse_datetime(v.get("updated_at"))

            await self.db.execute(
                sql,
                [
                    self.workspace,
                    k,
                    v.get("content_summary", ""),
                    v.get("content_length", 0),
                    v.get("status", "UNKNOWN"),
                    v.get("file_path", ""),
                    v.get("content_hash"),
                    v.get("error_msg"),
                    created_at,
                    updated_at,
                ],
            )

    async def atomic_upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Atomically update or insert document status within a transaction.

        This method ensures atomic document record creation for concurrent uploads.
        It wraps the upsert operation in a database transaction to handle race conditions.

        Args:
            data: dictionary of document IDs and their status data
        """
        logger.debug(
            f"[{self.workspace}] Atomic insert of {len(data)} to {self.namespace}"
        )
        if not data:
            return

        def parse_datetime(dt_str):
            """Parse datetime and ensure it's stored as UTC time in database"""
            if dt_str is None:
                return None
            if isinstance(dt_str, (datetime.date, datetime.datetime)):
                if isinstance(dt_str, datetime.datetime):
                    if dt_str.tzinfo is None:
                        dt_str = dt_str.replace(tzinfo=timezone.utc)
                    return dt_str.astimezone(timezone.utc).replace(tzinfo=None)
                return dt_str
            try:
                dt = datetime.datetime.fromisoformat(dt_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            except (ValueError, TypeError):
                logger.warning(
                    f"[{self.workspace}] Unable to parse datetime string: {dt_str}"
                )
                return None

        # Changed to UPDATE since C-RAG only updates existing rows created by KMS
        sql = """UPDATE "tbl_document" SET
                 "content_summary" = COALESCE($3, "content_summary"),
                 "content_length" = COALESCE($4, "content_length"),
                 status = $5,
                 "document_name" = COALESCE($6, "document_name"),
                 "content_hash" = COALESCE($7, "content_hash"),
                 "error_msg" = $8,
                 "create_time" = COALESCE($9, "create_time"),
                 "update_time" = $10
                 WHERE "workspace_id" = $1 AND "document_id" = $2"""

        async def _atomic_operation(connection: asyncpg.Connection) -> None:
            """Execute all upserts within a single transaction"""
            async with connection.transaction():
                for k, v in data.items():
                    created_at = parse_datetime(v.get("created_at"))
                    updated_at = parse_datetime(v.get("updated_at"))

                    await connection.execute(
                        sql,
                        self.workspace,
                        k,
                        v.get("content_summary", ""),
                        v.get("content_length", 0),
                        v.get("status", "UNKNOWN"),
                        v.get("file_path", ""),
                        v.get("content_hash"),
                        v.get("error_msg"),
                        created_at,
                        updated_at,
                    )

        await self.db._run_with_retry(_atomic_operation)

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            table_name = namespace_to_table_name(self.namespace)
            if not table_name:
                return {
                    "status": "error",
                    "message": f"Unknown namespace: {self.namespace}",
                }

            drop_sql = SQL_TEMPLATES["drop_specifiy_table_workspace"].format(
                table_name=table_name
            )
            await self.db.execute(drop_sql, {"workspace": to_uuid(self.workspace)})
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def check_duplicate_by_content_hash(
        self, content_hash: str, exclude_doc_id: str | None = None
    ) -> tuple[bool, str | None]:
        """Check if a document with the same content_hash already exists in the workspace.

        This function queries documents by (workspace, content_hash) and excludes the current
        doc_id to avoid self-matching. It uses parameterized queries with LIMIT 1 for performance
        and SQL injection prevention.

        Args:
            content_hash: MD5 hash of document content (32 hex characters)
            exclude_doc_id: Optional doc_id to exclude from search (avoid self-match)

        Returns:
            tuple[bool, str | None]: (is_duplicate, existing_doc_id)
                - (True, doc_id) if duplicate found
                - (False, None) if no duplicate exists

        Validates Requirements: 4.1, 4.2, 4.4, 11.2, 11.3

        Examples:
            >>> is_dup, existing_id = await storage.check_duplicate_by_content_hash(
            ...     content_hash="a1b2c3d4e5f6...",
            ...     exclude_doc_id="doc-550e8400-e29b-41d4-a716-446655440000"
            ... )
            >>> if is_dup:
            ...     print(f"Duplicate of: {existing_id}")
        """
        try:
            # Build parameterized query based on whether we need to exclude a doc_id
            if exclude_doc_id is not None:
                sql = """
                    SELECT id, status 
                    FROM LIGHTRAG_DOC_STATUS 
                    WHERE workspace_id=$1 AND content_hash=$2 AND id != $3 
                    LIMIT 1
                """
                params = [self.workspace, content_hash, exclude_doc_id]
            else:
                sql = """
                    SELECT id, status 
                    FROM LIGHTRAG_DOC_STATUS 
                    WHERE workspace_id=$1 AND content_hash=$2 
                    LIMIT 1
                """
                params = [self.workspace, content_hash]

            # Execute query
            result = await self.db.query(sql, params, multirows=True)

            # Check result
            if result and len(result) > 0:
                existing_doc_id = result[0]["id"]
                return (True, existing_doc_id)
            else:
                return (False, None)

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error checking duplicate by content_hash: {content_hash}, error: {e}"
            )
            # On error, return False to allow processing to continue
            # This is safer than raising an exception which could block document processing
            return (False, None)


NAMESPACE_TABLE_MAP = {
    # NOTE: Dữ liệu được quản lý bởi KMS (Hibernate/JDBC)
    NameSpace.KV_STORE_TEXT_CHUNKS: "tbl_lightrag_document_chunk",
    NameSpace.KV_STORE_LLM_RESPONSE_CACHE: "tbl_lightrag_llm_cache",
    NameSpace.DOC_STATUS: "tbl_document",
    NameSpace.KV_STORE_FULL_DOCS: "tbl_document",
}


def namespace_to_table_name(namespace: str) -> str:
    for k, v in NAMESPACE_TABLE_MAP.items():
        if is_namespace(namespace, k):
            return v


TABLES = {
    # NOTE: DDL được quản lý bởi KMS với Java Hibernate ORM.
    # Python chỉ đóng vai trò Client, không tự tạo bảng.
    "tbl_document": {"ddl": None},
    "tbl_lightrag_document_chunk": {"ddl": None},
    "tbl_lightrag_llm_cache": {"ddl": None},
}


SQL_TEMPLATES = {
    # SQL for KVStorage
    "get_by_id_full_docs": """SELECT document_id AS id, COALESCE(content_summary, '') AS content,
                                COALESCE(document_name, '') AS file_path
                                FROM tbl_document WHERE workspace_id=$1 AND document_id=$2
                            """,
    "get_by_id_text_chunks": """SELECT c.id, c.tokens, COALESCE(c.content, '') AS content,
                                c.order_index AS chunk_order_index, 
                                COALESCE(c.document_id::text, c.legal_document_id::text) AS full_doc_id, 
                                c.legal_document_id,
                                ld.status::text AS legal_status,
                                CASE WHEN c.legal_document_id IS NOT NULL THEN 'legal' ELSE 'general' END AS doc_type,
                                COALESCE(d.origin_path, d.document_name, ld.title, '') AS file_path,
                                COALESCE((SELECT jsonb_agg(l.id) FROM tbl_lightrag_llm_cache l WHERE l.chunk_id = c.id), '[]'::jsonb) AS llm_cache_list,
                                EXTRACT(EPOCH FROM c.create_time)::BIGINT AS create_time,
                                EXTRACT(EPOCH FROM c.update_time)::BIGINT AS update_time
                                FROM tbl_lightrag_document_chunk c
                                LEFT JOIN tbl_document d ON c.document_id = d.document_id
                                LEFT JOIN tbl_legal_document ld ON c.legal_document_id = ld.id
                                WHERE (d.workspace_id=$1 OR ld.workspace_id=$1) AND c.id=$2
                            """,
    "get_by_id_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, query_param AS queryparam,
                                EXTRACT(EPOCH FROM create_time)::BIGINT AS create_time,
                                EXTRACT(EPOCH FROM update_time)::BIGINT AS update_time
                                FROM tbl_lightrag_llm_cache WHERE workspace_id=$1 AND id=$2
                               """,
    "get_by_id_entity_chunks": """SELECT e.name AS id, jsonb_agg(c.chunk_id) AS chunk_ids
                                  FROM tbl_lightrag_entity e
                                  JOIN tbl_chunk_entity c ON e.id = c.entity_id
                                  WHERE e.workspace_id=$1 AND e.name=$2
                                  GROUP BY e.name
                               """,
    "get_by_id_relation_chunks": """
        SELECT $2::text || '<SEP>' || $3::text AS id, jsonb_agg(cr.chunk_id) AS chunk_ids
        FROM tbl_lightrag_entity e1
        JOIN tbl_lightrag_entity e2 ON e2.workspace_id = $1 AND ((e1.name = $2 AND e2.name = $3) OR (e1.name = $3 AND e2.name = $2))
        JOIN tbl_lightrag_relation r ON r.workspace_id = $1 AND r.source_entity_id = e1.id AND r.target_entity_id = e2.id
        JOIN tbl_chunk_relation cr ON cr.relation_id = r.id
        WHERE e1.workspace_id = $1
    """,
    "get_by_ids_full_docs": """SELECT document_id AS id, COALESCE(content_summary, '') AS content,
                                 COALESCE(document_name, '') AS file_path
                                 FROM tbl_document WHERE workspace_id=$1 AND document_id = ANY($2)
                             """,
    "get_by_ids_text_chunks": """SELECT c.id, c.tokens, COALESCE(c.content, '') AS content,
                                 c.order_index AS chunk_order_index, 
                                 COALESCE(c.document_id::text, c.legal_document_id::text) AS full_doc_id, 
                                 c.legal_document_id,
                                 ld.status::text AS legal_status,
                                 CASE WHEN c.legal_document_id IS NOT NULL THEN 'legal' ELSE 'general' END AS doc_type,
                                 COALESCE(d.origin_path, d.document_name, ld.title, '') AS file_path,
                                 COALESCE((SELECT jsonb_agg(l.id) FROM tbl_lightrag_llm_cache l WHERE l.chunk_id = c.id), '[]'::jsonb) AS llm_cache_list,
                                 EXTRACT(EPOCH FROM c.create_time)::BIGINT AS create_time,
                                 EXTRACT(EPOCH FROM c.update_time)::BIGINT AS update_time
                                 FROM tbl_lightrag_document_chunk c
                                 LEFT JOIN tbl_document d ON c.document_id = d.document_id
                                 LEFT JOIN tbl_legal_document ld ON c.legal_document_id = ld.id
                                 WHERE (d.workspace_id=$1 OR ld.workspace_id=$1) AND c.id = ANY($2)
                             """,
    "get_by_ids_llm_response_cache": """SELECT id, original_prompt, return_value, chunk_id, cache_type, query_param AS queryparam,
                                 EXTRACT(EPOCH FROM create_time)::BIGINT AS create_time,
                                 EXTRACT(EPOCH FROM update_time)::BIGINT AS update_time
                                 FROM tbl_lightrag_llm_cache WHERE workspace_id=$1 AND id = ANY($2)
                                """,
    "get_by_ids_entity_chunks": """SELECT e.name AS id, jsonb_agg(c.chunk_id) AS chunk_ids
                                   FROM tbl_lightrag_entity e
                                   JOIN tbl_chunk_entity c ON e.id = c.entity_id
                                   WHERE e.workspace_id=$1 AND e.name = ANY($2)
                                   GROUP BY e.name
                                """,
    "upsert_text_chunk": """INSERT INTO tbl_lightrag_document_chunk (document_id, id, tokens, order_index, content, create_time, update_time)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (id) DO UPDATE SET
                            tokens = EXCLUDED.tokens,
                            order_index = EXCLUDED.order_index,
                            content = EXCLUDED.content,
                            update_time = EXCLUDED.update_time
                         """,
    "upsert_doc_full": """UPDATE tbl_document SET
                          content_summary = $2,
                          document_name = $3
                          WHERE document_id = $1 AND workspace_id = $4
                       """,
    "upsert_llm_response_cache": """INSERT INTO tbl_lightrag_llm_cache (workspace_id, id, original_prompt, return_value, chunk_id, cache_type, query_param, create_time, update_time)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                                    ON CONFLICT (id) DO UPDATE SET
                                    return_value = EXCLUDED.return_value,
                                    query_param = EXCLUDED.query_param,
                                    update_time = EXCLUDED.update_time
                                 """,
    # DROP tables
    "drop_specifiy_table_workspace": """
        DELETE FROM {table_name} WHERE workspace_id=$1
       """,
}
