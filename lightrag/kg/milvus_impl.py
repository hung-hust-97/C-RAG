import asyncio
import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, final, Optional, Dict, Callable, ClassVar, Iterator
from dataclasses import dataclass, fields
import numpy as np
from lightrag.utils import logger, compute_mdhash_id
from ..base import BaseVectorStorage
from ..constants import DEFAULT_MAX_FILE_PATH_LENGTH
from ..kg.shared_storage import get_data_init_lock
import pipmaster as pm

if not pm.is_installed("pymilvus"):
    pm.install("pymilvus>=2.6.2")

import configparser
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema  # type: ignore
from packaging import version

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


# Supported index types
SUPPORTED_INDEX_TYPES = {
    "AUTOINDEX",
    "HNSW",
    "HNSW_SQ",
    "HNSW_PQ",
    "HNSW_PRQ",
    "IVF_FLAT",
    "IVF_SQ8",
    "IVF_PQ",
    "DISKANN",
    "SCANN",
    "IVF_RABITQ",
}

# Supported metric types
SUPPORTED_METRIC_TYPES = {"COSINE", "L2", "IP"}

# HNSW_SQ quantization types
SUPPORTED_SQ_TYPES = {"SQ4U", "SQ6", "SQ8", "BF16", "FP16"}
SUPPORTED_REFINE_TYPES = {"SQ6", "SQ8", "BF16", "FP16", "FP32"}

# Index type version requirements
# Important: HNSW_SQ was first introduced in Milvus 2.6.8 (not 2.5)
INDEX_VERSION_REQUIREMENTS = {
    "HNSW_SQ": "2.6.8",  # HNSW_SQ requires Milvus 2.6.8+ (supports sq_types such as SQ4U, SQ6, SQ8, BF16, FP16)
}


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Parse environment variable as boolean"""
    val = os.environ.get(key, "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    elif val in ("false", "0", "no", "off"):
        return False
    return default


def _get_env_int(key: str, default: int) -> int:
    """Parse environment variable as integer"""
    val = os.environ.get(key, "")
    if val:
        try:
            return int(val)
        except ValueError:
            logger.warning(
                f"Invalid integer value for {key}: {val}, using default {default}"
            )
    return default


def _get_env_float(key: str, default: float) -> float:
    """Parse environment variable as float"""
    val = os.environ.get(key, "")
    if val:
        try:
            return float(val)
        except ValueError:
            logger.warning(
                f"Invalid float value for {key}: {val}, using default {default}"
            )
    return default


def _get_env_optional_float(key: str) -> Optional[float]:
    """Parse an optional float environment variable."""
    val = os.environ.get(key, "")
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        logger.warning(f"Invalid float value for {key}: {val}, ignoring it")
        return None


def _get_env_optional_int(key: str) -> Optional[int]:
    """Parse an optional integer environment variable."""
    val = os.environ.get(key, "")
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        logger.warning(f"Invalid integer value for {key}: {val}, ignoring it")
        return None


@dataclass
class MilvusIndexConfig:
    """
    Milvus vector index configuration class

    Supports configuration via environment variables or initialization parameters.
    Initialization parameters take precedence over environment variables.
    """

    # Base configuration
    index_type: Optional[str] = None
    metric_type: Optional[str] = None

    # HNSW series parameters
    hnsw_m: Optional[int] = None
    hnsw_ef_construction: Optional[int] = None
    hnsw_ef: Optional[int] = None

    # HNSW_SQ specific parameters
    sq_type: Optional[str] = None
    sq_refine: Optional[bool] = None
    sq_refine_type: Optional[str] = None
    sq_refine_k: Optional[int] = None

    # IVF series parameters
    ivf_nlist: Optional[int] = None
    ivf_nprobe: Optional[int] = None

    def __post_init__(self):
        """Load configuration from environment variables (init parameters take precedence)"""
        # Index type
        self.index_type = (
            self.index_type or os.environ.get("MILVUS_INDEX_TYPE", "AUTOINDEX")
        ).upper()

        # Metric type
        self.metric_type = (
            self.metric_type or os.environ.get("MILVUS_METRIC_TYPE", "COSINE")
        ).upper()

        # HNSW parameters
        # Defaults aligned with Milvus 2.4+ official documentation
        if self.hnsw_m is None:
            self.hnsw_m = _get_env_int("MILVUS_HNSW_M", 16)
        if self.hnsw_ef_construction is None:
            self.hnsw_ef_construction = _get_env_int("MILVUS_HNSW_EF_CONSTRUCTION", 360)
        if self.hnsw_ef is None:
            self.hnsw_ef = _get_env_int("MILVUS_HNSW_EF", 200)

        # HNSW_SQ parameters
        if self.sq_type is None:
            self.sq_type = os.environ.get("MILVUS_HNSW_SQ_TYPE", "SQ8").upper()
        if self.sq_refine is None:
            self.sq_refine = _get_env_bool("MILVUS_HNSW_SQ_REFINE", False)
        if self.sq_refine_type is None:
            self.sq_refine_type = os.environ.get(
                "MILVUS_HNSW_SQ_REFINE_TYPE", "FP32"
            ).upper()
        if self.sq_refine_k is None:
            self.sq_refine_k = _get_env_int("MILVUS_HNSW_SQ_REFINE_K", 10)

        # IVF parameters
        if self.ivf_nlist is None:
            self.ivf_nlist = _get_env_int("MILVUS_IVF_NLIST", 1024)
        if self.ivf_nprobe is None:
            self.ivf_nprobe = _get_env_int("MILVUS_IVF_NPROBE", 16)

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate configuration validity"""
        if self.index_type not in SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"Unsupported index type: {self.index_type}. "
                f"Supported: {SUPPORTED_INDEX_TYPES}"
            )

        if self.metric_type not in SUPPORTED_METRIC_TYPES:
            raise ValueError(
                f"Unsupported metric type: {self.metric_type}. "
                f"Supported: {SUPPORTED_METRIC_TYPES}"
            )

        if self.index_type == "HNSW_SQ":
            if self.sq_type not in SUPPORTED_SQ_TYPES:
                raise ValueError(
                    f"Unsupported sq_type: {self.sq_type}. "
                    f"Supported: {SUPPORTED_SQ_TYPES}"
                )
            if self.sq_refine and self.sq_refine_type not in SUPPORTED_REFINE_TYPES:
                raise ValueError(
                    f"Unsupported refine_type: {self.sq_refine_type}. "
                    f"Supported: {SUPPORTED_REFINE_TYPES}"
                )

        # Parameter range validation
        if not (2 <= self.hnsw_m <= 2048):
            raise ValueError(f"hnsw_m must be in [2, 2048], got {self.hnsw_m}")
        if self.hnsw_ef_construction < 1:
            raise ValueError(
                f"hnsw_ef_construction must be >= 1, got {self.hnsw_ef_construction}"
            )
        if self.ivf_nlist < 1 or self.ivf_nlist > 65536:
            raise ValueError(f"ivf_nlist must be in [1, 65536], got {self.ivf_nlist}")

    def validate_milvus_version(self, server_version: str) -> None:
        """
        Validate Milvus server version supports the configured index type

        Args:
            server_version: Milvus server version string (e.g., "2.6.9")

        Raises:
            ValueError: Version does not meet index type requirements
        """
        current_ver = version.parse(
            server_version.split("-")[0]
        )  # Handle "2.6.9-dev" format

        # Check HNSW_SQ index type version requirements (requires 2.6.8+)
        if self.index_type == "HNSW_SQ":
            required = INDEX_VERSION_REQUIREMENTS["HNSW_SQ"]
            if current_ver < version.parse(required):
                raise ValueError(
                    f"HNSW_SQ requires Milvus {required}+, "
                    f"current version: {server_version}"
                )

        logger.info(
            f"Milvus version {server_version} validated for index type "
            f"{self.index_type}"
            + (f" with sq_type {self.sq_type}" if self.index_type == "HNSW_SQ" else "")
        )

    def build_index_params(self, index_params, field_name: str = "vector"):
        """
        Build pymilvus index parameters

        Args:
            index_params: IndexParams instance (from compatibility helper or client.prepare_index_params())
            field_name: Vector field name

        Returns:
            IndexParams object, or a dict fallback when direct API creation is needed.
        """
        if index_params is None:
            if self.index_type == "AUTOINDEX":
                logger.info(
                    "Using AUTOINDEX with direct API fallback because IndexParams is unavailable"
                )
                params = {}
                if _get_env_bool("MILVUS_ENABLE_MMAP_FOR_LARGE_COLLECTIONS", False):
                    params["mmap.enabled"] = True
                return {
                    "field_name": field_name,
                    "index_type": self.index_type,
                    "metric_type": self.metric_type,
                    "params": params,
                }
            raise RuntimeError(
                f"IndexParams not available but required for index type "
                f"'{self.index_type}'. Ensure pymilvus is installed correctly."
            )

        params: Dict[str, Any] = {}

        # HNSW series indexes
        if self.index_type in ("HNSW", "HNSW_SQ", "HNSW_PQ", "HNSW_PRQ"):
            params["M"] = self.hnsw_m
            params["efConstruction"] = self.hnsw_ef_construction

            # HNSW_SQ specific parameters
            if self.index_type == "HNSW_SQ":
                params["sq_type"] = self.sq_type
                if self.sq_refine:
                    params["refine"] = True
                    params["refine_type"] = self.sq_refine_type

        # IVF series indexes
        elif self.index_type in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
            params["nlist"] = self.ivf_nlist

        elif self.index_type == "DISKANN":
            diskann_options = {
                "MaxDegree": _get_env_optional_int("MILVUS_DISKANN_MAX_DEGREE"),
                "SearchListSize": _get_env_optional_int(
                    "MILVUS_DISKANN_SEARCH_LIST_SIZE"
                ),
                "PQCodeBugetGBRatio": _get_env_optional_float(
                    "MILVUS_DISKANN_PQ_CODE_BUDGET_GB_RATIO"
                ),
                "SearchCacheBudgetGBRatio": _get_env_optional_float(
                    "MILVUS_DISKANN_SEARCH_CACHE_BUDGET_GB_RATIO"
                ),
                "BeamWidthRatio": _get_env_optional_float(
                    "MILVUS_DISKANN_BEAM_WIDTH_RATIO"
                ),
            }
            params.update(
                {
                    key: value
                    for key, value in diskann_options.items()
                    if value is not None
                }
            )

        # SCANN has no additional params

        if _get_env_bool("MILVUS_ENABLE_MMAP_FOR_LARGE_COLLECTIONS", False):
            params["mmap.enabled"] = True

        index_params.add_index(
            field_name=field_name,
            index_type=self.index_type,
            metric_type=self.metric_type,
            params=params,
        )

        logger.info(
            f"Milvus index configured: type={self.index_type}, "
            f"metric={self.metric_type}, params={params}"
        )

        return index_params

    def build_search_params(self) -> Dict[str, Any]:
        """
        Build search parameters

        Returns:
            Search parameters dictionary
        """
        search_params: Dict[str, Any] = {}

        if self.index_type in ("HNSW", "HNSW_SQ", "HNSW_PQ", "HNSW_PRQ"):
            search_params["ef"] = self.hnsw_ef
            if self.index_type == "HNSW_SQ" and self.sq_refine:
                search_params["refine_k"] = self.sq_refine_k

        elif self.index_type in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
            search_params["nprobe"] = self.ivf_nprobe

        return {"params": search_params} if search_params else {}

    @classmethod
    def get_config_field_names(cls) -> set:
        """Get all configuration field names from the dataclass.

        This method provides a single source of truth for configuration parameter names,
        eliminating the need to maintain duplicate hardcoded lists elsewhere.

        Returns:
            Set of field names that can be used to extract configuration from kwargs
        """
        return {f.name for f in fields(cls)}

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary (for logging/debugging)"""
        return {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef": self.hnsw_ef,
            "sq_type": self.sq_type if self.index_type == "HNSW_SQ" else None,
            "sq_refine": self.sq_refine if self.index_type == "HNSW_SQ" else None,
            "sq_refine_type": (
                self.sq_refine_type
                if self.index_type == "HNSW_SQ" and self.sq_refine
                else None
            ),
            "sq_refine_k": (
                self.sq_refine_k
                if self.index_type == "HNSW_SQ" and self.sq_refine
                else None
            ),
            "ivf_nlist": (
                self.ivf_nlist if self.index_type.startswith("IVF") else None
            ),
            "ivf_nprobe": (
                self.ivf_nprobe if self.index_type.startswith("IVF") else None
            ),
        }


@dataclass
class _MilvusCollectionLoadState:
    collection_name: str
    client: MilvusClient
    loaded: bool
    last_used_at: float
    in_flight: int = 0


@final
@dataclass
class MilvusVectorDBStorage(BaseVectorStorage):
    _load_state_lock: ClassVar[threading.RLock] = threading.RLock()
    _load_states: ClassVar[dict[str, _MilvusCollectionLoadState]] = {}

    def _get_milvus_connection_kwargs(self, include_db_name: bool = True) -> dict:
        """Build Milvus connection kwargs from env/config."""
        connection_kwargs = {
            "uri": os.environ.get(
                "MILVUS_URI",
                config.get(
                    "milvus",
                    "uri",
                    fallback=os.path.join(
                        self.global_config["working_dir"], "milvus_lite.db"
                    ),
                ),
            ),
            "user": os.environ.get(
                "MILVUS_USER", config.get("milvus", "user", fallback=None)
            ),
            "password": os.environ.get(
                "MILVUS_PASSWORD",
                config.get("milvus", "password", fallback=None),
            ),
            "token": os.environ.get(
                "MILVUS_TOKEN", config.get("milvus", "token", fallback=None)
            ),
        }

        db_name = os.environ.get(
            "MILVUS_DB_NAME",
            config.get("milvus", "db_name", fallback=None),
        )
        if include_db_name and db_name:
            connection_kwargs["db_name"] = db_name

        return connection_kwargs

    def _get_milvus_db_name(self) -> Optional[str]:
        """Return the configured Milvus database name, if any."""
        db_name = self._get_milvus_connection_kwargs(include_db_name=True).get(
            "db_name"
        )
        if db_name is None:
            return None

        normalized_name = str(db_name).strip()
        return normalized_name or None

    def _connect_bulk_insert_alias(self, alias: str) -> None:
        """Connect ORM utility APIs to the same database used by MilvusClient."""
        from pymilvus import connections, db

        connection_kwargs = self._get_milvus_connection_kwargs(include_db_name=True)
        connections.connect(alias, **connection_kwargs)

        db_name = self._get_milvus_db_name()
        if db_name:
            db.using_database(db_name, using=alias)

    def _do_bulk_insert(self, collection_name: str, files: list[str], using: str) -> int:
        """Run Milvus bulk import while preserving the configured database name."""
        from pymilvus import utility

        db_name = self._get_milvus_db_name()
        if not db_name or db_name == "default":
            return utility.do_bulk_insert(
                collection_name=collection_name,
                files=files,
                using=using,
            )

        # PyMilvus 2.5.x utility.do_bulk_insert() does not set ImportRequest.db_name,
        # so Milvus handles the import in the default database even when the
        # connection alias is bound to a non-default database.
        from pymilvus.client.grpc_handler import (  # type: ignore
            ErrorCode,
            MilvusException,
            check_status,
            common_pb2,
        )
        from pymilvus.client.prepare import Prepare  # type: ignore
        from pymilvus.orm.utility import _get_connection  # type: ignore

        connection = _get_connection(using)
        request = Prepare.do_bulk_insert(collection_name, None, files)
        request.db_name = db_name
        response = connection._stub.Import(request)
        check_status(response.status)
        if len(response.tasks) == 0:
            raise MilvusException(
                ErrorCode.UNEXPECTED_ERROR,
                "no task id returned from server",
                common_pb2.UnexpectedError,
            )
        return response.tasks[0]

    async def _cleanup_expired_bulk_insert_objects(
        self,
        s3_client: Any,
        bucket: str,
        *,
        current_key: Optional[str] = None,
    ) -> None:
        """Delete old Milvus bulk insert objects from S3/MinIO."""
        import time

        ttl_seconds = _get_env_int("MILVUS_BULK_INSERT_TTL_SECONDS", 86400)
        if ttl_seconds <= 0:
            return

        cleanup_interval = _get_env_int(
            "MILVUS_BULK_INSERT_CLEANUP_INTERVAL_SECONDS", 3600
        )
        now = time.time()
        if now - self._last_bulk_insert_cleanup_at < cleanup_interval:
            return

        self._last_bulk_insert_cleanup_at = now
        prefix = "bulk_insert/"
        expires_before = now - ttl_seconds
        delete_batch: list[dict[str, str]] = []
        deleted_count = 0

        async def flush_delete_batch() -> None:
            nonlocal deleted_count
            if not delete_batch:
                return

            batch = list(delete_batch)
            delete_batch.clear()
            await s3_client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": batch, "Quiet": True},
            )
            deleted_count += len(batch)

        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key")
                    if not key or key == current_key:
                        continue

                    last_modified = obj.get("LastModified")
                    if last_modified is None:
                        continue

                    try:
                        modified_ts = last_modified.timestamp()
                    except AttributeError:
                        continue

                    if modified_ts >= expires_before:
                        continue

                    delete_batch.append({"Key": key})
                    if len(delete_batch) >= 1000:
                        await flush_delete_batch()

            await flush_delete_batch()
            if deleted_count:
                logger.info(
                    f"[{self.workspace}] Deleted {deleted_count} expired Milvus "
                    f"bulk insert files from {bucket}/{prefix}"
                )
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] Failed to cleanup expired Milvus bulk insert files: {e}"
            )

    def _sanitize_workspace_for_collection(self, workspace: str) -> str:
        """Sanitize workspace name for use as Milvus collection name prefix.
        
        Replaces special characters with underscores to ensure the name is valid
        for Milvus collection naming conventions. Adds "ws_" prefix if the 
        sanitized name starts with a digit (Milvus requires first char to be 
        letter or underscore). Returns "base" if workspace is empty.
        """
        workspace = workspace.strip()
        if not workspace:
            return "base"
        
        # Replace non-alphanumeric characters (except underscore) with underscore
        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", workspace).strip("_")
        if not sanitized:
            # If sanitization results in empty string, use "base"
            return "base"
        
        # Ensure first character is letter or underscore (Milvus requirement)
        if sanitized[0].isdigit():
            sanitized = f"ws_{sanitized}"
        
        return sanitized

    def _create_milvus_client(self) -> MilvusClient:
        """Create a Milvus client and ensure the configured database exists."""
        client = MilvusClient(
            **self._get_milvus_connection_kwargs(include_db_name=False)
        )
        db_name = self._get_milvus_db_name()

        if not db_name:
            return client

        existing_databases = set(client.list_databases())
        if db_name not in existing_databases:
            logger.warning(
                f"[{self.workspace}] Milvus database '{db_name}' not found, creating it"
            )
            client.create_database(db_name)

        use_database = getattr(client, "use_database", None) or getattr(
            client, "using_database", None
        )
        if callable(use_database):
            use_database(db_name)
            logger.debug(
                f"[{self.workspace}] Using Milvus database '{db_name}' for namespace '{self.namespace}'"
            )
            return client

        return MilvusClient(**self._get_milvus_connection_kwargs(include_db_name=True))

    @staticmethod
    def _is_closed_channel_error(error: Exception) -> bool:
        """Return True if the error indicates the Milvus gRPC channel is closed."""
        message = str(error).lower()
        return (
            "closed channel" in message
            or "cannot invoke rpc on closed channel" in message
        )

    def _reconnect_client(self, reason: str = "closed channel") -> None:
        """Recreate Milvus client after a recoverable connection failure."""
        logger.warning(
            f"[{self.workspace}] Reconnecting Milvus client for {self.namespace} (reason: {reason})"
        )

        if self._client is not None:
            try:
                close = getattr(self._client, "close", None)
                if callable(close):
                    close()
            except Exception as close_error:
                logger.debug(
                    f"[{self.workspace}] Error while closing stale Milvus client: {close_error}"
                )

        self._client = self._create_milvus_client()
        self._collection_loaded = False

    def _milvus_call(
        self,
        operation: str,
        action: Callable[[MilvusClient], Any],
        max_retries: int = 3,
    ) -> Any:
        """Execute a Milvus operation with retry on closed-channel failures."""
        import time

        if self._client is None:
            self._client = self._create_milvus_client()

        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                return action(self._client)
            except Exception as error:
                last_error = error

                if not self._is_closed_channel_error(error):
                    raise

                if attempt == max_retries - 1:
                    logger.error(
                        f"[{self.workspace}] {operation} failed after {max_retries} attempts: {error}"
                    )
                    raise

                backoff_seconds = 0.1 * (2**attempt)
                logger.warning(
                    f"[{self.workspace}] Closed channel during {operation} "
                    f"(attempt {attempt + 1}/{max_retries}). "
                    f"Reconnecting in {backoff_seconds:.1f}s"
                )

                time.sleep(backoff_seconds)
                self._reconnect_client(
                    f"{operation} closed channel (attempt {attempt + 1})"
                )

        if last_error is not None:
            raise last_error

        raise RuntimeError(f"[{self.workspace}] {operation} failed unexpectedly")

    def _create_schema_for_namespace(self) -> CollectionSchema:
        """Create schema based on the current instance's namespace"""

        # Get vector dimension from embedding_func
        dimension = self.embedding_func.embedding_dim

        # Base fields (common to all collections)
        base_fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True
            ),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(
                name="workspace_id",
                dtype=DataType.VARCHAR,
                max_length=256,
                nullable=True,
            ),
        ]

        # Determine specific fields based on namespace
        if self.namespace.endswith("entities"):
            specific_fields = [
                FieldSchema(
                    name="entity_name",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                    nullable=True,
                ),
                FieldSchema(
                    name="file_path",
                    dtype=DataType.VARCHAR,
                    max_length=DEFAULT_MAX_FILE_PATH_LENGTH,
                    nullable=True,
                ),
            ]
            description = "LightRAG entities vector storage"

        elif self.namespace.endswith("relationships"):
            specific_fields = [
                FieldSchema(
                    name="src_id", dtype=DataType.VARCHAR, max_length=512, nullable=True
                ),
                FieldSchema(
                    name="tgt_id", dtype=DataType.VARCHAR, max_length=512, nullable=True
                ),
                FieldSchema(
                    name="file_path",
                    dtype=DataType.VARCHAR,
                    max_length=DEFAULT_MAX_FILE_PATH_LENGTH,
                    nullable=True,
                ),
            ]
            description = "LightRAG relationships vector storage"

        elif self.namespace.endswith("chunks"):
            specific_fields = [
                FieldSchema(
                    name="full_doc_id",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                    nullable=True,
                ),
                FieldSchema(
                    name="file_path",
                    dtype=DataType.VARCHAR,
                    max_length=DEFAULT_MAX_FILE_PATH_LENGTH,
                    nullable=True,
                ),
            ]
            description = "LightRAG chunks vector storage"

        else:
            # Default generic schema (backward compatibility)
            specific_fields = [
                FieldSchema(
                    name="file_path",
                    dtype=DataType.VARCHAR,
                    max_length=DEFAULT_MAX_FILE_PATH_LENGTH,
                    nullable=True,
                ),
            ]
            description = "LightRAG generic vector storage"

        # Merge all fields
        all_fields = base_fields + specific_fields

        return CollectionSchema(
            fields=all_fields,
            description=description,
            enable_dynamic_field=True,  # Support dynamic fields
        )

    def _mmap_enabled_for_namespace(self) -> bool:
        """Return whether mmap should be enabled for this collection namespace."""
        if not _get_env_bool("MILVUS_ENABLE_MMAP_FOR_LARGE_COLLECTIONS", False):
            return False

        configured = os.environ.get(
            "MILVUS_MMAP_NAMESPACES", "entities,relationships"
        )
        namespace_suffixes = {
            item.strip().lower()
            for item in configured.split(",")
            if item.strip()
        }
        if "*" in namespace_suffixes:
            return True

        namespace = self.namespace.lower()
        return any(namespace.endswith(suffix) for suffix in namespace_suffixes)

    def _create_collection_with_properties(
        self, collection_name: str, schema: CollectionSchema
    ) -> None:
        """Create a collection and apply resource-saving properties when supported."""
        create_kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "schema": schema,
        }

        if self._mmap_enabled_for_namespace():
            create_kwargs["properties"] = {"mmap.enabled": "true"}

        try:
            self._client.create_collection(**create_kwargs)
        except TypeError:
            if "properties" not in create_kwargs:
                raise

            logger.warning(
                f"[{self.workspace}] Milvus client does not support create_collection "
                f"properties; creating {collection_name} without mmap properties"
            )
            create_kwargs.pop("properties", None)
            self._client.create_collection(**create_kwargs)

    def _get_index_params(self):
        """Get IndexParams in a version-compatible way"""
        try:
            # Try to use client's prepare_index_params method (most common)
            if hasattr(self._client, "prepare_index_params"):
                return self._client.prepare_index_params()
        except Exception:
            pass

        try:
            # Try to import IndexParams from different possible locations
            from pymilvus.client.prepare import IndexParams  # type: ignore

            return IndexParams()
        except ImportError:
            pass

        try:
            from pymilvus.client.types import IndexParams  # type: ignore

            return IndexParams()
        except ImportError:
            pass

        try:
            from pymilvus import IndexParams  # type: ignore

            return IndexParams()
        except ImportError:
            pass

        # If all else fails, return None to use fallback method
        return None

    def _create_scalar_index_fallback(self, field_name: str, index_type: str):
        """Fallback method to create scalar index using direct API"""
        # Skip unsupported index types
        if index_type == "SORTED":
            logger.info(
                f"[{self.workspace}] Skipping SORTED index for {field_name} (not supported in this Milvus version)"
            )
            return

        try:
            self._client.create_index(
                collection_name=self.final_namespace,
                field_name=field_name,
                index_params={"index_type": index_type},
            )
            logger.debug(
                f"[{self.workspace}] Created {field_name} index using fallback method"
            )
        except Exception as e:
            logger.info(
                f"[{self.workspace}] Could not create {field_name} index using fallback method: {e}"
            )

    def _create_indexes_after_collection(self):
        """Create indexes after collection is created"""
        # Build vector index using index configuration
        # Use compatibility helper to get IndexParams
        index_params_for_vector = self._get_index_params()

        vector_index_params = self.index_config.build_index_params(
            index_params_for_vector, field_name="vector"
        )

        # Re-raise exceptions to surface vector index creation failures
        if isinstance(vector_index_params, dict):
            self._client.create_index(
                collection_name=self.final_namespace,
                field_name=vector_index_params["field_name"],
                index_params={
                    "index_type": vector_index_params["index_type"],
                    "metric_type": vector_index_params["metric_type"],
                    "params": vector_index_params["params"],
                },
            )
        else:
            self._client.create_index(
                collection_name=self.final_namespace,
                index_params=vector_index_params,
            )

        logger.debug(
            f"[{self.workspace}] Created vector index with config: {self.index_config.to_dict()}"
        )

        # Create scalar indexes based on namespace
        # Wrap scalar index creation in try-except to allow graceful degradation
        try:
            # Try to get IndexParams in a version-compatible way
            scalar_index_params = self._get_index_params()

            if scalar_index_params is not None:
                # Create scalar indexes based on namespace
                if self.namespace.endswith("entities"):
                    # Create indexes for entity fields
                    try:
                        entity_name_index = self._get_index_params()
                        entity_name_index.add_index(
                            field_name="entity_name", index_type="INVERTED"
                        )
                        self._client.create_index(
                            collection_name=self.final_namespace,
                            index_params=entity_name_index,
                        )
                    except Exception as e:
                        logger.debug(
                            f"[{self.workspace}] IndexParams method failed for entity_name: {e}"
                        )
                        self._create_scalar_index_fallback("entity_name", "INVERTED")

                elif self.namespace.endswith("relationships"):
                    # Create indexes for relationship fields
                    try:
                        src_id_index = self._get_index_params()
                        src_id_index.add_index(
                            field_name="src_id", index_type="INVERTED"
                        )
                        self._client.create_index(
                            collection_name=self.final_namespace,
                            index_params=src_id_index,
                        )
                    except Exception as e:
                        logger.debug(
                            f"[{self.workspace}] IndexParams method failed for src_id: {e}"
                        )
                        self._create_scalar_index_fallback("src_id", "INVERTED")

                    try:
                        tgt_id_index = self._get_index_params()
                        tgt_id_index.add_index(
                            field_name="tgt_id", index_type="INVERTED"
                        )
                        self._client.create_index(
                            collection_name=self.final_namespace,
                            index_params=tgt_id_index,
                        )
                    except Exception as e:
                        logger.debug(
                            f"[{self.workspace}] IndexParams method failed for tgt_id: {e}"
                        )
                        self._create_scalar_index_fallback("tgt_id", "INVERTED")

                elif self.namespace.endswith("chunks"):
                    # Create indexes for chunk fields
                    try:
                        doc_id_index = self._get_index_params()
                        doc_id_index.add_index(
                            field_name="full_doc_id", index_type="INVERTED"
                        )
                        self._client.create_index(
                            collection_name=self.final_namespace,
                            index_params=doc_id_index,
                        )
                    except Exception as e:
                        logger.debug(
                            f"[{self.workspace}] IndexParams method failed for full_doc_id: {e}"
                        )
                        self._create_scalar_index_fallback("full_doc_id", "INVERTED")

                # Common index: workspace_id for multi-tenant filtering
                try:
                    ws_index = self._get_index_params()
                    ws_index.add_index(
                        field_name="workspace_id", index_type="INVERTED"
                    )
                    self._client.create_index(
                        collection_name=self.final_namespace,
                        index_params=ws_index,
                    )
                except Exception as e:
                    logger.debug(
                        f"[{self.workspace}] IndexParams method failed for workspace_id: {e}"
                    )
                    self._create_scalar_index_fallback("workspace_id", "INVERTED")

            else:
                # Fallback to direct API calls if IndexParams is not available
                logger.info(
                    f"[{self.workspace}] IndexParams not available, using fallback methods for {self.namespace}"
                )

                # Create scalar indexes using fallback
                if self.namespace.endswith("entities"):
                    self._create_scalar_index_fallback("entity_name", "INVERTED")
                elif self.namespace.endswith("relationships"):
                    self._create_scalar_index_fallback("src_id", "INVERTED")
                    self._create_scalar_index_fallback("tgt_id", "INVERTED")
                elif self.namespace.endswith("chunks"):
                    self._create_scalar_index_fallback("full_doc_id", "INVERTED")

                # Common fallback index: workspace_id
                self._create_scalar_index_fallback("workspace_id", "INVERTED")

            logger.info(
                f"[{self.workspace}] Created indexes for collection: {self.namespace}"
            )

        except Exception as e:
            # Scalar index failures are logged as warnings (not critical)
            logger.warning(
                f"[{self.workspace}] Failed to create some scalar indexes for {self.namespace}: {e}"
            )

    def _get_required_fields_for_namespace(self) -> dict:
        """Get required core field definitions for current namespace"""

        # Base fields (common to all types)
        base_fields = {
            "id": {"type": "VarChar", "is_primary": True},
            "vector": {"type": "FloatVector"},
            "created_at": {"type": "Int64"},
            "workspace_id": {"type": "VarChar"},
        }

        # Add specific fields based on namespace
        if self.namespace.endswith("entities"):
            specific_fields = {
                "entity_name": {"type": "VarChar"},
                "file_path": {"type": "VarChar"},
            }
        elif self.namespace.endswith("relationships"):
            specific_fields = {
                "src_id": {"type": "VarChar"},
                "tgt_id": {"type": "VarChar"},
                "file_path": {"type": "VarChar"},
            }
        elif self.namespace.endswith("chunks"):
            specific_fields = {
                "full_doc_id": {"type": "VarChar"},
                "file_path": {"type": "VarChar"},
            }
        else:
            specific_fields = {
                "file_path": {"type": "VarChar"},
            }

        return {**base_fields, **specific_fields}

    def _is_field_compatible(self, existing_field: dict, expected_config: dict) -> bool:
        """Check compatibility of a single field"""
        field_name = existing_field.get("name", "unknown")
        existing_type = existing_field.get("type")
        expected_type = expected_config.get("type")

        logger.debug(
            f"[{self.workspace}] Checking field '{field_name}': existing_type={existing_type} (type={type(existing_type)}), expected_type={expected_type}"
        )

        # Convert DataType enum values to string names if needed
        original_existing_type = existing_type
        if hasattr(existing_type, "name"):
            existing_type = existing_type.name
            logger.debug(
                f"[{self.workspace}] Converted enum to name: {original_existing_type} -> {existing_type}"
            )
        elif isinstance(existing_type, int):
            # Map common Milvus internal type codes to type names for backward compatibility
            type_mapping = {
                21: "VarChar",
                101: "FloatVector",
                5: "Int64",
                9: "Double",
            }
            mapped_type = type_mapping.get(existing_type, str(existing_type))
            logger.debug(
                f"[{self.workspace}] Mapped numeric type: {existing_type} -> {mapped_type}"
            )
            existing_type = mapped_type

        # Normalize type names for comparison
        type_aliases = {
            "VARCHAR": "VarChar",
            "String": "VarChar",
            "FLOAT_VECTOR": "FloatVector",
            "INT64": "Int64",
            "BigInt": "Int64",
            "DOUBLE": "Double",
            "Float": "Double",
        }

        original_existing = existing_type
        original_expected = expected_type
        existing_type = type_aliases.get(existing_type, existing_type)
        expected_type = type_aliases.get(expected_type, expected_type)

        if original_existing != existing_type or original_expected != expected_type:
            logger.debug(
                f"[{self.workspace}] Applied aliases: {original_existing} -> {existing_type}, {original_expected} -> {expected_type}"
            )

        # Basic type compatibility check
        type_compatible = existing_type == expected_type
        logger.debug(
            f"[{self.workspace}] Type compatibility for '{field_name}': {existing_type} == {expected_type} -> {type_compatible}"
        )

        if not type_compatible:
            logger.warning(
                f"[{self.workspace}] Type mismatch for field '{field_name}': expected {expected_type}, got {existing_type}"
            )
            return False

        # Primary key check - be more flexible about primary key detection
        if expected_config.get("is_primary"):
            # Check multiple possible field names for primary key status
            is_primary = (
                existing_field.get("is_primary_key", False)
                or existing_field.get("is_primary", False)
                or existing_field.get("primary_key", False)
            )
            logger.debug(
                f"[{self.workspace}] Primary key check for '{field_name}': expected=True, actual={is_primary}"
            )
            logger.debug(
                f"[{self.workspace}] Raw field data for '{field_name}': {existing_field}"
            )

            # For ID field, be more lenient - if it's the ID field, assume it should be primary
            if field_name == "id" and not is_primary:
                logger.info(
                    f"[{self.workspace}] ID field '{field_name}' not marked as primary in existing collection, but treating as compatible"
                )
                # Don't fail for ID field primary key mismatch
            elif not is_primary:
                logger.warning(
                    f"[{self.workspace}] Primary key mismatch for field '{field_name}': expected primary key, but field is not primary"
                )
                return False

        logger.debug(f"[{self.workspace}] Field '{field_name}' is compatible")
        return True

    def _check_vector_dimension(self, collection_info: dict):
        """Check vector dimension compatibility"""
        current_dimension = self.embedding_func.embedding_dim

        # Find vector field dimension
        for field in collection_info.get("fields", []):
            if field.get("name") == "vector":
                field_type = field.get("type")

                # Extract type name from DataType enum or string
                type_name = None
                if hasattr(field_type, "name"):
                    type_name = field_type.name
                elif isinstance(field_type, str):
                    type_name = field_type
                else:
                    type_name = str(field_type)

                # Check if it's a vector type (supports multiple formats)
                if type_name in ["FloatVector", "FLOAT_VECTOR"]:
                    existing_dimension = field.get("params", {}).get("dim")

                    # Convert both to int for comparison to handle type mismatches
                    # (Milvus API may return string "1024" vs int 1024)
                    try:
                        existing_dim_int = (
                            int(existing_dimension)
                            if existing_dimension is not None
                            else None
                        )
                        current_dim_int = (
                            int(current_dimension)
                            if current_dimension is not None
                            else None
                        )
                    except (TypeError, ValueError) as e:
                        logger.error(
                            f"[{self.workspace}] Failed to parse dimensions: existing={existing_dimension} (type={type(existing_dimension)}), "
                            f"current={current_dimension} (type={type(current_dimension)}), error={e}"
                        )
                        raise ValueError(
                            f"Invalid dimension values for collection '{self.final_namespace}': "
                            f"existing={existing_dimension}, current={current_dimension}"
                        ) from e

                    if existing_dim_int != current_dim_int:
                        raise ValueError(
                            f"Vector dimension mismatch for collection '{self.final_namespace}': "
                            f"existing={existing_dim_int}, current={current_dim_int}"
                        )

                    logger.debug(
                        f"[{self.workspace}] Vector dimension check passed: {current_dim_int}"
                    )
                    return

        # If no vector field found, this might be an old collection created with simple schema
        logger.warning(
            f"[{self.workspace}] Vector field not found in collection '{self.namespace}'. This might be an old collection created with simple schema."
        )
        logger.warning(
            f"[{self.workspace}] Consider recreating the collection for optimal performance."
        )
        return

    def _check_file_path_length_restriction(self, collection_info: dict) -> bool:
        """Check if collection has file_path length restrictions that need migration

        Returns:
            bool: True if migration is needed, False otherwise
        """
        existing_fields = {
            field["name"]: field for field in collection_info.get("fields", [])
        }

        # Check if file_path field exists and has length restrictions
        if "file_path" in existing_fields:
            file_path_field = existing_fields["file_path"]
            # Get max_length from field params
            max_length = file_path_field.get("params", {}).get("max_length")

            if max_length and max_length < DEFAULT_MAX_FILE_PATH_LENGTH:
                logger.info(
                    f"[{self.workspace}] Collection {self.namespace} has file_path max_length={max_length}, "
                    f"needs migration to {DEFAULT_MAX_FILE_PATH_LENGTH}"
                )
                return True

        return False

    def _check_schema_compatibility(self, collection_info: dict):
        """Check schema field compatibility and detect migration needs"""
        existing_fields = {
            field["name"]: field for field in collection_info.get("fields", [])
        }

        # Check if this is an old collection created with simple schema
        has_vector_field = any(
            field.get("name") == "vector" for field in collection_info.get("fields", [])
        )

        if not has_vector_field:
            logger.warning(
                f"[{self.workspace}] Collection {self.namespace} appears to be created with old simple schema (no vector field)"
            )
            logger.warning(
                f"[{self.workspace}] This collection will work but may have suboptimal performance"
            )
            logger.warning(
                f"[{self.workspace}] Consider recreating the collection for optimal performance"
            )
            return

        # Check if migration is needed for file_path length restrictions
        if self._check_file_path_length_restriction(collection_info):
            logger.info(
                f"[{self.workspace}] Starting automatic migration for collection {self.namespace}"
            )
            self._migrate_collection_schema()
            return

        # For collections with vector field, check basic compatibility
        # Only check for critical incompatibilities, not missing optional fields
        critical_fields = {"id": {"type": "VarChar", "is_primary": True}}

        incompatible_fields = []

        for field_name, expected_config in critical_fields.items():
            if field_name in existing_fields:
                existing_field = existing_fields[field_name]
                if not self._is_field_compatible(existing_field, expected_config):
                    incompatible_fields.append(
                        f"{field_name}: expected {expected_config['type']}, "
                        f"got {existing_field.get('type')}"
                    )

        if incompatible_fields:
            raise ValueError(
                f"Critical schema incompatibility in collection '{self.final_namespace}': {incompatible_fields}"
            )

        # Get all expected fields for informational purposes
        expected_fields = self._get_required_fields_for_namespace()
        missing_fields = [
            field for field in expected_fields if field not in existing_fields
        ]

        if missing_fields:
            logger.info(
                f"[{self.workspace}] Collection {self.namespace} missing optional fields: {missing_fields}"
            )
            logger.info(
                "These fields would be available in a newly created collection for better performance"
            )

        logger.debug(
            f"[{self.workspace}] Schema compatibility check passed for {self.namespace}"
        )

    def _migrate_collection_schema(self):
        """Migrate collection schema using query_iterator - completely solves query window limitations"""
        original_collection_name = self.final_namespace
        temp_collection_name = f"{self.final_namespace}_temp"
        iterator = None

        try:
            logger.info(
                f"[{self.workspace}] Starting iterator-based schema migration for {self.namespace}"
            )

            # Step 1: Create temporary collection with new schema
            logger.info(
                f"[{self.workspace}] Step 1: Creating temporary collection: {temp_collection_name}"
            )
            # Temporarily update final_namespace for index creation
            self.final_namespace = temp_collection_name
            new_schema = self._create_schema_for_namespace()
            self._create_collection_with_properties(temp_collection_name, new_schema)
            try:
                self._create_indexes_after_collection()
            except Exception as index_error:
                logger.warning(
                    f"[{self.workspace}] Failed to create indexes for new collection: {index_error}"
                )
                # Continue with migration even if index creation fails

            # Load the new collection
            self._client.load_collection(temp_collection_name)

            # Step 2: Copy data using query_iterator (solves query window limitation)
            logger.info(
                f"[{self.workspace}] Step 2: Copying data using query_iterator from: {original_collection_name}"
            )

            # Create query iterator
            try:
                iterator = self._client.query_iterator(
                    collection_name=original_collection_name,
                    batch_size=2000,  # Adjustable batch size for optimal performance
                    output_fields=["*"],  # Get all fields
                )
                logger.debug(f"[{self.workspace}] Query iterator created successfully")
            except Exception as iterator_error:
                logger.error(
                    f"[{self.workspace}] Failed to create query iterator: {iterator_error}"
                )
                raise

            # Iterate through all data
            total_migrated = 0
            batch_number = 1

            while True:
                try:
                    batch_data = iterator.next()
                    if not batch_data:
                        # No more data available
                        break

                    # Insert batch data to new collection
                    try:
                        self._client.insert(
                            collection_name=temp_collection_name, data=batch_data
                        )
                        total_migrated += len(batch_data)

                        logger.info(
                            f"[{self.workspace}] Iterator batch {batch_number}: "
                            f"processed {len(batch_data)} records, total migrated: {total_migrated}"
                        )
                        batch_number += 1

                    except Exception as batch_error:
                        logger.error(
                            f"[{self.workspace}] Failed to insert iterator batch {batch_number}: {batch_error}"
                        )
                        raise

                except Exception as next_error:
                    logger.error(
                        f"[{self.workspace}] Iterator next() failed at batch {batch_number}: {next_error}"
                    )
                    raise

            if total_migrated > 0:
                logger.info(
                    f"[{self.workspace}] Successfully migrated {total_migrated} records using iterator"
                )
            else:
                logger.info(
                    f"[{self.workspace}] No data found in original collection, migration completed"
                )

            # Step 3: Rename origin collection (keep for safety)
            logger.info(
                f"[{self.workspace}] Step 3: Rename origin collection to {original_collection_name}_old"
            )
            try:
                self._client.rename_collection(
                    original_collection_name, f"{original_collection_name}_old"
                )
            except Exception as rename_error:
                try:
                    logger.warning(
                        f"[{self.workspace}] Try to drop origin collection instead"
                    )
                    self._client.drop_collection(original_collection_name)
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Rename operation failed: {rename_error}"
                    )
                    raise e

            # Step 4: Rename temporary collection to original name
            logger.info(
                f"[{self.workspace}] Step 4: Renaming collection {temp_collection_name} -> {original_collection_name}"
            )
            try:
                self._client.rename_collection(
                    temp_collection_name, original_collection_name
                )
                logger.info(f"[{self.workspace}] Rename operation completed")
            except Exception as rename_error:
                logger.error(
                    f"[{self.workspace}] Rename operation failed: {rename_error}"
                )
                raise RuntimeError(
                    f"Failed to rename collection: {rename_error}"
                ) from rename_error

            # Restore final_namespace
            self.final_namespace = original_collection_name

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Iterator-based migration failed for {self.namespace}: {e}"
            )

            # Attempt cleanup of temporary collection if it exists
            try:
                if self._client and self._client.has_collection(temp_collection_name):
                    logger.info(
                        f"[{self.workspace}] Cleaning up failed migration temporary collection"
                    )
                    self._client.drop_collection(temp_collection_name)
            except Exception as cleanup_error:
                logger.warning(
                    f"[{self.workspace}] Failed to cleanup temporary collection: {cleanup_error}"
                )

            # Re-raise the original error
            raise RuntimeError(
                f"Iterator-based migration failed for collection {self.namespace}: {e}"
            ) from e

        finally:
            # Ensure iterator is properly closed
            if iterator:
                try:
                    iterator.close()
                    logger.debug(
                        f"[{self.workspace}] Query iterator closed successfully"
                    )
                except Exception as close_error:
                    logger.warning(
                        f"[{self.workspace}] Failed to close query iterator: {close_error}"
                    )

    def _validate_collection_compatibility(self):
        """Validate existing collection's dimension and schema compatibility"""
        try:
            collection_info = self._client.describe_collection(self.final_namespace)

            # 1. Check vector dimension
            self._check_vector_dimension(collection_info)

            # 2. Check schema compatibility
            self._check_schema_compatibility(collection_info)

            logger.info(
                f"[{self.workspace}] VectorDB Collection '{self.namespace}' compatibility validation passed"
            )

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Collection compatibility validation failed for {self.namespace}: {e}"
            )
            raise

    @staticmethod
    def _is_missing_vector_index_error(error: Exception) -> bool:
        """Return True when the error indicates the collection lacks a vector index."""
        error_message = str(error).lower()
        return (
            "no vector index" in error_message
            or "please create index firstly" in error_message
        )

    def _repair_missing_vector_index(self):
        """Create indexes for an existing collection that is missing its vector index."""
        logger.warning(
            f"[{self.workspace}] Collection '{self.namespace}' is missing a vector index, attempting repair"
        )
        self._create_indexes_after_collection()

    @staticmethod
    def _release_idle_collections_enabled() -> bool:
        return _get_env_bool("MILVUS_RELEASE_IDLE_COLLECTIONS", False)

    def _collection_load_key(self) -> str:
        connection_kwargs = self._get_milvus_connection_kwargs(include_db_name=True)
        uri = connection_kwargs.get("uri", "")
        db_name = connection_kwargs.get("db_name") or "default"
        return f"{uri}|{db_name}|{self.final_namespace}"

    def _release_state_locked(
        self,
        key: str,
        state: _MilvusCollectionLoadState,
        reason: str,
    ) -> bool:
        """Release a loaded collection. Caller must hold _load_state_lock."""
        if not state.loaded:
            return False
        if state.in_flight > 0:
            return False

        try:
            state.client.release_collection(state.collection_name)
            state.loaded = False
            state.last_used_at = time.monotonic()
            if key == self._collection_load_key():
                self._collection_loaded = False
            logger.info(
                f"[{self.workspace}] Released Milvus collection "
                f"{state.collection_name} ({reason})"
            )
            return True
        except Exception as release_error:
            logger.warning(
                f"[{self.workspace}] Failed to release Milvus collection "
                f"{state.collection_name} ({reason}): {release_error}"
            )
            return False

    def _release_idle_collections(self, preserve_key: Optional[str] = None) -> None:
        """Release idle loaded collections according to TTL and max-loaded limits."""
        if not self._release_idle_collections_enabled():
            return

        now = time.monotonic()
        ttl_seconds = max(0, _get_env_int("MILVUS_COLLECTION_LOAD_TTL_SECONDS", 900))
        max_loaded = _get_env_int("MILVUS_MAX_LOADED_COLLECTIONS", 3)

        with self._load_state_lock:
            for key, state in list(self._load_states.items()):
                if key == preserve_key or not state.loaded or state.in_flight > 0:
                    continue
                if ttl_seconds > 0 and now - state.last_used_at >= ttl_seconds:
                    self._release_state_locked(key, state, "idle ttl")

            if max_loaded <= 0:
                return

            loaded_states = [
                (key, state)
                for key, state in self._load_states.items()
                if state.loaded
            ]
            loaded_count = len(loaded_states)
            if loaded_count <= max_loaded:
                return

            for key, state in sorted(loaded_states, key=lambda item: item[1].last_used_at):
                if loaded_count <= max_loaded:
                    break
                if key == preserve_key or state.in_flight > 0:
                    continue
                if self._release_state_locked(key, state, "loaded collection limit"):
                    loaded_count -= 1

    def _register_collection_loaded(self) -> None:
        if not self._release_idle_collections_enabled() or self._client is None:
            return

        key = self._collection_load_key()
        now = time.monotonic()
        with self._load_state_lock:
            state = self._load_states.get(key)
            if state is None:
                state = _MilvusCollectionLoadState(
                    collection_name=self.final_namespace,
                    client=self._client,
                    loaded=True,
                    last_used_at=now,
                )
                self._load_states[key] = state
            else:
                state.collection_name = self.final_namespace
                state.client = self._client
                state.loaded = True
                state.last_used_at = now

    def _begin_collection_use(self) -> Optional[str]:
        if not self._release_idle_collections_enabled():
            return None

        key = self._collection_load_key()
        now = time.monotonic()
        with self._load_state_lock:
            state = self._load_states.get(key)
            if state is None:
                if self._client is None:
                    return None
                state = _MilvusCollectionLoadState(
                    collection_name=self.final_namespace,
                    client=self._client,
                    loaded=True,
                    last_used_at=now,
                )
                self._load_states[key] = state

            state.in_flight += 1
            state.last_used_at = now
        return key

    def _end_collection_use(self, key: Optional[str]) -> None:
        if key is None or not self._release_idle_collections_enabled():
            return

        with self._load_state_lock:
            state = self._load_states.get(key)
            if state is None:
                return
            state.in_flight = max(0, state.in_flight - 1)
            state.last_used_at = time.monotonic()

    @contextmanager
    def _loaded_collection_context(self) -> Iterator[None]:
        self._ensure_collection_loaded()
        key = self._begin_collection_use()
        try:
            yield
        finally:
            self._end_collection_use(key)

    def _release_collection_on_finalize(self, client: MilvusClient) -> None:
        if not self._release_idle_collections_enabled():
            return

        key = self._collection_load_key()
        with self._load_state_lock:
            state = self._load_states.get(key)
            if state is None or state.client is not client or state.in_flight > 0:
                return
            self._release_state_locked(key, state, "storage finalize")

    def _mark_collection_dropped(self) -> None:
        key = self._collection_load_key()
        with self._load_state_lock:
            self._load_states.pop(key, None)
        self._collection_loaded = False

    def _ensure_collection_loaded(self):
        """Ensure the collection is loaded into memory for search operations"""
        release_idle_enabled = self._release_idle_collections_enabled()
        load_key = self._collection_load_key() if release_idle_enabled else None

        if release_idle_enabled:
            self._release_idle_collections(preserve_key=load_key)
            with self._load_state_lock:
                state = self._load_states.get(load_key)
                if state is not None and state.loaded:
                    state.last_used_at = time.monotonic()
                    self._collection_loaded = True
                    return
        elif self._collection_loaded:
            return

        try:
            if not self._milvus_call(
                "has_collection",
                lambda client: client.has_collection(self.final_namespace),
            ):
                logger.error(
                    f"[{self.workspace}] Collection {self.namespace} does not exist"
                )
                raise ValueError(f"Collection {self.final_namespace} does not exist")

            # Verify index existence before loading
            indexes = self._milvus_call(
                "list_indexes",
                lambda client: client.list_indexes(self.final_namespace),
            )
            if not indexes:
                logger.warning(
                    f"[{self.workspace}] Indexes not found during query load for {self.final_namespace}, creating them now"
                )
                self._create_indexes_after_collection()

            self._milvus_call(
                "load_collection",
                lambda client: client.load_collection(self.final_namespace),
            )
            self._collection_loaded = True
            self._register_collection_loaded()
            if load_key is not None:
                self._release_idle_collections(preserve_key=load_key)
            return

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Failed to load collection {self.namespace}: {e}"
            )
            raise

    async def drop(self):
        """Drop collection - Disabled by user constraint"""
        logger.warning(f"[{self.workspace}] Drop operation is disabled by user constraint. Ignoring drop request for {self.final_namespace}.")
        return
    def _create_collection_if_not_exist(self):
        """Create collection if not exists - Disabled by user constraint"""
        logger.info(f"[{self.workspace}] Collection creation is disabled by user constraint. Skipping _create_collection_if_not_exist.")
        return

    def __post_init__(self):
        self._validate_embedding_func()

        # Extract MilvusIndexConfig parameters from vector_db_storage_cls_kwargs
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        index_config_keys = MilvusIndexConfig.get_config_field_names()
        index_config_params = {
            k: v for k, v in kwargs.items() if k in index_config_keys
        }

        # Initialize index configuration (if not already set)
        if not hasattr(self, "index_config") or self.index_config is None:
            self.index_config = MilvusIndexConfig(**index_config_params)

        # Ensure self.workspace is a string (dataclasses might pass None)
        if self.workspace is None:
            self.workspace = ""

        # Check for MILVUS_WORKSPACE environment variable first (higher priority)
        milvus_workspace_env = os.environ.get("MILVUS_WORKSPACE")
        if milvus_workspace_env and milvus_workspace_env.strip():
            effective_workspace = milvus_workspace_env.strip()
            logger.info(
                f"[{self.workspace}] Using MILVUS_WORKSPACE environment variable: '{effective_workspace}' (overriding initialized workspace '{self.workspace}')"
            )
        else:
            effective_workspace = self.workspace

        # Determine effective workspace
        if effective_workspace and str(effective_workspace).strip():
            self.workspace = self._sanitize_workspace_for_collection(effective_workspace)
        else:
            self.workspace = ""

        # Do not prefix workspace_id for global partitioned schema
        self.final_namespace = self.namespace
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Ensure created_at and workspace_id are in meta_fields
        if "created_at" not in self.meta_fields:
            self.meta_fields.add("created_at")
        if "workspace_id" not in self.meta_fields:
            self.meta_fields.add("workspace_id")

        # Initialize client as None - will be created in initialize() method
        self._client = None
        self._collection_loaded = False
        self._last_bulk_insert_cleanup_at = 0.0
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._initialized = False

    async def initialize(self):
        """Initialize Milvus collection"""
        async with get_data_init_lock():
            if self._initialized:
                return

            try:
                # Create MilvusClient if not already created
                if self._client is None:
                    self._client = self._create_milvus_client()
                    logger.debug(
                        f"[{self.workspace}] MilvusClient created successfully"
                    )

                # Validate Milvus version compatibility with configured index
                if self.index_config.index_type in INDEX_VERSION_REQUIREMENTS:
                    try:
                        server_version = self._client.get_server_version()
                        self.index_config.validate_milvus_version(server_version)
                    except Exception as version_error:
                        logger.error(
                            f"[{self.workspace}] Milvus version validation failed: {version_error}"
                        )
                        raise

                # Create collection and check compatibility
                self._create_collection_if_not_exist()
                self._initialized = True
                logger.info(
                    f"[{self.workspace}] Milvus collection '{self.namespace}' initialized successfully"
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to initialize Milvus collection '{self.namespace}': {e}"
                )
                raise

    async def finalize(self):
        """Close the Milvus client and release its gRPC channel."""
        client = self._client
        self._client = None
        self._collection_loaded = False
        self._initialized = False

        if client is None:
            return

        self._release_collection_on_finalize(client)

        close = getattr(client, "close", None)
        if not callable(close):
            return

        try:
            result = close()
            if asyncio.iscoroutine(result):
                await result
            logger.debug(
                f"[{self.workspace}] MilvusClient for '{self.namespace}' closed"
            )
        except Exception as close_error:
            logger.warning(
                f"[{self.workspace}] Failed to close MilvusClient for "
                f"'{self.namespace}': {close_error}"
            )

    async def _upsert_grpc(self, data: dict[str, dict[str, Any]]) -> Any:
        """Standard gRPC-based row-by-row upsert into Milvus."""
        if not data:
            return None

        import time
        current_time = int(time.time())
        import json

        list_data: list[dict[str, Any]] = []
        for k, v in data.items():
            base_dict = {
                "id": k,
                "created_at": current_time,
                "workspace_id": self.workspace,
            }
            meta_dict = {k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields or k1 in ["entity_id", "chunk_id"]}
            
            # Enforce dynamic field size limit (~65536 bytes)
            # 1. Truncate specific known fields first
            for field in ["entity_name", "src_id", "tgt_id"]:
                if field in meta_dict and isinstance(meta_dict[field], str):
                    encoded = meta_dict[field].encode("utf-8")
                    if len(encoded) > 512:
                        meta_dict[field] = encoded[:512].decode("utf-8", errors="ignore")
            
            MAX_DYN_SIZE = 60000
            
            # 2. Iterate to reduce size if necessary
            for attempt in range(5):
                try:
                    # Estimate dynamic field size by json dumping meta_dict
                    encoded_size = len(json.dumps(meta_dict, ensure_ascii=False).encode("utf-8"))
                    if encoded_size <= MAX_DYN_SIZE:
                        break
                except Exception:
                    pass
                
                # Find largest string field
                longest_key = None
                max_len = -1
                for mk, mv in meta_dict.items():
                    if isinstance(mv, str):
                        val_len = len(mv.encode("utf-8"))
                        if val_len > max_len:
                            max_len = val_len
                            longest_key = mk
                
                if longest_key is None or max_len < 500:
                    # If strings are small, the size is likely caused by lists/dicts. Stringify and truncate them.
                    for mk, mv in meta_dict.items():
                        if isinstance(mv, (list, dict)):
                            meta_dict[mk] = json.dumps(mv, ensure_ascii=False)[:2000]
                    continue
                
                # Truncate the largest string
                excess = encoded_size - MAX_DYN_SIZE
                target_len = max(100, max_len - excess - 500)
                meta_dict[longest_key] = meta_dict[longest_key].encode("utf-8")[:target_len].decode("utf-8", errors="ignore")
            
            base_dict.update(meta_dict)
            list_data.append(base_dict)
        import re
        
        # Helper to strip HTML tags for cleaner embeddings
        def clean_for_embedding(text: str) -> str:
            if not isinstance(text, str):
                return text
            # Replace HTML tags with spaces to avoid joining words
            cleaned = re.sub(r'<[^>]+>', ' ', text)
            # Remove extra spaces
            return re.sub(r'\s+', ' ', cleaned).strip()

        contents = [v["content"] for v in data.values()]
        # Create cleaned contents for embedding generation only
        embedding_contents = [clean_for_embedding(c) for c in contents]
        
        batches = [
            embedding_contents[i : i + self._max_batch_size]
            for i in range(0, len(embedding_contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        # Route to proper global collection
        col_groups: dict[str, list[dict[str, Any]]] = {}
        for i, d in enumerate(list_data):
            d["vector"] = embeddings[i]
            
            is_legal = False
            if "doc_type" in d and d["doc_type"] == "legal":
                is_legal = True
            elif d.get("legal_status") is not None or d.get("legal_document_id") is not None:
                is_legal = True
            elif os.environ.get("PIPELINE_DOC_TYPE") == "legal":
                is_legal = True

            if "workspace_id" not in d or not d["workspace_id"]:
                d["workspace_id"] = self.workspace

            if self.namespace.endswith("chunks"):
                col_name = "legal_chunks" if is_legal else "general_chunks"
                if "doc_id" in d:
                    del d["doc_id"]
                d["chunk_id"] = d.get("id")
                
                if "status" not in d or d["status"] is None:
                    d["status"] = d.get("legal_status_code") or "UNKNOWN"
                if "effective_from" not in d or d["effective_from"] is None:
                    d["effective_from"] = 0
                if "effective_to" not in d or d["effective_to"] is None:
                    d["effective_to"] = 0
            elif self.namespace.endswith("entities"):
                col_name = "legal_entities" if is_legal else "general_entities"
                d["entity_id"] = d.get("id")
                d["entity_name"] = d.get("entity_name") or d.get("id")
            else:
                continue

            col_groups.setdefault(col_name, []).append(d)

        upsert_batch_size = max(1, _get_env_int("MILVUS_UPSERT_BATCH_SIZE", 256))
        results = []
        for col_name, col_data in col_groups.items():
            for i in range(0, len(col_data), upsert_batch_size):
                batch_data = col_data[i : i + upsert_batch_size]
                
                result = await asyncio.to_thread(
                    self._milvus_call,
                    "upsert",
                    lambda client, batch_data=batch_data, c=col_name: client.upsert(
                        collection_name=c, data=batch_data
                    ),
                )
                results.append(result)
        return results[-1] if len(results) == 1 else results

    async def _bulk_upsert_impl(self, data: dict[str, dict[str, Any]]) -> Any:
        """Internal Bulk Insert implementation (uploads JSON to MinIO and registers import task)."""
        import time
        import os
        import json
        import aioboto3
        import uuid
        import re

        current_time = int(time.time())

        list_data: list[dict[str, Any]] = []
        for k, v in data.items():
            base_dict = {
                "id": k,
                "created_at": current_time,
                "workspace_id": self.workspace,
            }
            
            meta_dict = {k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields}
            
            for field in ["entity_name", "src_id", "tgt_id"]:
                if field in meta_dict and isinstance(meta_dict[field], str):
                    encoded = meta_dict[field].encode("utf-8")
                    if len(encoded) > 512:
                        meta_dict[field] = encoded[:512].decode("utf-8", errors="ignore")
            
            MAX_DYN_SIZE = 60000
            
            for attempt in range(5):
                try:
                    encoded_size = len(json.dumps(meta_dict, ensure_ascii=False).encode("utf-8"))
                    if encoded_size <= MAX_DYN_SIZE:
                        break
                except Exception:
                    pass
                
                longest_key = None
                max_len = -1
                for mk, mv in meta_dict.items():
                    if isinstance(mv, str):
                        val_len = len(mv.encode("utf-8"))
                        if val_len > max_len:
                            max_len = val_len
                            longest_key = mk
                
                if longest_key is None or max_len < 500:
                    for mk, mv in meta_dict.items():
                        if isinstance(mv, (list, dict)):
                            meta_dict[mk] = json.dumps(mv, ensure_ascii=False)[:2000]
                    continue
                
                excess = encoded_size - MAX_DYN_SIZE
                target_len = max(100, max_len - excess - 500)
                meta_dict[longest_key] = meta_dict[longest_key].encode("utf-8")[:target_len].decode("utf-8", errors="ignore")
            
            base_dict.update(meta_dict)
            list_data.append(base_dict)

        def clean_for_embedding(text: str) -> str:
            if not isinstance(text, str):
                return text
            cleaned = re.sub(r"<[^>]+>", " ", text)
            return re.sub(r"\s+", " ", cleaned).strip()

        contents = [v["content"] for v in data.values()]
        embedding_contents = [clean_for_embedding(c) for c in contents]

        batches = [
            embedding_contents[i : i + self._max_batch_size]
            for i in range(0, len(embedding_contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)

        # Route to proper global collection
        col_groups: dict[str, list[dict[str, Any]]] = {}
        for i, d in enumerate(list_data):
            d["vector"] = embeddings[i].tolist()
            
            is_legal = False
            if "doc_type" in d and d["doc_type"] == "legal":
                is_legal = True
            elif d.get("legal_status") is not None or d.get("legal_document_id") is not None:
                is_legal = True
            elif os.environ.get("PIPELINE_DOC_TYPE") == "legal":
                is_legal = True

            if "workspace_id" not in d or not d["workspace_id"]:
                d["workspace_id"] = self.workspace

            if self.namespace.endswith("chunks"):
                col_name = "legal_chunks" if is_legal else "general_chunks"
                if "doc_id" in d:
                    del d["doc_id"]
                d["chunk_id"] = d.get("id")
                
                if "status" not in d or d["status"] is None:
                    d["status"] = d.get("legal_status_code") or "UNKNOWN"
                if "effective_from" not in d or d["effective_from"] is None:
                    d["effective_from"] = 0
                if "effective_to" not in d or d["effective_to"] is None:
                    d["effective_to"] = 0
            elif self.namespace.endswith("entities"):
                col_name = "legal_entities" if is_legal else "general_entities"
                d["entity_id"] = d.get("id")
                d["entity_name"] = d.get("entity_name") or d.get("id")
            else:
                continue

            col_groups.setdefault(col_name, []).append(d)

        # Upload to MinIO
        endpoint_url = os.environ.get("S3_ENDPOINT_URL")
        access_key = os.environ.get("S3_ACCESS_KEY") or os.environ.get(
            "S3_ACCESS_KEY_ID"
        )
        secret_key = os.environ.get("S3_SECRET_KEY") or os.environ.get(
            "S3_SECRET_ACCESS_KEY"
        )
        bucket = (
            os.environ.get("MILVUS_BULK_INSERT_BUCKET")
            or os.environ.get("S3_BUCKET")
            or "lightrag-markdown"
        )

        if not all([endpoint_url, access_key, secret_key]):
            raise ValueError("MinIO credentials missing. Cannot run bulk insert.")

        session = aioboto3.Session()
        
        all_task_ids = []
        for col_name, col_data in col_groups.items():
            if not col_data:
                continue
                
            json_payload = {"rows": col_data}
            file_content = json.dumps(json_payload).encode("utf-8")
            object_key = f"bulk_insert/{col_name}_{uuid.uuid4().hex}.json"

            async with session.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            ) as s3:
                await s3.put_object(Bucket=bucket, Key=object_key, Body=file_content)
                await self._cleanup_expired_bulk_insert_objects(
                    s3,
                    bucket,
                    current_key=object_key,
                )

            logger.info(f"Uploaded bulk insert data to {bucket}/{object_key}")

            # Trigger Milvus bulk insert
            from pymilvus import connections, utility

            conn_alias = f"bulk_insert_{uuid.uuid4().hex}"
            self._connect_bulk_insert_alias(conn_alias)

            try:
                task_id = await asyncio.to_thread(
                    self._milvus_call,
                    "do_bulk_insert",
                    lambda client, o=object_key, c=col_name, a=conn_alias: self._do_bulk_insert(
                        collection_name=c,
                        files=[o],
                        using=a,
                    ),
                )
                logger.info(f"Started Milvus bulk insert task: {task_id}")
                all_task_ids.append(task_id)
                
                # Start background task to poll status and cleanup file
                async def _poll_and_cleanup(tid, s3_session, endpoint, access, secret, bucket_name, obj_key):
                    import asyncio
                    from pymilvus import connections, utility
                    poll_alias = f"poll_bulk_insert_{uuid.uuid4().hex}"
                    
                    try:
                        await asyncio.to_thread(self._connect_bulk_insert_alias, poll_alias)
                        
                        final_state_str = ""
                        while True:
                            try:
                                def _get_state():
                                    return utility.get_bulk_insert_state(task_id=tid, using=poll_alias)
                                    
                                state_obj = await asyncio.to_thread(_get_state)
                                state_val = getattr(state_obj, "state", getattr(state_obj, "state_name", str(state_obj)))
                                state_str = str(state_val).lower()
                                
                                # BulkInsertState.ImportCompleted = 6, ImportFailed = 7, ImportFailedAndCleaned = 8
                                if "completed" in state_str or "failed" in state_str or getattr(state_obj, "state", -1) in (6, 7, 8):
                                    final_state_str = state_str
                                    break
                            except Exception as poll_err:
                                logger.warning(f"Error polling bulk insert state for task {tid}: {poll_err}")
                                break
                            
                            await asyncio.sleep(5)
                            
                        # Delete file from MinIO only on success
                        if "completed" in final_state_str or getattr(state_obj, "state", -1) == 6:
                            try:
                                async with s3_session.client(
                                    "s3",
                                    endpoint_url=endpoint,
                                    aws_access_key_id=access,
                                    aws_secret_access_key=secret,
                                ) as s3_client:
                                    await s3_client.delete_object(Bucket=bucket_name, Key=obj_key)
                                    logger.info(f"Successfully cleaned up bulk insert file: {bucket_name}/{obj_key}")
                            except Exception as s3_err:
                                logger.error(f"Failed to delete bulk insert file {obj_key}: {s3_err}")
                        else:
                            # Tắt cơ chế xoá file khi bulk-insert thất bại (User requested)
                            logger.info(f"Bulk insert incomplete or failed (state={final_state_str}). File retained in MinIO: {bucket_name}/{obj_key}")
                    finally:
                        try:
                            await asyncio.to_thread(connections.disconnect, poll_alias)
                        except Exception:
                            pass
                            
                asyncio.create_task(
                    _poll_and_cleanup(
                        task_id, session, endpoint_url, access_key, secret_key, bucket, object_key
                    )
                )
            finally:
                connections.disconnect(conn_alias)
                
        return all_task_ids if len(all_task_ids) > 1 else (all_task_ids[0] if all_task_ids else None)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> Any:
        if not data:
            return None

        # Check if we should use bulk insert
        bulk_insert_threshold = _get_env_int("MILVUS_BULK_INSERT_THRESHOLD", 100)
        if bulk_insert_threshold > 0 and len(data) >= bulk_insert_threshold:
            endpoint_url = os.environ.get("S3_ENDPOINT_URL")
            access_key = os.environ.get("S3_ACCESS_KEY") or os.environ.get("S3_ACCESS_KEY_ID")
            secret_key = os.environ.get("S3_SECRET_KEY") or os.environ.get("S3_SECRET_ACCESS_KEY")
            if all([endpoint_url, access_key, secret_key]):
                logger.info(
                    f"[{self.workspace}] Payload size {len(data)} >= threshold {bulk_insert_threshold}. "
                    f"Routing to bulk_upsert for {self.final_namespace}."
                )
                try:
                    return await self._bulk_upsert_impl(data)
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Bulk insert failed: {e}. "
                        f"Falling back to standard gRPC upsert."
                    )
                    # Fall back to standard gRPC
                    return await self._upsert_grpc(data)

        # Fallback/standard path
        return await self._upsert_grpc(data)

    async def bulk_upsert(self, data: dict[str, dict[str, Any]]) -> Any:
        """
        Public bulk_upsert interface. Calls the implementation directly, falling back to standard gRPC on failure.
        """
        try:
            return await self._bulk_upsert_impl(data)
        except Exception as e:
            logger.error(f"Milvus do_bulk_insert failed: {e}. Falling back to standard gRPC upsert.")
            return await self._upsert_grpc(data)

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        # Use provided embedding or compute it
        if query_embedding is not None:
            embedding = [query_embedding]  # Milvus expects a list of embeddings
        else:
            embedding = await self.embedding_func(
                [query], _priority=5
            )  # higher priority for query

        # Include all meta_fields (created_at is now always included)
        output_fields = list(self.meta_fields)

        # Build search params from index config
        search_params_base = self.index_config.build_search_params()

        # Do NOT use radius in search_params: Milvus radius filter is too
        # aggressive and can return 0 results when embeddings have low cosine
        # similarity (e.g. documents with heavy HTML table content).
        # Instead, we apply cosine_better_than_threshold as a post-filter below.
        search_params = {
            "metric_type": self.index_config.metric_type,
            "params": {
                **search_params_base.get("params", {}),
            },
        }

        # No workspace_id filter needed: the collection is already workspace-scoped
        # via final_namespace (e.g., ws_<uuid>_relationships). Double-filtering by
        # workspace_id breaks queries when legacy data has workspace_id="".
        logger.info(
            f"[{self.workspace}] Querying Milvus collection: {self.final_namespace}, "
            f"top_k: {top_k}, threshold: {self.cosine_better_than_threshold}"
        )

        with self._loaded_collection_context():
            import asyncio
            results = await asyncio.to_thread(
                self._milvus_call,
                "search",
                lambda client: client.search(
                    collection_name=self.final_namespace,
                    data=embedding,
                    limit=top_k,
                    output_fields=output_fields,
                    search_params=search_params,
                ),
            )

        # Post-filter by cosine similarity threshold.
        # NOTE: We apply threshold as a post-filter rather than via Milvus
        # radius param, because Milvus radius filtering is too aggressive
        # and returns 0 results for Vietnamese/HTML-heavy content where
        # cosine similarities tend to be lower (0.3-0.4 range).
        raw_count = len(results[0]) if results and len(results) > 0 else 0
        filtered = [
            {
                **dp["entity"],
                "id": dp["id"],
                "distance": dp["distance"],
                "created_at": dp.get("created_at"),
            }
            for dp in results[0]
            if dp["distance"] >= self.cosine_better_than_threshold
        ]

        logger.info(
            f"[{self.workspace}] Query returned {raw_count} raw results, "
            f"{len(filtered)} after threshold filter ({self.cosine_better_than_threshold})"
        )

        return filtered

    async def index_done_callback(self) -> None:
        """Build indexes and load the collection after insertion is done."""
        if _get_env_bool("MILVUS_DEFER_INDEX_BUILDING", False):
            logger.info(
                f"[{self.workspace}] Index building is deferred for {self.final_namespace} (MILVUS_DEFER_INDEX_BUILDING=True)"
            )
            return

        def _build_and_load():
            try:
                # Check if collection exists
                if not self._client.has_collection(self.final_namespace):
                    logger.warning(
                        f"[{self.workspace}] Collection {self.final_namespace} does not exist, skipping index creation"
                    )
                    return
                
                # Check if index exists
                indexes = self._client.list_indexes(self.final_namespace)
                if not indexes:
                    logger.info(
                        f"[{self.workspace}] Indexes not found in index_done_callback for {self.final_namespace}, creating them now"
                    )
                    self._create_indexes_after_collection()
                else:
                    logger.debug(
                        f"[{self.workspace}] Indexes already exist for collection {self.final_namespace}: {indexes}"
                    )
                
                # Load the collection
                if not self._collection_loaded:
                    if self._release_idle_collections_enabled():
                        logger.info(
                            f"[{self.workspace}] Skipping post-index load for "
                            f"{self.final_namespace} because "
                            "MILVUS_RELEASE_IDLE_COLLECTIONS=True"
                        )
                        return
                    logger.info(
                        f"[{self.workspace}] Loading collection {self.final_namespace} in index_done_callback"
                    )
                    self._client.load_collection(self.final_namespace)
                    self._collection_loaded = True
                    self._register_collection_loaded()
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to build index / load collection in index_done_callback: {e}"
                )
                
        await asyncio.to_thread(_build_and_load)

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity from the vector database

        Args:
            entity_name: The name of the entity to delete
        """
        try:
            # Compute entity ID from name
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"[{self.workspace}] Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            # Delete the entity from Milvus collection
            import asyncio
            result = await asyncio.to_thread(
                self._milvus_call,
                "delete",
                lambda client: client.delete(
                    collection_name=self.final_namespace, pks=[entity_id]
                ),
            )

            if result and result.get("delete_count", 0) > 0:
                logger.debug(
                    f"[{self.workspace}] Successfully deleted entity {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] Entity {entity_name} not found in storage"
                )

        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity

        Args:
            entity_name: The name of the entity whose relations should be deleted
        """
        try:
            # Search for relations where entity is either source or target
            expr = f'src_id == "{entity_name}" or tgt_id == "{entity_name}"'

            # Find all relations involving this entity
            import asyncio
            with self._loaded_collection_context():
                results = await asyncio.to_thread(
                    self._milvus_call,
                    "query",
                    lambda client: client.query(
                        collection_name=self.final_namespace,
                        filter=expr,
                        output_fields=["id"],
                    ),
                )

                if not results or len(results) == 0:
                    logger.debug(
                        f"[{self.workspace}] No relations found for entity {entity_name}"
                    )
                    return

                # Extract IDs of relations to delete
                relation_ids = [item["id"] for item in results]
                logger.debug(
                    f"[{self.workspace}] Found {len(relation_ids)} relations for entity {entity_name}"
                )

                # Delete the relations
                if relation_ids:
                    delete_result = await asyncio.to_thread(
                        self._milvus_call,
                        "delete",
                        lambda client: client.delete(
                            collection_name=self.final_namespace, pks=relation_ids
                        ),
                    )

                    logger.debug(
                        f"[{self.workspace}] Deleted {delete_result.get('delete_count', 0)} relations for {entity_name}"
                    )

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
            )

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """
        if not ids:
            return

        try:
            batch_size = 1000
            total_deleted = 0

            # Delete vectors by IDs in batches
            import asyncio
            with self._loaded_collection_context():
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    result = await asyncio.to_thread(
                        self._milvus_call,
                        "delete",
                        lambda client, b=batch_ids: client.delete(
                            collection_name=self.final_namespace, pks=b
                        ),
                    )
                    if result:
                        total_deleted += result.get("delete_count", 0)

            if total_deleted > 0:
                logger.debug(
                    f"[{self.workspace}] Successfully deleted {total_deleted} vectors from {self.namespace}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] No vectors were deleted from {self.namespace}"
                )

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}"
            )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        try:
            # Include all meta_fields (created_at is now always included) plus id
            output_fields = list(self.meta_fields) + ["id"]

            # Query Milvus for a specific ID
            import asyncio
            with self._loaded_collection_context():
                result = await asyncio.to_thread(
                    self._milvus_call,
                    "query",
                    lambda client: client.query(
                        collection_name=self.final_namespace,
                        filter=f'id == "{id}"',
                        output_fields=output_fields,
                    ),
                )

            if not result or len(result) == 0:
                return None

            return result[0]
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}"
            )
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        try:
            # Include all meta_fields (created_at is now always included) plus id
            output_fields = list(self.meta_fields) + ["id"]

            batch_size = 1000
            result_map: dict[str, dict[str, Any]] = {}

            # Process in batches to avoid Milvus expression length limits
            with self._loaded_collection_context():
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    id_list = '", "'.join(batch_ids)
                    filter_expr = f'id in ["{id_list}"]'

                    # Query Milvus with the filter
                    import asyncio
                    result = await asyncio.to_thread(
                        self._milvus_call,
                        "query",
                        lambda client, f=filter_expr: client.query(
                            collection_name=self.final_namespace,
                            filter=f,
                            output_fields=output_fields,
                        ),
                    )

                    if result:
                        for row in result:
                            if not row:
                                continue
                            row_id = row.get("id")
                            if row_id is not None:
                                result_map[str(row_id)] = row

            ordered_results: list[dict[str, Any] | None] = []
            for requested_id in ids:
                ordered_results.append(result_map.get(str(requested_id)))

            return ordered_results
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for IDs {len(ids)} items: {e}"
            )
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        if not ids:
            return {}

        try:
            vectors_dict = {}
            batch_size = 1000

            # Process in batches to avoid Milvus expression length limits
            with self._loaded_collection_context():
                for i in range(0, len(ids), batch_size):
                    batch_ids = ids[i:i + batch_size]
                    id_list = '", "'.join(batch_ids)
                    filter_expr = f'id in ["{id_list}"]'

                    # Query Milvus with the filter, requesting only vector field
                    import asyncio
                    result = await asyncio.to_thread(
                        self._milvus_call,
                        "query",
                        lambda client, f=filter_expr: client.query(
                            collection_name=self.final_namespace,
                            filter=f,
                            output_fields=["vector"],
                        ),
                    )

                    if result:
                        for item in result:
                            if item and "vector" in item and "id" in item:
                                # Convert numpy array to list if needed
                                vector_data = item["vector"]
                                if isinstance(vector_data, np.ndarray):
                                    vector_data = vector_data.tolist()
                                vectors_dict[str(item["id"])] = vector_data

            return vectors_dict
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}"
            )
            return {}

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources

        This method will delete all data from the Milvus collection.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            # Drop the collection and recreate it
            if self._milvus_call(
                "has_collection",
                lambda client: client.has_collection(self.final_namespace),
            ):
                self._milvus_call(
                    "drop_collection",
                    lambda client: client.drop_collection(self.final_namespace),
                )
            self._mark_collection_dropped()

            # Recreate the collection
            self._create_collection_if_not_exist()

            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop Milvus collection {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping Milvus collection {self.namespace}: {e}"
            )
            return {"status": "error", "message": str(e)}
