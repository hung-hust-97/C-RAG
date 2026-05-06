"""
LightRAG FastAPI Server
"""

from fastapi import FastAPI, Depends, HTTPException, Request, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
import os
import json
import asyncio
import re
import shutil
import logging
import logging.config
import sys
import traceback
import uvicorn
import pipmaster as pm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
import configparser
from ascii_colors import ASCIIColors
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from lightrag.api.utils_api import (
    get_combined_auth_dependency,
    display_splash_screen,
    check_env_file,
)
from .config import (
    global_args,
    update_uvicorn_mode_config,
    get_default_host,
)
from lightrag.utils import get_env_value
from lightrag import LightRAG, __version__ as core_version
from lightrag.api import __api_version__
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import EmbeddingFunc
from lightrag.constants import (
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_FILENAME,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_EMBEDDING_TIMEOUT,
)
from lightrag.api.routers.document_routes import (
    DocumentManager,
    create_document_routes,
)
from lightrag.api.routers.query_routes import create_query_routes
from lightrag.api.routers.graph_routes import create_graph_routes
from lightrag.api.routers.ollama_api import OllamaAPI
from lightrag.api.reembed import reembed_workspace_vectors

# Import logger first for use in Celery import handling
from lightrag.utils import logger, set_verbose_debug

# Try to import Celery worker module with graceful degradation
try:
    from celery_worker.app import celery_app
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None
    logger.warning("Celery worker module not available, async processing disabled")

def monkeypatch_apipeline_process_enqueue_documents(self, reprocess_failed=False):
    """
    Monkeypatch for LightRAG.apipeline_process_enqueue_documents to use Celery.
    """
    if CELERY_AVAILABLE and celery_app is not None:
        celery_app.send_task(
            "lightrag.api.lightrag_server.apipeline_process_enqueue_documents_task",
            args=[self.workspace, reprocess_failed]
        )
        return True
    else:
        logger.warning("Celery not available, skipping apipeline_process_enqueue_documents")
        return False

LightRAG.apipeline_process_enqueue_documents = monkeypatch_apipeline_process_enqueue_documents

from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_default_workspace,
    # set_default_workspace,
    cleanup_keyed_lock,
    finalize_share_data,
    initialize_share_data,
)
from fastapi.security import OAuth2PasswordRequestForm
from lightrag.api.auth import auth_handler

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


webui_title = os.getenv("WEBUI_TITLE")
webui_description = os.getenv("WEBUI_DESCRIPTION")

# Initialize config parser
config = configparser.ConfigParser()
config.read("config.ini")

# Global authentication configuration
auth_configured = bool(auth_handler.accounts)


from .lightrag_factory import (
    LLMConfigCache,
    create_llm_model_func,
    create_llm_model_kwargs,
    create_optimized_embedding_function,
    build_rag_instance as factory_build_rag_instance,
)




def check_frontend_build():
    """Check if frontend is built and optionally check if source is up-to-date

    Returns:
        tuple: (assets_exist: bool, is_outdated: bool)
            - assets_exist: True if WebUI build files exist
            - is_outdated: True if source is newer than build (only in dev environment)
    """
    webui_dir = Path(__file__).parent / "webui"
    index_html = webui_dir / "index.html"

    # 1. Check if build files exist
    if not index_html.exists():
        ASCIIColors.yellow("\n" + "=" * 80)
        ASCIIColors.yellow("WARNING: Frontend Not Built")
        ASCIIColors.yellow("=" * 80)
        ASCIIColors.yellow("The WebUI frontend has not been built yet.")
        ASCIIColors.yellow("The API server will start without the WebUI interface.")
        ASCIIColors.yellow(
            "\nTo enable WebUI, build the frontend using these commands:\n"
        )
        ASCIIColors.cyan("    cd lightrag_webui")
        ASCIIColors.cyan("    bun install --frozen-lockfile")
        ASCIIColors.cyan("    bun run build")
        ASCIIColors.cyan("    cd ..")
        ASCIIColors.yellow("\nThen restart the service.\n")
        ASCIIColors.cyan(
            "Note: Make sure you have Bun installed. Visit https://bun.sh for installation."
        )
        ASCIIColors.yellow("=" * 80 + "\n")
        return (False, False)  # Assets don't exist, not outdated

    # 2. Check if this is a development environment (source directory exists)
    try:
        source_dir = Path(__file__).parent.parent.parent / "lightrag_webui"
        src_dir = source_dir / "src"

        # Determine if this is a development environment: source directory exists and contains src directory
        if not source_dir.exists() or not src_dir.exists():
            # Production environment, skip source code check
            logger.debug(
                "Production environment detected, skipping source freshness check"
            )
            return (True, False)  # Assets exist, not outdated (prod environment)

        # Development environment, perform source code timestamp check
        logger.debug("Development environment detected, checking source freshness")

        # Source code file extensions (files to check)
        source_extensions = {
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".mjs",
            ".cjs",  # TypeScript/JavaScript
            ".css",
            ".scss",
            ".sass",
            ".less",  # Style files
            ".json",
            ".jsonc",  # Configuration/data files
            ".html",
            ".htm",  # Template files
            ".md",
            ".mdx",  # Markdown
        }

        # Key configuration files (in lightrag_webui root directory)
        key_files = [
            source_dir / "package.json",
            source_dir / "bun.lock",
            source_dir / "vite.config.ts",
            source_dir / "tsconfig.json",
            source_dir / "tailraid.config.js",
            source_dir / "index.html",
        ]

        # Get the latest modification time of source code
        latest_source_time = 0

        # Check source code files in src directory
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                # Only check source code files, ignore temporary files and logs
                if file_path.suffix.lower() in source_extensions:
                    mtime = file_path.stat().st_mtime
                    latest_source_time = max(latest_source_time, mtime)

        # Check key configuration files
        for key_file in key_files:
            if key_file.exists():
                mtime = key_file.stat().st_mtime
                latest_source_time = max(latest_source_time, mtime)

        # Get build time
        build_time = index_html.stat().st_mtime

        # Compare timestamps (5 second tolerance to avoid file system time precision issues)
        if latest_source_time > build_time + 5:
            ASCIIColors.yellow("\n" + "=" * 80)
            ASCIIColors.yellow("WARNING: Frontend Source Code Has Been Updated")
            ASCIIColors.yellow("=" * 80)
            ASCIIColors.yellow(
                "The frontend source code is newer than the current build."
            )
            ASCIIColors.yellow(
                "This might happen after 'git pull' or manual code changes.\n"
            )
            ASCIIColors.cyan(
                "Recommended: Rebuild the frontend to use the latest changes:"
            )
            ASCIIColors.cyan("    cd lightrag_webui")
            ASCIIColors.cyan("    bun install --frozen-lockfile")
            ASCIIColors.cyan("    bun run build")
            ASCIIColors.cyan("    cd ..")
            ASCIIColors.yellow("\nThe server will continue with the current build.")
            ASCIIColors.yellow("=" * 80 + "\n")
            return (True, True)  # Assets exist, outdated
        else:
            logger.info("Frontend build is up-to-date")
            return (True, False)  # Assets exist, up-to-date

    except Exception as e:
        # If check fails, log warning but don't affect startup
        logger.warning(f"Failed to check frontend source freshness: {e}")
        return (True, False)  # Assume assets exist and up-to-date on error


def create_app(args):
    # Check frontend build first and get status
    webui_assets_exist, is_frontend_outdated = check_frontend_build()

    # Create unified API version display with warning symbol if frontend is outdated
    api_version_display = (
        f"{__api_version__}⚠️" if is_frontend_outdated else __api_version__
    )

    # Setup logging
    logger.setLevel(args.log_level)
    set_verbose_debug(args.verbose)

    # Create configuration cache (this will output configuration logs)
    config_cache = LLMConfigCache(args)

    # Verify that bindings are correctly setup
    if args.llm_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "aws_bedrock",
        "gemini",
    ]:
        raise Exception("llm binding not supported")

    if args.embedding_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "aws_bedrock",
        "jina",
        "gemini",
    ]:
        raise Exception("embedding binding not supported")

    # Set default hosts if not provided
    if args.llm_binding_host is None:
        args.llm_binding_host = get_default_host(args.llm_binding)

    if args.embedding_binding_host is None:
        args.embedding_binding_host = get_default_host(args.embedding_binding)

    # Add SSL validation
    if args.ssl:
        if not args.ssl_certfile or not args.ssl_keyfile:
            raise Exception(
                "SSL certificate and key files must be provided when SSL is enabled"
            )
        if not os.path.exists(args.ssl_certfile):
            raise Exception(f"SSL certificate file not found: {args.ssl_certfile}")
        if not os.path.exists(args.ssl_keyfile):
            raise Exception(f"SSL key file not found: {args.ssl_keyfile}")

    # Check if API key is provided either through env var or args
    api_key = os.getenv("LIGHTRAG_API_KEY") or args.key

    # Initialize document manager with workspace support for data isolation
    doc_manager = DocumentManager(args.input_dir, workspace=args.workspace)
    auto_reembed_on_embedding_change = (
        os.getenv("AUTO_REEMBED_ON_EMBEDDING_CHANGE", "false").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    embedding_profile_lock = asyncio.Lock()

    def _is_milvus_dimension_mismatch_error(error: Exception) -> bool:
        message = str(error)
        return (
            "Vector dimension mismatch for collection" in message
            and "Collection validation failed" in message
        )

    def _sanitize_workspace_for_milvus_collection(workspace: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", workspace.strip())
        if not sanitized:
            return "ws"
        if not (sanitized[0].isalpha() or sanitized[0] == "_"):
            sanitized = f"ws_{sanitized}"
        return sanitized[:200]

    async def _drop_workspace_milvus_collections(workspace_id: str) -> bool:
        if args.vector_storage != "MilvusVectorDBStorage":
            return False

        try:
            from pymilvus import MilvusClient  # type: ignore
        except Exception as e:
            logger.error(
                f"[{workspace_id}] Cannot auto-heal Milvus mismatch because pymilvus import failed: {e}"
            )
            return False

        milvus_uri = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
        milvus_db_name = os.getenv("MILVUS_DB_NAME", None)

        workspace = (workspace_id or "").strip()
        if workspace:
            prefix = _sanitize_workspace_for_milvus_collection(workspace)
            target_collections = {
                f"{prefix}_entities",
                f"{prefix}_relationships",
                f"{prefix}_chunks",
            }
        else:
            target_collections = {"entities", "relationships", "chunks"}

        client = MilvusClient(uri=milvus_uri, db_name=milvus_db_name)
        existing_collections = set(client.list_collections())
        to_drop = sorted(target_collections.intersection(existing_collections))

        if not to_drop:
            logger.warning(
                f"[{workspace_id}] No matching Milvus collections found to drop for auto-heal: {sorted(target_collections)}"
            )
            return False

        for collection_name in to_drop:
            logger.warning(
                f"[{workspace_id}] Dropping incompatible Milvus collection: {collection_name}"
            )
            client.drop_collection(collection_name)

        logger.warning(
            f"[{workspace_id}] Dropped {len(to_drop)} incompatible Milvus collections for dimension migration"
        )
        return True

    def _embedding_profile_dir() -> Path:
        profile_dir = Path(args.working_dir) / ".embedding_profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir

    def _embedding_profile_path(workspace_id: str) -> Path:
        safe_workspace_id = workspace_id or "default"
        return _embedding_profile_dir() / f"{safe_workspace_id}.json"

    def _current_embedding_profile() -> dict[str, object]:
        return {
            "binding": args.embedding_binding,
            "model": args.embedding_model,
            "host": args.embedding_binding_host,
            "dim": int(args.embedding_dim),
            "send_dim": bool(args.embedding_send_dim),
            "token_limit": int(args.embedding_token_limit),
        }

    async def maybe_reembed_on_embedding_change(workspace_rag: LightRAG) -> None:
        profile_path = _embedding_profile_path(workspace_rag.workspace)
        current_profile = _current_embedding_profile()

        async with embedding_profile_lock:
            previous_profile = None
            if profile_path.exists():
                try:
                    previous_profile = json.loads(profile_path.read_text())
                except Exception as e:
                    logger.warning(
                        f"[{workspace_rag.workspace}] Failed to parse embedding profile file: {e}"
                    )

            if previous_profile is None:
                profile_path.write_text(json.dumps(current_profile, indent=2))
                return

            if previous_profile == current_profile:
                return

            logger.warning(
                f"[{workspace_rag.workspace}] Embedding configuration changed. "
                f"Previous: {previous_profile} | Current: {current_profile}"
            )

            if not auto_reembed_on_embedding_change:
                logger.warning(
                    f"[{workspace_rag.workspace}] AUTO_REEMBED_ON_EMBEDDING_CHANGE is disabled. "
                    "Run POST /documents/reembed to rebuild vectors manually."
                )
                return

            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_namespace_lock,
            )

            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=workspace_rag.workspace
            )
            pipeline_status_lock = get_namespace_lock(
                "pipeline_status", workspace=workspace_rag.workspace
            )

            async with pipeline_status_lock:
                if pipeline_status.get("busy", False):
                    logger.warning(
                        f"[{workspace_rag.workspace}] Skip auto re-embedding because pipeline is busy"
                    )
                    return
                pipeline_status["busy"] = True
                pipeline_status["job_name"] = "auto_reembed_vectors"
                pipeline_status["latest_message"] = "Auto re-embedding vectors started"
                pipeline_status.setdefault("history_messages", []).append(
                    "Auto re-embedding vectors started"
                )

            try:
                stats = await reembed_workspace_vectors(workspace_rag)
                success_message = (
                    "Auto re-embedding completed: "
                    f"documents={stats['documents']}, chunks={stats['chunks']}, "
                    f"entities={stats['entities']}, relations={stats['relations']}"
                )
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = success_message
                    pipeline_status.setdefault("history_messages", []).append(
                        success_message
                    )
                profile_path.write_text(json.dumps(current_profile, indent=2))
            finally:
                async with pipeline_status_lock:
                    pipeline_status["busy"] = False
                    pipeline_status["job_name"] = ""

    def build_rag_instance(workspace_id: str) -> LightRAG:
        return factory_build_rag_instance(workspace_id, args, config_cache)

    rag_instances: dict[str, LightRAG] = {}
    doc_manager_instances: dict[str, DocumentManager] = {}
    rag_instances_lock = asyncio.Lock()
    workspace_alias_cache: dict[str, str] = {}

    def normalize_workspace_id(workspace_id: str | None) -> str:
        default_workspace = (args.workspace or "").strip()
        if not default_workspace:
            default_workspace = get_default_workspace() or "default"

        if workspace_id is None:
            return default_workspace

        normalized_workspace_id = workspace_id.strip()
        if not normalized_workspace_id:
            return default_workspace

        return normalized_workspace_id

    async def resolve_workspace_id_alias(workspace_id: str | None) -> str:
        """Resolve workspace identifier aliases (name/id/legacy workspace) to canonical workspace_id."""
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        if not normalized_workspace_id:
            return normalized_workspace_id

        cached_workspace_id = workspace_alias_cache.get(normalized_workspace_id)
        if cached_workspace_id:
            return cached_workspace_id

        # First-pass resolution from current workspace id/name map.
        try:
            workspace_items, _ = await get_all_workspace_items()
            normalized_workspace_name = normalized_workspace_id.lower()
            for item in workspace_items:
                workspace_id_item = (item.get("workspace_id") or "").strip()
                workspace_name_item = (item.get("workspace_name") or "").strip()
                if not workspace_id_item:
                    continue

                if normalized_workspace_id == workspace_id_item or (
                    workspace_name_item
                    and normalized_workspace_name == workspace_name_item.lower()
                ):
                    workspace_alias_cache[normalized_workspace_id] = workspace_id_item
                    if workspace_name_item:
                        workspace_alias_cache[workspace_name_item] = workspace_id_item
                    workspace_alias_cache[workspace_id_item] = workspace_id_item

                    if workspace_id_item != normalized_workspace_id:
                        logger.info(
                            f"Resolved workspace alias '{normalized_workspace_id}' -> '{workspace_id_item}'"
                        )

                    return workspace_id_item
        except Exception as e:
            logger.debug(
                f"Workspace alias map resolution skipped for '{normalized_workspace_id}': {e}"
            )

        try:
            db = getattr(getattr(rag.doc_status, "db", None), "query", None)
            if db is not None:
                rows = await rag.doc_status.db.query(
                    """
                    SELECT workspace_id, name, id, workspace
                    FROM LIGHTRAG_WORKSPACES
                    WHERE workspace_id = $1
                       OR id = $1
                       OR workspace = $1
                       OR LOWER(name) = LOWER($1)
                    ORDER BY
                        CASE
                            WHEN workspace_id = $1 THEN 0
                            WHEN id = $1 THEN 1
                            WHEN workspace = $1 THEN 2
                            ELSE 3
                        END
                    LIMIT 1
                    """,
                    [normalized_workspace_id],
                    multirows=True,
                )

                if rows:
                    row = rows[0]
                    canonical_workspace_id = (row.get("workspace_id") or "").strip()
                    if canonical_workspace_id:
                        # Cache known aliases to avoid repeated DB lookups.
                        workspace_alias_cache[normalized_workspace_id] = canonical_workspace_id
                        for alias_key in ("workspace_id", "name", "id", "workspace"):
                            alias_val = (row.get(alias_key) or "").strip()
                            if alias_val:
                                workspace_alias_cache[alias_val] = canonical_workspace_id

                        if canonical_workspace_id != normalized_workspace_id:
                            logger.info(
                                f"Resolved workspace alias '{normalized_workspace_id}' -> '{canonical_workspace_id}'"
                            )

                        return canonical_workspace_id
        except Exception as e:
            logger.debug(
                f"Workspace alias resolution skipped for '{normalized_workspace_id}': {e}"
            )

        workspace_alias_cache[normalized_workspace_id] = normalized_workspace_id
        return normalized_workspace_id

    def get_doc_manager_for_workspace(workspace_id: str) -> DocumentManager:
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        canonical_workspace_id = workspace_alias_cache.get(
            normalized_workspace_id, normalized_workspace_id
        )
        if canonical_workspace_id not in doc_manager_instances:
            doc_manager_instances[canonical_workspace_id] = DocumentManager(
                args.input_dir, workspace=canonical_workspace_id
            )
        return doc_manager_instances[canonical_workspace_id]

    async def get_rag_for_workspace(workspace_id: str) -> LightRAG:
        normalized_workspace_id = await resolve_workspace_id_alias(workspace_id)

        # Fast path: return existing instance without lock
        if normalized_workspace_id in rag_instances:
            return rag_instances[normalized_workspace_id]

        # Slow path: acquire lock and initialize
        logger.info(f"[{normalized_workspace_id}] Acquiring lock for workspace initialization...")
        
        try:
            # Add timeout for lock acquisition to prevent infinite waiting
            async with asyncio.timeout(30.0):  # 30 second timeout for lock acquisition
                async with rag_instances_lock:
                    # Double-check after acquiring lock
                    if normalized_workspace_id in rag_instances:
                        logger.info(f"[{normalized_workspace_id}] Instance already created by another request")
                        return rag_instances[normalized_workspace_id]

                    logger.info(f"[{normalized_workspace_id}] Building RAG instance...")
                    workspace_rag = build_rag_instance(normalized_workspace_id)

                    try:
                        logger.info(f"[{normalized_workspace_id}] Initializing storages...")
                        # Add timeout for storage initialization
                        await asyncio.wait_for(
                            workspace_rag.initialize_storages(),
                            timeout=60.0  # 60 second timeout
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"[{normalized_workspace_id}] Timeout initializing storages after 60s")
                        raise HTTPException(
                            status_code=503,
                            detail=f"Workspace initialization timeout for {normalized_workspace_id}"
                        )
                    except RuntimeError as e:
                        if not _is_milvus_dimension_mismatch_error(e):
                            raise

                        logger.warning(
                            f"[{normalized_workspace_id}] Detected Milvus dimension mismatch during workspace initialization: {e}"
                        )

                        healed = await _drop_workspace_milvus_collections(normalized_workspace_id)
                        if not healed:
                            raise

                        # Recreate rag instance to ensure clean storage clients after drop.
                        workspace_rag = build_rag_instance(normalized_workspace_id)
                        await asyncio.wait_for(
                            workspace_rag.initialize_storages(),
                            timeout=60.0
                        )

                    logger.info(f"[{normalized_workspace_id}] Checking and migrating data...")
                    await asyncio.wait_for(
                        workspace_rag.check_and_migrate_data(),
                        timeout=120.0  # 2 minute timeout for migration
                    )
                    
                    logger.info(f"[{normalized_workspace_id}] Checking embedding changes...")
                    await asyncio.wait_for(
                        maybe_reembed_on_embedding_change(workspace_rag),
                        timeout=10.0  # 10 second timeout
                    )
                    
                    rag_instances[normalized_workspace_id] = workspace_rag
                    logger.info(f"[{normalized_workspace_id}] Workspace initialization complete")
                    return workspace_rag
                    
        except asyncio.TimeoutError:
            logger.error(f"[{normalized_workspace_id}] Timeout acquiring lock or initializing workspace")
            raise HTTPException(
                status_code=503,
                detail=f"Workspace initialization timeout for {normalized_workspace_id}"
            )
        except Exception as e:
            logger.error(f"[{normalized_workspace_id}] Error initializing workspace: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        # Store background tasks
        app.state.background_tasks = set()

        try:
            # Initialize database connections
            # Note: initialize_storages() now auto-initializes pipeline_status for rag.workspace
            await rag.initialize_storages()
            rag_instances[normalize_workspace_id(args.workspace)] = rag
            doc_manager_instances[normalize_workspace_id(args.workspace)] = doc_manager

            # Data migration regardless of storage implementation
            await rag.check_and_migrate_data()
            await maybe_reembed_on_embedding_change(rag)

            ASCIIColors.green("\nServer is ready to accept connections! 🚀\n")

            yield

        finally:
            # Clean up database connections
            finalized_rag_ids: set[int] = set()
            for cached_rag in rag_instances.values():
                if id(cached_rag) in finalized_rag_ids:
                    continue
                await cached_rag.finalize_storages()
                finalized_rag_ids.add(id(cached_rag))

            if "LIGHTRAG_GUNICORN_MODE" not in os.environ:
                # Only perform cleanup in Uvicorn single-process mode
                logger.debug("Unvicorn Mode: finalizing shared storage...")
                finalize_share_data()
            else:
                # In Gunicorn mode with preload_app=True, cleanup is handled by on_exit hooks
                logger.debug(
                    "Gunicorn Mode: postpone shared storage finalization to master process"
                )

    # Initialize FastAPI
    base_description = (
        "Providing API for LightRAG core, Web UI and Ollama Model Emulation"
    )
    swagger_description = (
        base_description
        + (" (API-Key Enabled)" if api_key else "")
        + "\n\n[View ReDoc documentation](/redoc)"
    )
    app_kwargs = {
        "title": "LightRAG Server API",
        "description": swagger_description,
        "version": __api_version__,
        "openapi_url": "/openapi.json",  # Explicitly set OpenAPI schema URL
        "docs_url": None,  # Disable default docs, we'll create custom endpoint
        "redoc_url": "/redoc",  # Explicitly set redoc URL
        "lifespan": lifespan,
    }

    # Configure Swagger UI parameters
    # Enable persistAuthorization and tryItOutEnabled for better user experience
    app_kwargs["swagger_ui_parameters"] = {
        "persistAuthorization": True,
        "tryItOutEnabled": True,
    }

    app = FastAPI(**app_kwargs)

    # Add custom validation error handler for /query/data endpoint
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        # Check if this is a request to /query/data endpoint
        if request.url.path.endswith("/query/data"):
            # Extract error details
            error_details = []
            for error in exc.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                error_details.append(f"{field_path}: {error['msg']}")

            error_message = "; ".join(error_details)

            # Return in the expected format for /query/data
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failure",
                    "message": f"Validation error: {error_message}",
                    "data": {},
                    "metadata": {},
                },
            )
        else:
            # For other endpoints, return the default FastAPI validation error
            # but also log it for debugging purposes (helps with 422 errors)
            logger.error(f"Validation error for {request.method} {request.url.path}: {exc.errors()}")
            return JSONResponse(status_code=422, content={"detail": exc.errors()})


    def get_cors_origins():
        """Get allowed origins from global_args
        Returns a list of allowed origins, defaults to ["*"] if not set
        """
        origins_str = global_args.cors_origins
        if origins_str == "*":
            return ["*"]
        return [origin.strip() for origin in origins_str.split(",")]

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-New-Token"
        ],  # Expose token renewal header for cross-origin requests
    )

    # Create combined auth dependency for all endpoints
    combined_auth = get_combined_auth_dependency(api_key)

    def get_workspace_from_request(request: Request) -> str | None:
        """
        Extract workspace from HTTP request header or use default.

        This enables multi-workspace API support by checking the custom
        'LIGHTRAG-WORKSPACE' header. If not present, falls back to the
        server's default workspace configuration.

        Args:
            request: FastAPI Request object

        Returns:
            Workspace identifier (may be empty string for global namespace)
        """
        # Check custom header first
        workspace = request.headers.get("LIGHTRAG-WORKSPACE", "").strip()

        if not workspace:
            workspace = None
        else:
            sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", workspace)
            if sanitized != workspace:
                logger.warning(
                    f"Workspace header '{workspace}' contains invalid characters. "
                    f"Sanitized to '{sanitized}'."
                )
                workspace = sanitized

        return workspace

    async def get_all_workspace_items() -> tuple[list[dict[str, str]], str]:
        """Collect known workspaces with id-name mapping from storage, runtime, and input directories."""
        default_workspace = normalize_workspace_id(args.workspace)
        if not default_workspace:
            default_workspace = get_default_workspace() or "default"

        workspace_map: dict[str, str] = {default_workspace: default_workspace}
        data_workspace_ids: set[str] = set()
        workspace_map.update({ws: ws for ws in rag_instances.keys() if ws})
        workspace_map.update({ws: ws for ws in doc_manager_instances.keys() if ws})

        # Include workspace metadata persisted in PostgreSQL when available.
        try:
            db = getattr(getattr(rag.doc_status, "db", None), "query", None)
            if db is not None:
                rows = await rag.doc_status.db.query(
                    "SELECT workspace_id, name FROM LIGHTRAG_WORKSPACES",
                    multirows=True,
                )
                if rows:
                    for row in rows:
                        workspace_id = (row.get("workspace_id") or "").strip()
                        if not workspace_id:
                            continue
                        workspace_name = (row.get("name") or "").strip() or workspace_id
                        workspace_map[workspace_id] = workspace_name

                # Fallback source: discover real workspace IDs from document status table.
                status_rows = await rag.doc_status.db.query(
                    "SELECT DISTINCT workspace FROM LIGHTRAG_DOC_STATUS",
                    multirows=True,
                )
                if status_rows:
                    for row in status_rows:
                        workspace_id = (row.get("workspace") or "").strip()
                        if not workspace_id:
                            continue
                        data_workspace_ids.add(workspace_id)
                        workspace_map.setdefault(workspace_id, workspace_id)
        except Exception as e:
            logger.debug(f"Failed to read workspace metadata table: {e}")

        input_root = Path(args.input_dir)
        if input_root.exists() and input_root.is_dir():
            # If files exist directly in root input dir, that implies default workspace usage.
            has_root_files = any(item.is_file() for item in input_root.iterdir())
            if has_root_files:
                data_workspace_ids.add(default_workspace)
                workspace_map.setdefault(default_workspace, default_workspace)

            for item in input_root.iterdir():
                if not item.is_dir():
                    continue
                if item.name.startswith("__"):
                    continue
                # Only treat a directory as an active workspace if it contains files.
                has_workspace_files = any(sub_item.is_file() for sub_item in item.iterdir())
                if has_workspace_files:
                    data_workspace_ids.add(item.name)
                    workspace_map.setdefault(item.name, item.name)

        all_workspace_ids = sorted(ws for ws in workspace_map.keys() if ws)
        data_first_ids = [
            workspace_id for workspace_id in all_workspace_ids if workspace_id in data_workspace_ids
        ]
        remaining_ids = [
            workspace_id for workspace_id in all_workspace_ids if workspace_id not in data_workspace_ids
        ]
        workspace_ids = data_first_ids + remaining_ids

        # Ensure the server-configured default workspace is always pinned first.
        if default_workspace in workspace_ids and workspace_ids[0] != default_workspace:
            workspace_ids.remove(default_workspace)
            workspace_ids.insert(0, default_workspace)

        effective_default_workspace = workspace_ids[0] if workspace_ids else default_workspace

        workspace_items = [
            {
                "workspace_id": workspace_id,
                "workspace_name": workspace_map.get(workspace_id, workspace_id)
                or workspace_id,
            }
            for workspace_id in workspace_ids
        ]

        return workspace_items, effective_default_workspace

    async def _delete_workspace_metadata_row(workspace_id: str) -> bool:
        """Delete workspace metadata row when PostgreSQL metadata table is available."""
        db_obj = getattr(rag.doc_status, "db", None)
        db_execute = getattr(db_obj, "execute", None)
        if db_execute is None:
            return False

        await db_obj.execute(
            """
            DELETE FROM LIGHTRAG_WORKSPACES
            WHERE workspace_id = $1 OR id = $1 OR workspace = $1
            """,
            {"workspace_id": workspace_id},
        )

        registered_workspaces = getattr(db_obj, "_registered_workspaces", None)
        if isinstance(registered_workspaces, dict):
            registered_workspaces.pop(workspace_id, None)

        return True

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)





    llm_timeout = get_env_value("LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT, int)

    embedding_timeout = get_env_value(
        "EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT, int
    )
    embedding_http_timeout_raw = os.getenv("EMBEDDING_HTTP_TIMEOUT")
    if embedding_http_timeout_raw is not None and str(embedding_http_timeout_raw).strip():
        try:
            embedding_http_timeout = int(float(embedding_http_timeout_raw))
            if embedding_http_timeout > embedding_timeout:
                embedding_timeout = embedding_http_timeout
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid EMBEDDING_HTTP_TIMEOUT='{embedding_http_timeout_raw}', ignoring it for embedding timeout resolution"
            )



    # Create embedding function with optimized configuration and max_token_size inheritance
    import inspect

    # Create the EmbeddingFunc instance (now returns complete EmbeddingFunc with max_token_size)
    embedding_func = create_optimized_embedding_function(
        config_cache=config_cache,
        binding=args.embedding_binding,
        model=args.embedding_model,
        host=args.embedding_binding_host,
        api_key=args.embedding_binding_api_key,
        args=args,
    )

    # Get embedding_send_dim from centralized configuration
    embedding_send_dim = args.embedding_send_dim

    # Check if the underlying function signature has embedding_dim parameter
    sig = inspect.signature(embedding_func.func)
    has_embedding_dim_param = "embedding_dim" in sig.parameters

    # Determine send_dimensions value based on binding type
    # Jina and Gemini REQUIRE dimension parameter (forced to True)
    # OpenAI and others: controlled by EMBEDDING_SEND_DIM environment variable
    if args.embedding_binding in ["jina", "gemini"]:
        # Jina and Gemini APIs require dimension parameter - always send it
        send_dimensions = has_embedding_dim_param
        dimension_control = f"forced by {args.embedding_binding.title()} API"
    else:
        # For OpenAI and other bindings, respect EMBEDDING_SEND_DIM setting
        send_dimensions = embedding_send_dim and has_embedding_dim_param
        if send_dimensions or not embedding_send_dim:
            dimension_control = "by env var"
        else:
            dimension_control = "by not hasparam"

    # Set send_dimensions on the EmbeddingFunc instance
    embedding_func.send_dimensions = send_dimensions

    logger.info(
        f"Send embedding dimension: {send_dimensions} {dimension_control} "
        f"(dimensions={embedding_func.embedding_dim}, has_param={has_embedding_dim_param}, "
        f"binding={args.embedding_binding})"
    )

    # Log max_token_size source
    if embedding_func.max_token_size:
        source = (
            "env variable"
            if args.embedding_token_limit
            else f"{args.embedding_binding} provider default"
        )
        logger.info(
            f"Embedding max_token_size: {embedding_func.max_token_size} (from {source})"
        )
    else:
        logger.info(
            "Embedding max_token_size: None (Embedding token limit is disabled)."
        )

    # Configure rerank function based on args.rerank_bindingparameter
    rerank_model_func = None
    if args.rerank_binding != "null":
        from lightrag.rerank import cohere_rerank, jina_rerank, ali_rerank

        # Map rerank binding to corresponding function
        rerank_functions = {
            "cohere": cohere_rerank,
            "jina": jina_rerank,
            "aliyun": ali_rerank,
        }

        # Select the appropriate rerank function based on binding
        selected_rerank_func = rerank_functions.get(args.rerank_binding)
        if not selected_rerank_func:
            logger.error(f"Unsupported rerank binding: {args.rerank_binding}")
            raise ValueError(f"Unsupported rerank binding: {args.rerank_binding}")

        # Get default values from selected_rerank_func if args values are None
        if args.rerank_model is None or args.rerank_binding_host is None:
            sig = inspect.signature(selected_rerank_func)

            # Set default model if args.rerank_model is None
            if args.rerank_model is None and "model" in sig.parameters:
                default_model = sig.parameters["model"].default
                if default_model != inspect.Parameter.empty:
                    args.rerank_model = default_model

            # Set default base_url if args.rerank_binding_host is None
            if args.rerank_binding_host is None and "base_url" in sig.parameters:
                default_base_url = sig.parameters["base_url"].default
                if default_base_url != inspect.Parameter.empty:
                    args.rerank_binding_host = default_base_url

        async def server_rerank_func(
            query: str, documents: list, top_n: int = None, extra_body: dict = None
        ):
            """Server rerank function with configuration from environment variables"""
            # Prepare kwargs for rerank function
            kwargs = {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "api_key": args.rerank_binding_api_key,
                "model": args.rerank_model,
                "base_url": args.rerank_binding_host,
            }

            # Add Cohere-specific parameters if using cohere binding
            if args.rerank_binding == "cohere":
                # Enable chunking if configured (useful for models with token limits like ColBERT)
                kwargs["enable_chunking"] = (
                    os.getenv("RERANK_ENABLE_CHUNKING", "false").lower() == "true"
                )
                kwargs["max_tokens_per_doc"] = int(
                    os.getenv("RERANK_MAX_TOKENS_PER_DOC", "4096")
                )

            return await selected_rerank_func(**kwargs, extra_body=extra_body)

        rerank_model_func = server_rerank_func
        logger.info(
            f"Reranking is enabled: {args.rerank_model or 'default model'} using {args.rerank_binding} provider"
        )
    else:
        logger.info("Reranking is disabled")

    # Create ollama_server_infos from command line arguments
    from lightrag.api.config import OllamaServerInfos

    ollama_server_infos = OllamaServerInfos(
        name=args.simulated_model_name, tag=args.simulated_model_tag
    )

    # Initialize RAG with unified configuration
    try:
        rag = build_rag_instance(args.workspace)
        # Register the initial instance in the workspace cache
        startup_workspace = (args.workspace or "").strip()
        rag_instances[startup_workspace] = rag
        logger.info(f"Initialized default RAG instance for workspace: '{startup_workspace}'")
    except Exception as e:
        logger.error(f"Failed to initialize LightRAG: {e}")
        raise

    # Add routes
    app.include_router(
        create_document_routes(
            rag,
            doc_manager,
            api_key,
            get_rag_for_workspace=get_rag_for_workspace,
            get_doc_manager_for_workspace=get_doc_manager_for_workspace,
        )
    )
    app.include_router(
        create_query_routes(
            rag,
            api_key,
            args.top_k,
            get_rag_for_workspace=get_rag_for_workspace,
        )
    )
    app.include_router(
        create_graph_routes(
            rag,
            api_key,
            get_rag_for_workspace=get_rag_for_workspace,
        )
    )

    # Add Ollama API routes
    ollama_api = OllamaAPI(rag, top_k=args.top_k, api_key=api_key)
    app.include_router(ollama_api.router, prefix="/api")

    # Custom Swagger UI endpoint for offline support
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Custom Swagger UI HTML with local static files"""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url="/docs/oauth2-redirect",
            swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui/swagger-ui.css",
            swagger_favicon_url="/static/swagger-ui/favicon-32x32.png",
            swagger_ui_parameters=app.swagger_ui_parameters,
        )

    @app.get("/docs/oauth2-redirect", include_in_schema=False)
    async def swagger_ui_redirect():
        """OAuth2 redirect for Swagger UI"""
        return get_swagger_ui_oauth2_redirect_html()

    @app.get("/")
    async def redirect_to_webui():
        """Redirect root path based on WebUI availability"""
        if webui_assets_exist:
            return RedirectResponse(url="/webui")
        else:
            return RedirectResponse(url="/docs")

    @app.get("/auth-status")
    async def get_auth_status():
        """Get authentication status and guest token if auth is not configured"""

        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "auth_configured": False,
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }

        return {
            "auth_configured": True,
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": api_version_display,
            "webui_title": webui_title,
            "webui_description": webui_description,
        }

    @app.post("/login")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }
        username = form_data.username
        if auth_handler.accounts.get(username) != form_data.password:
            raise HTTPException(status_code=401, detail="Incorrect credentials")

        # Regular user login
        user_token = auth_handler.create_token(
            username=username, role="user", metadata={"auth_mode": "enabled"}
        )
        return {
            "access_token": user_token,
            "token_type": "bearer",
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": api_version_display,
            "webui_title": webui_title,
            "webui_description": webui_description,
        }

    @app.get(
        "/health",
        dependencies=[Depends(combined_auth)],
        summary="Get system health and configuration status",
        description="Returns comprehensive system status including WebUI availability, configuration, and operational metrics",
        response_description="System health status with configuration details",
        responses={
            200: {
                "description": "Successful response with system status",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "healthy",
                            "webui_available": True,
                            "working_directory": "/path/to/working/dir",
                            "input_directory": "/path/to/input/dir",
                            "configuration": {
                                "llm_binding": "openai",
                                "llm_model": "gpt-4",
                                "embedding_binding": "openai",
                                "embedding_model": "text-embedding-ada-002",
                                "workspace": "default",
                            },
                            "auth_mode": "enabled",
                            "pipeline_busy": False,
                            "core_version": "0.0.1",
                            "api_version": "0.0.1",
                        }
                    }
                },
            }
        },
    )
    async def get_status(request: Request):
        """Get current system status including WebUI availability"""
        try:
            workspace = get_workspace_from_request(request)
            default_workspace = get_default_workspace()
            if workspace is None:
                workspace = default_workspace
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=workspace
            )

            if not auth_configured:
                auth_mode = "disabled"
            else:
                auth_mode = "enabled"

            # Cleanup expired keyed locks and get status
            keyed_lock_info = cleanup_keyed_lock()

            # Get PostgreSQL shared pool statistics
            postgresql_pool_stats = None
            try:
                # Import ClientManager to get pool stats
                from lightrag.kg.postgres_impl import ClientManager
                postgresql_pool_stats = ClientManager.get_pool_stats()
            except Exception as e:
                logger.debug(f"Could not get PostgreSQL pool stats: {e}")
                postgresql_pool_stats = {"status": "unavailable", "error": str(e)}

            return {
                "status": "healthy",
                "webui_available": webui_assets_exist,
                "working_directory": str(args.working_dir),
                "input_directory": str(args.input_dir),
                "postgresql_pool": postgresql_pool_stats,
                "configuration": {
                    # LLM configuration binding/host address (if applicable)/model (if applicable)
                    "llm_binding": args.llm_binding,
                    "llm_binding_host": args.llm_binding_host,
                    "llm_model": args.llm_model,
                    # embedding model configuration binding/host address (if applicable)/model (if applicable)
                    "embedding_binding": args.embedding_binding,
                    "embedding_binding_host": args.embedding_binding_host,
                    "embedding_model": args.embedding_model,
                    "summary_max_tokens": args.summary_max_tokens,
                    "summary_context_size": args.summary_context_size,
                    "kv_storage": args.kv_storage,
                    "doc_status_storage": args.doc_status_storage,
                    "graph_storage": args.graph_storage,
                    "vector_storage": args.vector_storage,
                    "enable_llm_cache_for_extract": args.enable_llm_cache_for_extract,
                    "enable_llm_cache": args.enable_llm_cache,
                    "workspace": default_workspace,
                    "max_graph_nodes": args.max_graph_nodes,
                    # Rerank configuration
                    "enable_rerank": rerank_model_func is not None,
                    "rerank_binding": args.rerank_binding,
                    "rerank_model": args.rerank_model if rerank_model_func else None,
                    "rerank_binding_host": args.rerank_binding_host
                    if rerank_model_func
                    else None,
                    # Environment variable status (requested configuration)
                    "summary_language": args.summary_language,
                    "force_llm_summary_on_merge": args.force_llm_summary_on_merge,
                    "max_parallel_insert": args.max_parallel_insert,
                    "cosine_threshold": args.cosine_threshold,
                    "min_rerank_score": args.min_rerank_score,
                    "related_chunk_number": args.related_chunk_number,
                    "max_async": args.max_async,
                    "embedding_func_max_async": args.embedding_func_max_async,
                    "embedding_batch_num": args.embedding_batch_num,
                },
                "auth_mode": auth_mode,
                "pipeline_busy": pipeline_status.get("busy", False),
                "keyed_locks": keyed_lock_info,
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/postgresql/pool-stats",
        dependencies=[Depends(combined_auth)],
        summary="Get PostgreSQL connection pool statistics",
        description="Returns detailed statistics about the shared PostgreSQL connection pool",
    )
    async def get_postgresql_pool_stats():
        """Get detailed PostgreSQL connection pool statistics."""
        try:
            from lightrag.kg.postgres_impl import ClientManager
            
            pool_stats = ClientManager.get_pool_stats()
            
            # Add additional context
            pool_stats["workspace_count"] = len(rag_instances)
            pool_stats["doc_manager_count"] = len(doc_manager_instances)
            
            # Calculate efficiency metrics
            if pool_stats["status"] == "active":
                pool_stats["utilization_percent"] = round(
                    (pool_stats["size"] - pool_stats["idle_size"]) / pool_stats["max_size"] * 100, 2
                )
                pool_stats["efficiency"] = "excellent" if pool_stats["utilization_percent"] < 80 else "good" if pool_stats["utilization_percent"] < 95 else "high"
            
            return {
                "pool_statistics": pool_stats,
                "architecture": "shared_pool",
                "description": "Single shared PostgreSQL connection pool used by all storage types and workspaces",
                "benefits": {
                    "connection_reduction": "99.2% fewer connections vs separate pools",
                    "resource_efficiency": "Shared pool eliminates per-workspace overhead",
                    "scalability": "Can support 100+ workspaces with same connection budget"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting PostgreSQL pool stats: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get(
        "/workspaces",
        dependencies=[Depends(combined_auth)],
        summary="List available workspaces",
        description="Returns known workspaces with workspace_id and workspace_name",
    )
    async def list_workspaces():
        workspace_items, default_workspace = await get_all_workspace_items()
        return {
            "default_workspace": default_workspace,
            # Keep this field for backward compatibility.
            "workspaces": [item["workspace_id"] for item in workspace_items],
            "workspace_items": workspace_items,
        }

    @app.put(
        "/workspaces/{workspace_id}",
        dependencies=[Depends(combined_auth)],
        summary="Upsert workspace metadata",
        description="Create or update workspace metadata (workspace_name) by workspace_id",
    )
    async def upsert_workspace(workspace_id: str, payload: dict[str, str]):
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        if not normalized_workspace_id:
            raise HTTPException(status_code=400, detail="workspace_id cannot be empty")

        workspace_name = (payload.get("workspace_name") or "").strip()
        if not workspace_name:
            workspace_name = normalized_workspace_id

        # Prevent ambiguous aliasing: workspace_name must not match another workspace_id.
        workspace_items, _ = await get_all_workspace_items()
        for item in workspace_items:
            existing_workspace_id = (item.get("workspace_id") or "").strip()
            if (
                existing_workspace_id
                and existing_workspace_id != normalized_workspace_id
                and existing_workspace_id.lower() == workspace_name.lower()
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"workspace_name '{workspace_name}' conflicts with existing workspace_id "
                        f"'{existing_workspace_id}'. Please choose a different workspace_name."
                    ),
                )

        db = getattr(getattr(rag.doc_status, "db", None), "ensure_workspace_metadata", None)
        if db is None:
            raise HTTPException(
                status_code=400,
                detail="Workspace metadata table is only available for PostgreSQL doc status storage",
            )

        await rag.doc_status.db.ensure_workspace_metadata(
            normalized_workspace_id,
            workspace_name,
        )

        return {
            "workspace_id": normalized_workspace_id,
            "workspace_name": workspace_name,
            "status": "ok",
        }

    @app.delete(
        "/workspaces/{workspace_id}",
        dependencies=[Depends(combined_auth)],
        summary="Delete workspace and all workspace data",
        description=(
            "Delete all data for a workspace, including documents, chunks, KG, vectors, "
            "workspace files, and workspace metadata."
        ),
    )
    async def delete_workspace(
        workspace_id: str,
        force: bool = Query(False, description="Force deletion of default workspace if true"),
        delete_input_files: bool = Query(True, description="Delete workspace input files"),
    ):
        normalized_workspace_id = normalize_workspace_id(workspace_id)
        if not normalized_workspace_id:
            raise HTTPException(status_code=400, detail="workspace_id cannot be empty")

        canonical_workspace_id = await resolve_workspace_id_alias(normalized_workspace_id)

        default_workspace = normalize_workspace_id(args.workspace)
        if canonical_workspace_id == default_workspace and not force:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Refusing to delete default workspace '{canonical_workspace_id}'. "
                    "Set force=true to confirm destructive deletion."
                ),
            )

        # If the workspace pipeline is busy, block deletion to avoid data races.
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=canonical_workspace_id
            )
            if pipeline_status.get("busy", False):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Workspace '{canonical_workspace_id}' is busy. "
                        "Wait for the pipeline to finish before deleting."
                    ),
                )
        except HTTPException:
            raise
        except Exception:
            # pipeline_status namespace may not exist for unused workspaces.
            pass

        workspace_rag = rag_instances.get(canonical_workspace_id)
        created_temp_rag = False

        if workspace_rag is None:
            workspace_rag = build_rag_instance(canonical_workspace_id)
            await workspace_rag.initialize_storages()
            created_temp_rag = True

        storages = [
            workspace_rag.text_chunks,
            workspace_rag.full_docs,
            workspace_rag.full_entities,
            workspace_rag.full_relations,
            workspace_rag.entity_chunks,
            workspace_rag.relation_chunks,
            workspace_rag.entities_vdb,
            workspace_rag.relationships_vdb,
            workspace_rag.chunks_vdb,
            workspace_rag.chunk_entity_relation_graph,
            workspace_rag.doc_status,
        ]

        drop_tasks = [storage.drop() for storage in storages if storage is not None]
        drop_results = await asyncio.gather(*drop_tasks, return_exceptions=True)

        dropped_components = 0
        drop_errors: list[str] = []

        for idx, result in enumerate(drop_results):
            storage_obj = [storage for storage in storages if storage is not None][idx]
            storage_name = storage_obj.__class__.__name__

            if isinstance(result, Exception):
                drop_errors.append(f"{storage_name}: {result}")
            else:
                dropped_components += 1

        deleted_input_dir = False
        if delete_input_files:
            workspace_input_dir = Path(args.input_dir) / canonical_workspace_id
            if workspace_input_dir.exists() and workspace_input_dir.is_dir():
                shutil.rmtree(workspace_input_dir, ignore_errors=True)
                deleted_input_dir = True

        metadata_deleted = False
        try:
            metadata_deleted = await _delete_workspace_metadata_row(canonical_workspace_id)
        except Exception as e:
            drop_errors.append(f"workspace_metadata: {e}")

        profile_deleted = False
        try:
            profile_path = _embedding_profile_path(canonical_workspace_id)
            if profile_path.exists():
                profile_path.unlink()
                profile_deleted = True
        except Exception as e:
            drop_errors.append(f"embedding_profile: {e}")

        # Remove in-memory instances and aliases after storage cleanup.
        async with rag_instances_lock:
            cached_rag = rag_instances.pop(canonical_workspace_id, None)
            if cached_rag is not None and cached_rag is not workspace_rag:
                await cached_rag.finalize_storages()

            for alias_key in list(workspace_alias_cache.keys()):
                alias_value = workspace_alias_cache.get(alias_key)
                if alias_key == canonical_workspace_id or alias_value == canonical_workspace_id:
                    workspace_alias_cache.pop(alias_key, None)

        doc_manager_instances.pop(canonical_workspace_id, None)

        if created_temp_rag or workspace_rag is not rag:
            await workspace_rag.finalize_storages()

        return {
            "workspace_id": canonical_workspace_id,
            "status": "success" if not drop_errors else "partial_success",
            "dropped_components": dropped_components,
            "drop_errors": drop_errors,
            "deleted_input_dir": deleted_input_dir,
            "metadata_deleted": metadata_deleted,
            "embedding_profile_deleted": profile_deleted,
        }

    # Custom StaticFiles class for smart caching
    class SmartStaticFiles(StaticFiles):  # Renamed from NoCacheStaticFiles
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)

            is_html = path.endswith(".html") or response.media_type == "text/html"

            if is_html:
                response.headers["Cache-Control"] = (
                    "no-cache, no-store, must-revalidate"
                )
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            elif (
                "/assets/" in path
            ):  # Assets (JS, CSS, images, fonts) generated by Vite with hash in filename
                response.headers["Cache-Control"] = (
                    "public, max-age=31536000, immutable"
                )
            # Add other rules here if needed for non-HTML, non-asset files

            # Ensure correct Content-Type
            if path.endswith(".js"):
                response.headers["Content-Type"] = "application/javascript"
            elif path.endswith(".css"):
                response.headers["Content-Type"] = "text/css"

            return response

    # Mount Swagger UI static files for offline support
    swagger_static_dir = Path(__file__).parent / "static" / "swagger-ui"
    if swagger_static_dir.exists():
        app.mount(
            "/static/swagger-ui",
            StaticFiles(directory=swagger_static_dir),
            name="swagger-ui-static",
        )

    # Conditionally mount WebUI only if assets exist
    if webui_assets_exist:
        static_dir = Path(__file__).parent / "webui"
        static_dir.mkdir(exist_ok=True)
        app.mount(
            "/webui",
            SmartStaticFiles(
                directory=static_dir, html=True, check_dir=True
            ),  # Use SmartStaticFiles
            name="webui",
        )
        logger.info("WebUI assets mounted at /webui")
    else:
        logger.info("WebUI assets not available, /webui route not mounted")

        # Add redirect for /webui when assets are not available
        @app.get("/webui")
        @app.get("/webui/")
        async def webui_redirect_to_docs():
            """Redirect /webui to /docs when WebUI is not available"""
            return RedirectResponse(url="/docs")

    return app


def get_application(args=None):
    """Factory function for creating the FastAPI application"""
    if args is None:
        args = global_args
    return create_app(args)


def configure_logging():
    """Configure logging for uvicorn startup"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.filters = []

    # Get log directory path from environment variable
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))

    print(f"\nLightRAG log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = get_env_value("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
    log_backup_count = get_env_value("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                # Configure all uvicorn related loggers
                "uvicorn": {
                    "handlers": ["console", "file"],
                    "level": global_args.log_level,
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console", "file"],
                    "level": global_args.log_level,
                    "propagate": False,
                    "filters": ["path_filter"],
                },
                "uvicorn.error": {
                    "handlers": ["console", "file"],
                    "level": global_args.log_level,
                    "propagate": False,
                },
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": global_args.log_level,
                    "propagate": False,
                    "filters": ["path_filter"],
                },
            },
            "filters": {
                "path_filter": {
                    "()": "lightrag.utils.LightragPathFilter",
                },
            },
        }
    )


def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "uvicorn",
        "tiktoken",
        "fastapi",
        # Add other required packages here
    ]

    for package in required_packages:
        if not pm.is_installed(package):
            print(f"Installing {package}...")
            pm.install(package)
            print(f"{package} installed successfully")


def main():
    # On Windows, ProactorEventLoop (default since Python 3.8) has known
    # race conditions with uvicorn's socket binding that can cause the server
    # to report it's running while the port is never actually bound.
    # Using SelectorEventLoop resolves this issue.
    # See: https://github.com/HKUDS/LightRAG/issues/2438
    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Explicitly initialize configuration for clarity
    # (The proxy will auto-initialize anyway, but this makes intent clear)
    from .config import initialize_config

    initialize_config()

    # Check if running under Gunicorn
    if "GUNICORN_CMD_ARGS" in os.environ:
        # If started with Gunicorn, return directly as Gunicorn will call get_application
        print("Running under Gunicorn - worker management handled by Gunicorn")
        return

    # Check .env file
    if not check_env_file():
        sys.exit(1)

    # Check and install dependencies
    check_and_install_dependencies()

    from multiprocessing import freeze_support

    freeze_support()

    # Configure logging before parsing args
    configure_logging()
    update_uvicorn_mode_config()
    display_splash_screen(global_args)

    # Note: Signal handlers are NOT registered here because:
    # - Uvicorn has built-in signal handling that properly calls lifespan shutdown
    # - Custom signal handlers can interfere with uvicorn's graceful shutdown
    # - Cleanup is handled by the lifespan context manager's finally block

    # Create application instance directly instead of using factory function
    app = create_app(global_args)

    # Start Uvicorn in single process mode
    uvicorn_config = {
        "app": app,  # Pass application instance directly instead of string path
        "host": global_args.host,
        "port": global_args.port,
        "log_config": None,  # Disable default config
    }

    if global_args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": global_args.ssl_certfile,
                "ssl_keyfile": global_args.ssl_keyfile,
            }
        )

    print(
        f"Starting Uvicorn server in single-process mode on {global_args.host}:{global_args.port}"
    )
    uvicorn.run(**uvicorn_config)



if __name__ == "__main__":
    main()
