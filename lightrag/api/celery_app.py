import os
import asyncio
from celery import Celery
import logging
import warnings

logger = logging.getLogger(__name__)

# DEPRECATION WARNING: This module is deprecated and will be removed in a future version.
# Please use celery_worker.app instead:
#   from celery_worker.app import celery_app
warnings.warn(
    "lightrag.api.celery_app is deprecated. Use celery_worker.app instead.",
    DeprecationWarning,
    stacklevel=2
)

# Config from environment variables
CELERY_BROKER_URL = os.environ.get("EMBEDDING_CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("EMBEDDING_CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
CELERY_CONCURRENCY = int(os.environ.get("CELERY_CONCURRENCY", "2"))

celery_app = Celery(
    "lightrag_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["lightrag.api.celery_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=CELERY_CONCURRENCY,
    task_track_started=True,
    worker_prefetch_multiplier=1, # process 1 task at a time per worker to prevent hogging LLM limits
)

logger.info(f"Initialized Celery App with broker {CELERY_BROKER_URL} and concurrency {CELERY_CONCURRENCY}")
