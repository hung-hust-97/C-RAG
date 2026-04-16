"""
Celery application instance for LightRAG.
"""
import logging
from celery import Celery
from celery_worker.config import Config

logger = logging.getLogger(__name__)

config = Config()

celery_app = Celery(
    "lightrag_tasks",
    broker=config.BROKER_URL,
    backend=config.RESULT_BACKEND,
    include=["celery_worker.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=config.CONCURRENCY,
    task_track_started=True,
    worker_prefetch_multiplier=1,
)

logger.info(
    f"Initialized Celery App with broker {config.BROKER_URL} "
    f"and concurrency {config.CONCURRENCY}"
)
