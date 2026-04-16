"""
Celery worker module for LightRAG background processing.

This module provides an independent Celery worker implementation for processing
LightRAG document ingestion and query tasks. It is designed to run in a separate
container from the API server, enabling independent scaling and resource allocation.

The module exposes:
- celery_app: The Celery application instance
- tasks: Task definitions for document processing and queries
"""
from celery_worker.app import celery_app
from celery_worker import tasks

__all__ = ["celery_app", "tasks"]
