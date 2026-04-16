"""
Configuration for Celery worker.
"""
import os


class Config:
    """Celery worker configuration from environment variables."""
    
    BROKER_URL = os.environ.get(
        "EMBEDDING_CELERY_BROKER_URL",
        "redis://localhost:6379/0"
    )
    
    RESULT_BACKEND = os.environ.get(
        "EMBEDDING_CELERY_RESULT_BACKEND",
        "redis://localhost:6379/1"
    )
    
    CONCURRENCY = int(os.environ.get("CELERY_CONCURRENCY", "2"))
    
    # Storage configuration (shared with API)
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
    MILVUS_URI = os.environ.get("MILVUS_URI", "http://milvus:19530")
    
    WORKING_DIR = os.environ.get("WORKING_DIR", "/app/data/rag_storage")
    INPUT_DIR = os.environ.get("INPUT_DIR", "/app/data/inputs")
    LOG_DIR = os.environ.get("LIGHTRAG_LOG_DIR", "/app/data/logs")
