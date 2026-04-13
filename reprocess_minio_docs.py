import asyncio
import os
import aioboto3
from lightrag.api.config import global_args
from lightrag.api.lightrag_factory import build_rag_instance, LLMConfigCache

async def main():
    # Setup MinIO credentials
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
    access_key = os.environ.get("S3_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("S3_SECRET_KEY", "minioadmin")
    bucket = os.environ.get("S3_BUCKET", "lightrag-markdown")
    
    print("WARNING: Run this inside the container or install aioboto3 locally.")
    
    # Init RAG instance wrapper to call ainsert
    # Or just use the API!
