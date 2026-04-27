
import asyncio
import os
import sys
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.getcwd())

from lightrag.base import DocStatus
from lightrag.api.lightrag_factory import build_rag_instance, LLMConfigCache
from lightrag.api.config import global_args

async def check():
    load_dotenv()
    workspace_id = "e24b3749-1233-495f-ad03-e9d88b647435"
    print(f"Checking workspace: {workspace_id}")
    
    config_cache = LLMConfigCache(global_args)
    rag = build_rag_instance(workspace_id, global_args, config_cache)
    await rag.initialize_storages()
    
    processing_docs, chunking_docs, failed_docs, pending_docs, extracted_docs, extracting_docs = await asyncio.gather(
        rag.doc_status.get_docs_by_status(DocStatus.PROCESSING),
        rag.doc_status.get_docs_by_status(DocStatus.CHUNKING),
        rag.doc_status.get_docs_by_status(DocStatus.FAILED),
        rag.doc_status.get_docs_by_status(DocStatus.PENDING),
        rag.doc_status.get_docs_by_status(DocStatus.EXTRACTED),
        rag.doc_status.get_docs_by_status(DocStatus.EXTRACTING),
    )
    
    print(f"PROCESSING: {len(processing_docs)}")
    print(f"CHUNKING: {len(chunking_docs)}")
    print(f"FAILED: {len(failed_docs)}")
    print(f"PENDING: {len(pending_docs)}")
    print(f"EXTRACTED: {len(extracted_docs)}")
    print(f"EXTRACTING: {len(extracting_docs)}")
    
    if len(chunking_docs) > 0:
        for doc_id, doc in chunking_docs.items():
            print(f"Doc {doc_id} is CHUNKING. Updated at: {doc.updated_at}")
            print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    asyncio.run(check())
