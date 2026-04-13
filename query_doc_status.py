import asyncio
from lightrag import LightRAG
from lightrag.utils import load_env
import json

async def main():
    print("loading env")
    load_env()
    print("init rag")
    rag = LightRAG(working_dir="./rag_storage")
    print("init storages")
    await rag.initialize_storages()
    print("getting processing docs")
    processing = await rag.doc_status.get_docs_by_status("processing")
    print("PROCESSING:", len(processing.keys()))
    print("getting pending docs")
    pending = await rag.doc_status.get_docs_by_status("pending")
    print("PENDING:", len(pending.keys()))

    await rag.finalize_storages()
    print("done")

if __name__ == "__main__":
    asyncio.run(main())
