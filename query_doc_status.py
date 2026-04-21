import asyncio
from lightrag import LightRAG
from lightrag.utils import load_dotenv
import json

async def main():
    print("loading env")
    load_dotenv()
    print("init rag")
    rag = LightRAG(working_dir="./rag_storage")
    print("init storages")
    await rag.initialize_storages()
    
    for status in ["UPLOADING", "EXTRACTING", "EXTRACTED", "CHUNKING", "PROCESSED", "FAILED"]:
        try:
            docs = await rag.doc_status.get_docs_by_status(status)
            print(f"{status}:", len(docs.keys()) if docs else 0)
        except Exception as e:
            print(f"{status} error: {e}")

    await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
