import asyncio
import httpx
import os
import glob
import shutil

API_URL = "http://localhost:9621"

async def main():
    print("WARNING: This script requires 'crag_lite_api' to be running on port 9621.")
    async with httpx.AsyncClient(timeout=300.0) as client:
        # 1. Clear LLM Cache
        print("1. Clearing Global LLM Cache...")
        try:
            r = await client.post(f"{API_URL}/documents/clear_cache", json={})
            print("   Response:", r.status_code)
        except Exception as e:
            print("   Error clearing cache:", e)

        # 2. Get Workspaces & Delete Documents (which wipes vector, chunking, KG, contents)
        workspaces = [d for d in os.listdir("data/inputs") if not d.startswith("__") and os.path.exists(f"data/inputs/{d}")]
        md_files_found = []

        print("\n2. Scanning and Wiping Workspace Documents...")
        for ws in workspaces:
            enqueued_dir = f"data/inputs/{ws}/__enqueued__"
            
            # Find .md files before wiping
            if os.path.exists(enqueued_dir):
                for md_file in glob.glob(f"{enqueued_dir}/*.md"):
                    md_files_found.append((ws, md_file))
            
            # Query paginated documents
            print(f"   [{ws}] Fetching old documents...")
            try:
                r = await client.post(f"{API_URL}/documents/paginated", json={"workspace_id": ws, "page": 1, "page_size": 100})
                if r.status_code == 200:
                    docs = r.json().get("documents", [])
                    doc_ids = [d["id"] for d in docs]
                    if doc_ids:
                        print(f"   [{ws}] Deleting {len(doc_ids)} documents (Wipes vector, chunks & KG)...")
                        await client.request("DELETE", f"{API_URL}/documents/delete_document", json={
                            "workspace_id": ws,
                            "doc_ids": doc_ids,
                            "delete_file": True,
                            "delete_llm_cache": True
                        })
                        await asyncio.sleep(3)
                else:
                    print(f"   [{ws}] Fetch failed: {r.status_code}")
            except Exception as e:
                 print(f"   [{ws}] API unreachable: {e}")

        # 3. Reload .md files to retry extraction!
        print("\n3. Re-uploading .md files to Trigger Extraction Pipeline...")
        if not md_files_found:
            print("   No .md files found to retry.")
        
        for ws, md_file in md_files_found:
            filename = os.path.basename(md_file)
            print(f"   [{ws}] Uploading '{filename}'...")
            try:
                with open(md_file, "rb") as f:
                    r = await client.post(
                        f"{API_URL}/documents/upload",
                        headers={"LIGHTRAG-WORKSPACE-ID": ws},
                        files={"file": (filename, f, "text/markdown")}
                    )
                    print(f"   [{ws}] Upload Status: {r.status_code}")
            except Exception as e:
                print(f"   [{ws}] Failed to upload: {e}")

if __name__ == "__main__":
    asyncio.run(main())
