import asyncio
import httpx
import os
import sys

API_URL = "http://localhost:9621"

async def main():
    print("🚀 Script: Reprocess FAILED Documents via API")
    print("   This script fetches FAILED documents, clears their stuck records,")
    print("   and re-uploads the original files so they go through the updated OCR flow")
    print("   (which now natively outputs .md and saves to MinIO).")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Check workspaces locally or through API
        if not os.path.exists("data/inputs"):
            print("❌ No data/inputs directory found. Please run this script in the C-RAG root directory.")
            return

        workspaces = [d for d in os.listdir("data/inputs") if not d.startswith("__") and os.path.isdir(f"data/inputs/{d}")]
        
        for ws in workspaces:
            print(f"\n📂 Workspace: [{ws}] - Fetching failed documents...")
            try:
                # 1. Fetch failing docs
                r = await client.post(f"{API_URL}/documents/paginated", json={
                    "workspace_id": ws.strip('",'), 
                    "page": 1, 
                    "page_size": 200,
                    "status_filter": "failed"
                })
                if r.status_code != 200:
                    print(f"   ❌ Fetch failed: {r.status_code}, {r.text}")
                    continue
                
                docs = r.json().get("documents", [])
                if not docs:
                    print(f"   ✅ No failed documents found in this workspace.")
                    continue
                
                doc_ids = [d["id"] for d in docs]
                # We need the full disk path to the original files to re-upload them
                file_paths = [d["file_path"] for d in docs if d["file_path"] and d["file_path"] != "unknown_source"]
                
                valid_files = []
                for fp in file_paths:
                    full_path = os.path.join("data", "inputs", ws, fp)
                    if os.path.exists(full_path):
                        valid_files.append((fp, full_path))
                    else:
                        print(f"   ⚠️ Original file missing from disk: {full_path} (Skipping...)")
                
                # 2. Delete the failed entries using the API so they don't block the new ingestion
                if doc_ids:
                    print(f"   🗑️ Deleting {len(doc_ids)} stuck FAILED records...")
                    delete_r = await client.request("DELETE", f"{API_URL}/documents/delete_document", json={
                        "workspace_id": ws.strip('",'),
                        "doc_ids": doc_ids,
                        "delete_file": False, # Preserve the local disk file! We need it for re-uploading
                        "delete_llm_cache": False
                    })
                    if delete_r.status_code != 200:
                         print(f"   ❌ Failed to delete old records: {delete_r.text}")
                    await asyncio.sleep(1)

                # 3. Re-upload the files
                print(f"   🔄 Re-queueing {len(valid_files)} documents into Hybrid Flow...")
                for filename, full_path in valid_files:
                    try:
                        # Determine roughly the mime type
                        ext = os.path.splitext(filename)[1].lower()
                        mime = "application/octet-stream"
                        if ext == ".docx":
                            mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        elif ext in [".jpg", ".jpeg"]:
                            mime = "image/jpeg"
                        elif ext == ".png":
                            mime = "image/png"
                        elif ext == ".pdf":
                            mime = "application/pdf"
                        elif ext == ".txt":
                            mime = "text/plain"
                        
                        with open(full_path, "rb") as f:
                            upload_r = await client.post(
                                f"{API_URL}/documents/upload",
                                headers={"LIGHTRAG-WORKSPACE-ID": ws.strip('",')},
                                files={"file": (filename, f, mime)}
                            )
                            if upload_r.status_code == 200:
                                print(f"      ✅ [{filename}] Uploaded! -> Routing to Docling/DeepSeek OCR -> saving .md to MinIO.")
                            else:
                                print(f"      ❌ [{filename}] Upload failed: {upload_r.status_code} - {upload_r.text}")
                    except Exception as file_e:
                        print(f"      ❌ [{filename}] Error during upload: {file_e}")

            except Exception as e:
                 print(f"   ❌ API unreachable or unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
