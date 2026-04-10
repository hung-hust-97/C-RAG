import asyncio
import httpx
import glob
import os
import shutil
import subprocess

API_URL = "http://localhost:9621"

async def clear_cache_and_db(client):
    print("1. Xóa LLM Cache, nội dung tài liệu, vector, chunking và KG...")
    try:
        # Gọi API clear_cache (Xóa cache LLM)
        await client.post(f"{API_URL}/documents/clear_cache", json={})
        print("   -> Xóa LLM Cache thành công.")
    except Exception as e:
        print(f"   -> Lỗi khi xóa LLM Cache: {e}")

    # Lấy danh sách workspace có dữ liệu
    input_dirs = glob.glob("data/inputs/*")
    workspaces = [os.path.basename(d) for d in input_dirs if os.path.isdir(d) and not os.path.basename(d).startswith("__")]

    for ws in workspaces:
        api_ws = "" if ws == "default" else ws
        print(f"\n--- Đang xử lý Workspace: {ws} (API_WS: '{api_ws}') ---")
        try:
            # Query tất cả documents cũ để xóa
            resp = await client.post(f"{API_URL}/documents/paginated", json={"workspace_id": api_ws, "page": 1, "page_size": 100})
            if resp.status_code == 200:
                docs = resp.json().get("documents", [])
                doc_ids = [d["id"] for d in docs]
                if doc_ids:
                    print(f"   [{ws}] Đang gửi lệnh xóa {len(doc_ids)} tài liệu (Xóa Neo4j, Vector, SQLite)...")
                    del_payload = {
                        "workspace_id": api_ws,
                        "doc_ids": doc_ids,
                        "delete_file": True,
                        "delete_llm_cache": True
                    }
                    del_resp = await client.request("DELETE", f"{API_URL}/documents/delete_document", json=del_payload)
                    print(f"   [{ws}] Kết quả xóa API: {del_resp.status_code}")
                    await asyncio.sleep(3) # Đợi API chạy background delete
                else:
                    print(f"   [{ws}] Không tìm thấy tài liệu cũ cần xóa trên DB.")
            else:
                print(f"   [{ws}] Server trả về lỗi. Chú ý: Cần tắt bản gốc `make compose-prod-down` và bật bản tuỳ chỉnh `make compose-dev-up` để API delete hoạt động!")
        except Exception as e:
            print(f"   [{ws}] Không gọi được API xóa: {e}")

async def restore_md_and_retry(client):
    print("\n2. Tiến hành đẩy lại các file tài liệu vào Queue và gọi API retry...")
    
    # Lấy danh sách workspace có dữ liệu
    input_dirs = glob.glob("data/inputs/*")
    workspaces_paths = [d for d in input_dirs if os.path.isdir(d) and not os.path.basename(d).startswith("__")]
    
    all_files_to_process = []
    
    # Tìm kiếm tất cả file trong mỗi workspace directory
    for ws_path in workspaces_paths:
        workspace_id = os.path.basename(ws_path)
        # Tìm tất cả file, bao gồm cả trong subfolders như __enqueued__
        for root, dirs, files in os.walk(ws_path):
            for file in files:
                if not file.startswith("."): # Skip hidden files
                    all_files_to_process.append((workspace_id, os.path.join(root, file)))

    # Thêm các file ở root data/inputs (default workspace)
    root_files = [f for f in glob.glob("data/inputs/*") if os.path.isfile(f) and not os.path.basename(f).startswith(".")]
    for rf in root_files:
        all_files_to_process.append(("default", rf))

    if not all_files_to_process:
         print("   Không tìm thấy file nào để xử lý! Hệ thống chưa có dữ liệu.")
         return
         
    for workspace_id, pf in all_files_to_process:
        filename = os.path.basename(pf)
        
        # Determine mime type rudimentarily
        mime_type = "application/octet-stream"
        if filename.endswith(".md"): mime_type = "text/markdown"
        elif filename.endswith(".pdf"): mime_type = "application/pdf"
        
        print(f"   [{workspace_id}] Nạp file '{filename}' để chạy lại Flow nhúng...")
        try:
            with open(pf, "rb") as f:
                r = await client.post(
                    f"{API_URL}/documents/upload",
                    headers={"LIGHTRAG-WORKSPACE-ID": workspace_id},
                    files={"file": (filename, f, mime_type)}
                )
                print(f"   [{workspace_id}] Trạng thái đẩy file API: {r.status_code}")
        except Exception as e:
            print(f"   [{workspace_id}] Thất bại khi kết nối upload API: {e}")

    # Gọi API reprocess_failed (nếu hệ thống hỗ trợ quét lại hàng đợi đọng)
    print("\n3. Gọi API `/documents/reprocess_failed` để đảm bảo hệ thống ưu tiên chạy tiến trình Queue...")
    try:
        r = await client.post(f"{API_URL}/documents/reprocess_failed", json={})
        print(f"   -> Kết quả Retry API: {r.status_code}")
    except Exception as e:
        print(f"   -> Bỏ qua gọi API reprocess_failed (Server không phản hồi/hỗ trợ): {e}")

async def main():
    print("================================================================")
    print(" SCRIPT XÓA CACHE & TÁI TRÍCH XUẤT (RETRY) TỪ FILE MARKDOWN ĐÃ CÓ ")
    print("================================================================")
    print("Lưu ý: Bạn phải chạy server `crag_lite_api` qua lệnh `make compose-dev-up`.")
    print("Đang kêt nối vào C-RAG / LightRAG API tại port 9621...\n")

    async with httpx.AsyncClient(timeout=300.0) as client:
        await clear_cache_and_db(client)
        await restore_md_and_retry(client)
    
    print("\nHOÀN THÀNH. Bạn có thể kiểm tra tab Retrieval hoặc Queue trên Web UI.")

if __name__ == "__main__":
    asyncio.run(main())
