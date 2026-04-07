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
        print(f"\n--- Đang xử lý Workspace: {ws} ---")
        try:
            # Query tất cả documents cũ để xóa (cơ chế xóa này sẽ dọn dẹp VectorDB, Neon4j, Chunks tự động qua API của C-RAG)
            resp = await client.post(f"{API_URL}/documents/paginated", json={"workspace_id": ws, "page": 1, "page_size": 100})
            if resp.status_code == 200:
                docs = resp.json().get("documents", [])
                doc_ids = [d["id"] for d in docs]
                if doc_ids:
                    print(f"   [{ws}] Đang gửi lệnh xóa {len(doc_ids)} tài liệu (Xóa Neo4j, Vector, SQLite)...")
                    del_payload = {
                        "workspace_id": ws,
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
    print("\n2. Tiến hành đẩy lại các file .md vào Queue và gọi API retry...")
    # Vì file .md nằm trong __enqueued__, ta cần đem nó ngược ra thư mục input để retry
    md_files = glob.glob("data/inputs/*/__enqueued__/*.md")
    if not md_files:
         print("   Không tìm thấy file .md nào trong __enqueued__! Hệ thống chưa có dữ liệu nào chờ retry.")
         return
         
    for pf in md_files:
        workspace_id = pf.split('/')[-3]
        filename = os.path.basename(pf)
        
        print(f"   [{workspace_id}] Nạp file '{filename}' để chạy lại Flow nhúng...")
        # Sử dụng đúng API upload mà hệ thống đã dùng Docling/Deepseek
        try:
            with open(pf, "rb") as f:
                r = await client.post(
                    f"{API_URL}/documents/upload",
                    headers={"LIGHTRAG-WORKSPACE-ID": workspace_id},
                    files={"file": (filename, f, "text/markdown")}
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
