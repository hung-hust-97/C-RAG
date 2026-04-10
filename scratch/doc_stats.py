
import asyncio
import asyncpg
import os
from dotenv import load_dotenv

async def main():
    load_dotenv()
    
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    if postgres_host == "host.docker.internal":
        postgres_host = "localhost"
        
    postgres_port = os.getenv("POSTGRES_PORT", "5432")
    postgres_user = os.getenv("POSTGRES_USER", "postgres")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    postgres_database = os.getenv("POSTGRES_DATABASE", "crag")
    
    try:
        conn = await asyncpg.connect(
            host=postgres_host,
            port=postgres_port,
            user=postgres_user,
            password=postgres_password,
            database=postgres_database
        )
        
        query = """
        SELECT workspace, status, count(*) 
        FROM LIGHTRAG_DOC_STATUS 
        GROUP BY workspace, status 
        ORDER BY workspace, status;
        """
        
        rows = await conn.fetch(query)
        
        if not rows:
            print("Không có tài liệu nào trong hệ thống.")
            return

        workspaces = {}
        for row in rows:
            ws = row['workspace'] or "default"
            status = row['status']
            count = row['count']
            
            if ws not in workspaces:
                workspaces[ws] = {"pending": 0, "processing": 0, "done": 0, "error": 0, "total": 0}
            
            if status == "pending":
                workspaces[ws]["pending"] += count
            elif status in ["processing", "preprocessed"]:
                workspaces[ws]["processing"] += count
            elif status == "processed":
                workspaces[ws]["done"] += count
            elif status == "failed":
                workspaces[ws]["error"] += count
            
            workspaces[ws]["total"] += count
            
        print("\nThống kê tài liệu:")
        print("-" * 60)
        for ws, stats in workspaces.items():
            print(f"Workspace: {ws}")
            print(f"  - Đang chờ (Pending): {stats['pending']}")
            print(f"  - Đang xử lý (Processing): {stats['processing']}")
            print(f"  - Đã xong (Done): {stats['done']}")
            print(f"  - Lỗi (Error): {stats['error']}")
            print(f"  - Tổng cộng: {stats['total']}")
            print("-" * 60)
            
        await conn.close()
    except Exception as e:
        print(f"Lỗi khi kết nối database: {e}")

if __name__ == "__main__":
    asyncio.run(main())
