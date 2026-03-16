📁 Hệ thống KMS: Tài liệu Kiến trúc Multi-tenancy cho RAG
1. Tổng quan chiến lược
Hệ thống chuyển đổi từ đơn nhiệm (Single-user) sang đa nhiệm (Multi-tenant) dựa trên cấp độ: Tenant > Workspace > User.

Nguyên tắc cốt lõi: Dữ liệu tri thức được cô lập tuyệt đối ở cấp Workspace.

Cơ chế: Mọi thực thể (Entity), Quan hệ (Relationship) và Vector Embedding đều phải mang "dấu vân tay" định danh workspace_id.

2. Cấu trúc DB & Quy tắc định danh (Entity Correction)
A. Neo4j (Graph Database)
Node Labels: :Entity (mặc định) kết hợp thuộc tính.

Properties bắt buộc: * workspace_id: ID của Workspace chứa tri thức.

tenant_id: ID của tổ chức sở hữu Workspace.

Quy tắc MERGE: Không dùng name làm định danh duy nhất. Định danh duy nhất phải là bộ đôi (name + workspace_id).

Mẫu Cypher:

Cypher
MERGE (n:Entity {name: $name, workspace_id: $workspace_id})
ON CREATE SET n.type = $type, n.tenant_id = $tenant_id
B. Milvus (Vector Database)
Metadata Schema: Bổ sung trường workspace_id (kiểu dữ liệu VarChar/String).

Search Filter: Luôn áp dụng biểu thức logic expr="workspace_id == 'WS_ID'" trong mọi truy vấn tìm kiếm vector.

C. Minio (Object Storage)
Cấu trúc Folder: bucket/tenants/{tenant_id}/workspaces/{workspace_id}/source_files/.

3. Thay đổi giao diện API (API Contract)
A. API Import / Upload (Python & Java)
Mọi yêu cầu nạp tài liệu phải đính kèm ngữ cảnh quản trị.

Endpoint: POST /api/v1/documents/upload

Payload:

JSON
{
  "file": "binary_data",
  "workspace_id": "string",
  "tenant_id": "string",
  "metadata": { "user_id": "string" }
}
Nhiệm vụ Agent: Truyền workspace_id xuyên suốt từ lúc nhận file đến lúc gọi hàm lightrag.insert().

B. API Retrieval / Get Knowledge
Endpoint: POST /api/v1/query

Payload:

JSON
{
  "query": "Nội dung câu hỏi",
  "workspace_id": "string",
  "mode": "hybrid/global/local"
}
Nhiệm vụ Agent: Ép buộc LightRAG chỉ tìm kiếm trong "vùng chứa" (Namespace) của workspace_id đó.

4. Hướng dẫn chỉnh sửa Logic LightRAG (Dành cho Agent Code)
Bước 1: Sửa lớp khởi tạo lightrag.py
Bổ sung workspace_id vào hàm __init__ để làm biến môi trường nội bộ cho instance đó.

Bước 2: Sửa storage.py (Neo4j & Milvus)
Neo4j: Sửa hàm upsert_node và upsert_edge. Mọi câu lệnh MATCH và MERGE phải bao gồm thuộc tính workspace_id.

Milvus: Sửa hàm upsert để chèn workspace_id vào phần metadata của vector.

Bước 3: Sửa operate.py (Query Logic)
Khi thực hiện duyệt đồ thị (Graph Traversal), phải chèn thêm mệnh đề WHERE vào các câu lệnh Cypher động để lọc đúng workspace_id.
5. Lưu ý về Hiệu suất (Performance)
Indexing: Bắt buộc tạo Composite Index trên Neo4j cho (name, workspace_id).

Partitioning: Sử dụng tính năng Partition Key của Milvus để tăng tốc độ truy vấn đa khách hàng.

Ghi chú cho Agent: Luôn kiểm tra xem workspace_id có tồn tại trong payload trước khi thực hiện bất kỳ thao tác ghi (Write) nào vào DB. Tuyệt đối không để xảy ra tình trạng "Node mồ côi" không có workspace_id.