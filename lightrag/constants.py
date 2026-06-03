"""
Centralized configuration constants for LightRAG.

This module defines default values for configuration constants used across
different parts of the LightRAG system. Centralizing these values ensures
consistency and makes maintenance easier.
"""

# Default values for server settings
DEFAULT_WOKERS = 2
DEFAULT_MAX_GRAPH_NODES = 1000

# Default values for extraction settings
DEFAULT_SUMMARY_LANGUAGE = "English"  # Default language for document processing
DEFAULT_MAX_GLEANING = 1
DEFAULT_ENTITY_NAME_MAX_LENGTH = 256

# Number of description fragments to trigger LLM summary
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 8
# Max description token size to trigger LLM summary
DEFAULT_SUMMARY_MAX_TOKENS = 1200
# Recommended LLM summary output length in tokens
DEFAULT_SUMMARY_LENGTH_RECOMMENDED = 600
# Maximum token size sent to LLM for summary
DEFAULT_SUMMARY_CONTEXT_SIZE = 12000
# Maximum token size allowed for entity extraction input context
DEFAULT_MAX_EXTRACT_INPUT_TOKENS = 20480
# Default entities to extract if ENTITY_TYPES is not specified in .env
DEFAULT_ENTITY_TYPES = [
    "Person",
    "Creature",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Method",
    "Content",
    "Data",
    "Artifact",
    "NaturalObject",
]

# Entity types specifically designed for Vietnamese legal documents (Văn bản QPPL)
# Covers subjects, document types, substantive concepts, procedures, and specialized domains
DEFAULT_LEGAL_ENTITY_TYPES = [
    # --- Chủ thể pháp luật (Legal Subjects) ---
    "LegalSubject",              # Chủ thể pháp lý chung (cá nhân, tổ chức, cơ quan)
    "GovernmentAgency",          # Cơ quan nhà nước (Bộ, Sở, UBND, Chính phủ, Quốc hội)
    "Court",                     # Tòa án (TAND tối cao, TAND cấp tỉnh/huyện)
    "Procuracy",                 # Viện Kiểm sát (VKSND tối cao, VKSND cấp tỉnh/huyện)
    "Inspector",                 # Thanh tra (Thanh tra Chính phủ, Thanh tra Bộ, Thanh tra Sở)
    "PublicOfficial",            # Công chức, viên chức, người có thẩm quyền
    "Enterprise",                # Doanh nghiệp (công ty, hộ kinh doanh)
    "Citizen",                   # Công dân, cá nhân, người dân
    "ForeignEntity",             # Tổ chức/cá nhân nước ngoài
    # --- Loại văn bản QPPL (Legal Document Types) ---
    "LegalDocument",             # Văn bản QPPL chung
    "Law",                       # Luật, Bộ luật
    "Ordinance",                 # Pháp lệnh
    "Decree",                    # Nghị định
    "Circular",                  # Thông tư
    "JointCircular",             # Thông tư liên tịch
    "Resolution",                # Nghị quyết (Quốc hội, HĐND, Chính phủ)
    "Decision",                  # Quyết định (Thủ tướng, Bộ trưởng, UBND)
    "Directive",                 # Chỉ thị
    "InternationalTreaty",       # Điều ước quốc tế
    # --- Khái niệm pháp lý cốt lõi (Core Legal Concepts) ---
    "LegalAction",               # Hành vi pháp lý (hành vi vi phạm, hành vi hợp pháp)
    "LegalPenalty",              # Hình phạt, chế tài (phạt tiền, phạt tù, cảnh cáo)
    "LegalRight",                # Quyền (quyền sở hữu, quyền con người, quyền công dân)
    "LegalObligation",           # Nghĩa vụ pháp lý
    "LegalCondition",            # Điều kiện pháp lý (điều kiện kinh doanh, điều kiện cấp phép)
    "LegalConcept",              # Khái niệm/Định nghĩa pháp lý (thuật ngữ chuyên ngành)
    "LegalPrinciple",            # Nguyên tắc pháp lý (nguyên tắc suy đoán vô tội, thiện chí)
    "LegalStatus",               # Tình trạng pháp lý (còn hiệu lực, hết hiệu lực, sửa đổi)
    "Violation",                 # Vi phạm (vi phạm hành chính, vi phạm hình sự)
    "Crime",                     # Tội phạm (tội danh cụ thể trong Bộ luật Hình sự)
    "Sanction",                  # Biện pháp xử lý (biện pháp khắc phục hậu quả, biện pháp ngăn chặn)
    # --- Thủ tục hành chính & Tố tụng (Procedures) ---
    "AdministrativeProcedure",   # Thủ tục hành chính (đăng ký, cấp phép, phê duyệt)
    "LitigationProcedure",       # Thủ tục tố tụng (khởi kiện, xét xử, kháng cáo)
    "ArbitrationProcedure",      # Thủ tục trọng tài
    "License",                   # Giấy phép, chứng chỉ, giấy chứng nhận
    "LegalDeadline",             # Thời hạn, thời hiệu (thời hiệu khởi kiện, thời hạn giải quyết)
    # --- Tài chính, Thuế, Ngân sách (Finance & Tax) ---
    "Tax",                       # Thuế (thuế GTGT, thuế TNDN, thuế TNCN, thuế XNK)
    "Fee",                       # Phí, lệ phí (phí công chứng, lệ phí đăng ký)
    "Budget",                    # Ngân sách nhà nước, quỹ tài chính
    "FinancialInstrument",       # Công cụ tài chính (trái phiếu, cổ phiếu, chứng khoán)
    # --- Đất đai, Bất động sản, Xây dựng (Land & Property) ---
    "LandCategory",              # Loại đất (đất nông nghiệp, đất ở, đất thương mại)
    "LandUseRight",              # Quyền sử dụng đất
    "PropertyRight",             # Quyền sở hữu tài sản, bất động sản
    "ConstructionPermit",        # Giấy phép xây dựng, quy hoạch
    # --- Lao động, Bảo hiểm (Labor & Insurance) ---
    "LaborContract",             # Hợp đồng lao động
    "SocialInsurance",           # Bảo hiểm xã hội (BHXH, BHYT, BHTN)
    "LaborStandard",             # Tiêu chuẩn lao động (lương tối thiểu, giờ làm việc)
    # --- Doanh nghiệp, Thương mại (Corporate & Commerce) ---
    "BusinessLicense",           # Giấy phép kinh doanh, đăng ký doanh nghiệp
    "Contract",                  # Hợp đồng (hợp đồng dân sự, thương mại)
    "IntellectualProperty",      # Sở hữu trí tuệ (bằng sáng chế, nhãn hiệu, bản quyền)
    "Competition",               # Cạnh tranh (chống độc quyền, cạnh tranh không lành mạnh)
    # --- Môi trường, Tài nguyên (Environment & Resources) ---
    "EnvironmentalStandard",     # Tiêu chuẩn/Quy chuẩn môi trường
    "NaturalResource",           # Tài nguyên thiên nhiên (khoáng sản, nước, rừng)
    "EnvironmentalImpact",       # Đánh giá tác động môi trường (ĐTM)
]

# Separator for: description, source_id and relation-key fields(Can not be changed after data inserted)
GRAPH_FIELD_SEP = "<SEP>"

# Query and retrieval configuration defaults
DEFAULT_TOP_K = 40
DEFAULT_CHUNK_TOP_K = 20
DEFAULT_MAX_ENTITY_TOKENS = 6000
DEFAULT_MAX_RELATION_TOKENS = 8000
DEFAULT_MAX_TOTAL_TOKENS = 30000
DEFAULT_COSINE_THRESHOLD = 0.2
DEFAULT_RELATED_CHUNK_NUMBER = 5
DEFAULT_KG_CHUNK_PICK_METHOD = "VECTOR"

# TODO: Deprated. All conversation_history messages is send to LLM.
DEFAULT_HISTORY_TURNS = 0

# Rerank configuration defaults
DEFAULT_MIN_RERANK_SCORE = 0.0
DEFAULT_RERANK_BINDING = "null"

# Default source ids limit in meta data for entity and relation
DEFAULT_MAX_SOURCE_IDS_PER_ENTITY = 300
DEFAULT_MAX_SOURCE_IDS_PER_RELATION = 300
### control chunk_ids limitation method: FIFO, FIFO
###    FIFO: First in first out
###    KEEP: Keep oldest (less merge action and faster)
SOURCE_IDS_LIMIT_METHOD_KEEP = "KEEP"
SOURCE_IDS_LIMIT_METHOD_FIFO = "FIFO"
DEFAULT_SOURCE_IDS_LIMIT_METHOD = SOURCE_IDS_LIMIT_METHOD_FIFO
VALID_SOURCE_IDS_LIMIT_METHODS = {
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
}
# Maximum number of file paths stored in entity/relation file_path field (For displayed only, does not affect query performance)
DEFAULT_MAX_FILE_PATHS = 100

# Field length of file_path in Milvus Schema for entity and relation (Should not be changed)
# file_path must store all file paths up to the DEFAULT_MAX_FILE_PATHS limit within the metadata.
DEFAULT_MAX_FILE_PATH_LENGTH = 32768
# Placeholder for more file paths in meta data for entity and relation (Should not be changed)
DEFAULT_FILE_PATH_MORE_PLACEHOLDER = "truncated"

# Default temperature for LLM
DEFAULT_TEMPERATURE = 1.0

# Async configuration defaults
DEFAULT_MAX_ASYNC = 4  # Default maximum async operations
DEFAULT_MAX_PARALLEL_INSERT = 2  # Default maximum parallel insert operations

# Embedding configuration defaults
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 8  # Default max async for embedding functions
DEFAULT_EMBEDDING_BATCH_NUM = 10  # Default batch size for embedding computations

# Gunicorn worker timeout
DEFAULT_TIMEOUT = 300

# Default llm and embedding timeout
DEFAULT_LLM_TIMEOUT = 180
DEFAULT_EMBEDDING_TIMEOUT = 30

# Logging configuration defaults
DEFAULT_LOG_MAX_BYTES = 10485760  # Default 10MB
DEFAULT_LOG_BACKUP_COUNT = 5  # Default 5 backups
DEFAULT_LOG_FILENAME = "lightrag.log"  # Default log filename

# Ollama server configuration defaults
DEFAULT_OLLAMA_MODEL_NAME = "lightrag"
DEFAULT_OLLAMA_MODEL_TAG = "latest"
DEFAULT_OLLAMA_MODEL_SIZE = 7365960935
DEFAULT_OLLAMA_CREATED_AT = "2024-01-15T00:00:00Z"
DEFAULT_OLLAMA_DIGEST = "sha256:lightrag"
