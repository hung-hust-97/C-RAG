import os
import socket

# Thông tin kết nối Kafka broker
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID")

# Tên các Topic
KMS_EXTRACTING_TOPIC = os.getenv("KMS_EXTRACTING_TOPIC", "kms.doc.extracting")
KMS_EXTRACTED_TOPIC = os.getenv("KMS_EXTRACTED_TOPIC", "kms.doc.extracted")
KMS_CHUNKING_TOPIC = os.getenv("KMS_CHUNKING_TOPIC", "kms.doc.chunking")
KMS_PROCESSED_TOPIC = os.getenv("KMS_PROCESSED_TOPIC", "kms.doc.processed")
KMS_FAILED_TOPIC = os.getenv("KMS_FAILED_TOPIC", "kms.doc.failed")

KMS_EXTRACTING_LEGAL_TOPIC = os.getenv("KMS_EXTRACTING_LEGAL_TOPIC", "kms.doc.legal.extracting")
KMS_CHUNKING_LEGAL_TOPIC = os.getenv("KMS_CHUNKING_LEGAL_TOPIC", "kms.doc.legal.chunking")
KMS_EXTRACTED_LEGAL_TOPIC = os.getenv("KMS_EXTRACTED_LEGAL_TOPIC", "kms.doc.legal.extracted")
KMS_PROCESSED_LEGAL_TOPIC = os.getenv("KMS_PROCESSED_LEGAL_TOPIC", "kms.doc.legal.processed")
KMS_FAILED_LEGAL_TOPIC = os.getenv("KMS_FAILED_LEGAL_TOPIC", "kms.doc.legal.failed")

# --- General pipeline topics ---
PIPELINE_GENERAL_EXTRACTION_TOPIC = os.getenv("PIPELINE_GENERAL_EXTRACTION_TOPIC", "lightrag.pipeline.general.extraction")
PIPELINE_GENERAL_CHUNKING_TOPIC = os.getenv("PIPELINE_GENERAL_CHUNKING_TOPIC", "lightrag.pipeline.general.chunking")
PIPELINE_GENERAL_CHUNK_EMBEDDING_TOPIC = os.getenv("PIPELINE_GENERAL_CHUNK_EMBEDDING_TOPIC", "lightrag.pipeline.general.chunk_embedding")
PIPELINE_GENERAL_ENTITY_EXTRACTION_TOPIC = os.getenv("PIPELINE_GENERAL_ENTITY_EXTRACTION_TOPIC", "lightrag.pipeline.general.entity_extraction")
PIPELINE_GENERAL_KG_MERGE_TOPIC = os.getenv("PIPELINE_GENERAL_KG_MERGE_TOPIC", "lightrag.pipeline.general.kg_merge")
PIPELINE_GENERAL_KG_EMBEDDING_TOPIC = os.getenv("PIPELINE_GENERAL_KG_EMBEDDING_TOPIC", "lightrag.pipeline.general.kg_embedding")
PIPELINE_GENERAL_FINALIZE_TOPIC = os.getenv("PIPELINE_GENERAL_FINALIZE_TOPIC", "lightrag.pipeline.general.finalize")

# --- Legal pipeline topics ---
PIPELINE_LEGAL_EXTRACTION_TOPIC = os.getenv("PIPELINE_LEGAL_EXTRACTION_TOPIC", "lightrag.pipeline.legal.extraction")
PIPELINE_LEGAL_CHUNKING_TOPIC = os.getenv("PIPELINE_LEGAL_CHUNKING_TOPIC", "lightrag.pipeline.legal.chunking")
PIPELINE_LEGAL_CHUNK_EMBEDDING_TOPIC = os.getenv("PIPELINE_LEGAL_CHUNK_EMBEDDING_TOPIC", "lightrag.pipeline.legal.chunk_embedding")
PIPELINE_LEGAL_ENTITY_EXTRACTION_TOPIC = os.getenv("PIPELINE_LEGAL_ENTITY_EXTRACTION_TOPIC", "lightrag.pipeline.legal.entity_extraction")
PIPELINE_LEGAL_KG_MERGE_TOPIC = os.getenv("PIPELINE_LEGAL_KG_MERGE_TOPIC", "lightrag.pipeline.legal.kg_merge")
PIPELINE_LEGAL_KG_EMBEDDING_TOPIC = os.getenv("PIPELINE_LEGAL_KG_EMBEDDING_TOPIC", "lightrag.pipeline.legal.kg_embedding")
PIPELINE_LEGAL_FINALIZE_TOPIC = os.getenv("PIPELINE_LEGAL_FINALIZE_TOPIC", "lightrag.pipeline.legal.finalize")

PIPELINE_FAILED_TOPIC = os.getenv(
    "PIPELINE_FAILED_TOPIC", "lightrag.pipeline.failed"
)

PIPELINE_CONSUMER_GROUP_PREFIX = os.getenv(
    "PIPELINE_CONSUMER_GROUP_PREFIX", f"{KAFKA_GROUP_ID}_pipeline"
)
PIPELINE_WORKER_ID = os.getenv(
    "PIPELINE_WORKER_ID", f"{socket.gethostname()}:{os.getpid()}"
)
PIPELINE_DEBUG_SYNC = os.getenv("PIPELINE_DEBUG_SYNC", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Số luồng xử lý đồng thời
MAX_CONCURRENT_EXTRACTIONS = int(os.getenv("MAX_CONCURRENT_EXTRACTIONS", "3"))
MAX_CONCURRENT_CHUNKING = int(os.getenv("MAX_CONCURRENT_CHUNKING", "5"))

# Các loại file
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".mdx",
    ".html",
    ".htm",
    ".tex",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".log",
    ".conf",
    ".ini",
    ".properties",
    ".sql",
    ".bat",
    ".sh",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".py",
    ".java",
    ".js",
    ".ts",
    ".swift",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".css",
    ".scss",
    ".less",
}

OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}

BINARY_EXTENSIONS = {
    ".rtf",
    ".odt",
    ".epub",
    ".csv",
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tiff",
    ".bmp",
}
