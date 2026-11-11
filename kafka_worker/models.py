from pydantic import BaseModel, ConfigDict, Field
from lightrag.base import DocStatus


class KmsEvent(BaseModel):
    """Cấu trúc dữ liệu Event trao đổi giữa Java KMS và C-RAG qua Kafka."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    document_id: str | None = Field(alias="documentId", default=None)
    legal_document_id: int | None = Field(alias="legalDocumentId", default=None)
    workspace_id: str = Field(alias="workspaceId")
    workspace_name: str | None = Field(alias="workspaceName", default=None)
    document_name: str = Field(alias="documentName")
    status: DocStatus = Field(alias="status")
    message: str | None = Field(alias="message", default="")
    doc_type: str = Field(alias="docType", default="general")
    content_path: str | None = Field(alias="contentPath", default=None)
    content_hash: str | None = Field(alias="contentHash", default=None)
    content_length: int | None = Field(alias="contentLength", default=None)
    content_summary: str | None = Field(alias="contentSummary", default=None)
