from __future__ import annotations

import os
import re
from enum import Enum
from typing import Any

from lightrag.legal_document.status import LEGAL_DOCUMENT_STATUS_LABELS
from lightrag.legal_document.status import normalize_legal_document_status

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility

    class StrEnum(str, Enum):
        pass


class ExtractionMode(StrEnum):
    VECTOR_ONLY = "VECTOR_ONLY"
    KG_RULE_ONLY = "KG_RULE_ONLY"
    KG_LLM = "KG_LLM"
    KG_RULE_ONLY_FALLBACK = "KG_RULE_ONLY_FALLBACK"


VALID_LEGAL_STATUS_CODES = {"VALID", "CHUA_CO_HIEU_LUC", "KNOWN"}
VECTOR_ONLY_PROCESSING_STATUSES = {"DUPLICATED"}

BOILERPLATE_MARKERS = (
    "nơi nhận",
    "lưu:",
    "tm.",
    "kt.",
    "tl.",
    "chánh văn phòng",
    "bộ trưởng",
    "thứ trưởng",
    "chủ tịch",
    "ký thay",
)

LEGAL_SIGNAL_WORDS = (
    "phải",
    "không được",
    "nghiêm cấm",
    "cấm",
    "có trách nhiệm",
    "điều kiện",
    "trình tự",
    "thủ tục",
    "hồ sơ",
    "xử phạt",
    "mức phạt",
    "phạt tiền",
    "nghĩa vụ",
    "quyền",
    "trường hợp",
    "định nghĩa",
    "áp dụng",
    "chịu trách nhiệm",
    "bồi thường",
    "thu hồi",
    "cấp phép",
    "được phép",
    "bị xử lý",
)

RULE_SIGNAL_RE = re.compile(
    r"\b("
    r"điều\s+\d+[a-zA-Z]?|"
    r"khoản\s+\d+|"
    r"điểm\s+[a-z]|"
    r"(luật|nghị định|thông tư|quyết định|nghị quyết|chỉ thị)\s+"
    r"(số\s+)?\d+/"
    r")",
    re.IGNORECASE,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _metadata_value(doc_meta: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = doc_meta.get(key)
        if value not in (None, ""):
            return value
    for nested_key in ("metadata", "context", "legal_info"):
        nested = doc_meta.get(nested_key)
        if isinstance(nested, dict):
            value = _metadata_value(nested, *keys)
            if value not in (None, ""):
                return value
    return None


def legal_extraction_metadata(
    chunk: dict[str, Any] | None,
    doc_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if isinstance(doc_meta, dict):
        metadata.update(doc_meta)
    if isinstance(chunk, dict):
        for nested_key in ("metadata", "context", "legal_info"):
            nested = chunk.get(nested_key)
            if isinstance(nested, dict):
                metadata.update(nested)
        metadata.update(chunk)
    return metadata


def _legal_status_code(value: Any) -> str | None:
    if value in (None, ""):
        return None
    raw_value = str(value).strip()
    upper_value = raw_value.upper()
    if upper_value in LEGAL_DOCUMENT_STATUS_LABELS:
        return upper_value

    normalized = normalize_legal_document_status(raw_value)
    normalized_code = str(normalized.get("legal_status_code") or "").upper()
    if normalized_code in LEGAL_DOCUMENT_STATUS_LABELS:
        return normalized_code
    return None


def _has_inactive_legal_status(doc_meta: dict[str, Any]) -> bool:
    for key in ("legal_status_code", "legal_status", "status"):
        code = _legal_status_code(_metadata_value(doc_meta, key))
        if code is not None:
            return code not in VALID_LEGAL_STATUS_CODES
    return False


def _is_duplicate(doc_meta: dict[str, Any]) -> bool:
    if _as_bool(_metadata_value(doc_meta, "is_duplicated", "duplicated")):
        return True
    if _metadata_value(doc_meta, "duplicate_of", "duplicated_doc_id"):
        return True
    raw_status = str(_metadata_value(doc_meta, "status") or "").upper()
    return raw_status in VECTOR_ONLY_PROCESSING_STATUSES


def _skip_kg_by_metadata(doc_meta: dict[str, Any]) -> bool:
    if _as_bool(
        _metadata_value(
            doc_meta,
            "skip_kg_extraction",
            "skip_graph_extraction",
            "vector_only",
            "disable_kg_extraction",
        )
    ):
        return True

    file_path = str(_metadata_value(doc_meta, "file_path", "document_name") or "")
    return "[-R!]" in file_path or ".[-R!]" in file_path


def _looks_like_boilerplate(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    head = stripped[:800]
    tail = stripped[-1000:]
    if any(marker in head or marker in tail for marker in BOILERPLATE_MARKERS):
        return True

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines and lines[0].startswith(("nơi nhận", "kính gửi")):
        return True
    return False


def _looks_like_table_or_appendix(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped[:120].startswith(("phụ lục", "mẫu số", "biểu mẫu")):
        return True
    line_count = max(1, stripped.count("\n") + 1)
    pipe_count = stripped.count("|")
    return pipe_count >= 12 and pipe_count / line_count >= 2


def classify_extraction_mode(
    doc_meta: dict[str, Any] | None,
    chunk_text: str,
) -> ExtractionMode:
    metadata = legal_extraction_metadata(None, doc_meta)
    text = (chunk_text or "").casefold()
    min_chars = _env_int("LEGAL_EXTRACTION_MIN_LLM_CHARS", 300)

    if _skip_kg_by_metadata(metadata):
        return ExtractionMode.VECTOR_ONLY

    if _has_inactive_legal_status(metadata):
        return ExtractionMode.VECTOR_ONLY

    if _is_duplicate(metadata):
        return ExtractionMode.VECTOR_ONLY

    if len(text.strip()) < min_chars:
        return ExtractionMode.VECTOR_ONLY

    if _looks_like_boilerplate(text):
        return ExtractionMode.VECTOR_ONLY

    if _looks_like_table_or_appendix(text):
        return ExtractionMode.VECTOR_ONLY

    if any(word in text for word in LEGAL_SIGNAL_WORDS):
        return ExtractionMode.KG_LLM

    if RULE_SIGNAL_RE.search(text):
        return ExtractionMode.KG_RULE_ONLY

    return ExtractionMode.KG_RULE_ONLY


def classify_chunk_extraction_mode(
    chunk: dict[str, Any],
    doc_meta: dict[str, Any] | None = None,
) -> ExtractionMode:
    metadata = legal_extraction_metadata(chunk, doc_meta)
    return classify_extraction_mode(metadata, str(chunk.get("content") or ""))
