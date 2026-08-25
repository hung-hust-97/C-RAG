from __future__ import annotations

from typing import Any

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import logger


LEGAL_DOCUMENT_STATUS_LABELS = {
    "CHUA_CO_HIEU_LUC": "Chưa có hiệu lực",
    "UNDETERMINED": "Chưa xác định",
    "VALID": "Còn hiệu lực",
    "EXPIRED": "Hết hiệu lực",
    "EXPIRED_TOTALLY": "Hết hiệu lực toàn bộ",
    "NO_LONGER_SUITABLE": "Không còn phù hợp",
    "CEASED_EFFECT": "Ngừng hiệu lực",
    "KNOWN": "Đã biết",
    "UNKNOWN": "Không xác định",
}

LEGAL_DOCUMENT_STATUS_PRIORITY = {
    "VALID": 100,
    "KNOWN": 60,
    "UNDETERMINED": 45,
    "UNKNOWN": 40,
    "CHUA_CO_HIEU_LUC": 30,
    "NO_LONGER_SUITABLE": 20,
    "CEASED_EFFECT": 10,
    "EXPIRED": 5,
    "EXPIRED_TOTALLY": 0,
}

_LABEL_TO_CODE = {
    label.casefold(): code for code, label in LEGAL_DOCUMENT_STATUS_LABELS.items()
}


def normalize_legal_document_status(status: Any) -> dict[str, Any]:
    raw_status = "" if status is None else str(status).strip()
    if not raw_status:
        return {}

    code = raw_status.upper()
    if code not in LEGAL_DOCUMENT_STATUS_LABELS:
        code = _LABEL_TO_CODE.get(raw_status.casefold(), "")

    if not code and raw_status.casefold().startswith("còn hiệu lực đến"):
        code = "VALID"
    elif not code and raw_status.casefold().startswith("có hiệu lực từ"):
        code = "CHUA_CO_HIEU_LUC"
    elif not code and raw_status.casefold().startswith("ngừng hiệu lực"):
        code = "CEASED_EFFECT"
    elif not code and raw_status.casefold().startswith("hết hiệu lực toàn bộ"):
        code = "EXPIRED_TOTALLY"
        
    if not code:
        code = "UNKNOWN"

    label = LEGAL_DOCUMENT_STATUS_LABELS.get(code, raw_status)
    priority = LEGAL_DOCUMENT_STATUS_PRIORITY.get(code, 0)
    return {
        "legal_status": label,
        "legal_status_code": code or raw_status,
        "legal_status_priority": priority,
    }


def best_legal_status_metadata(records: list[dict[str, Any] | None]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    labels = []
    codes = []
    document_ids = []

    for record in records:
        if not isinstance(record, dict):
            continue

        metadata = normalize_legal_document_status(
            record.get("legal_status") or record.get("status")
        )
        if not metadata and record.get("legal_status_code"):
            metadata = normalize_legal_document_status(record.get("legal_status_code"))
        if not metadata:
            continue

        label = metadata["legal_status"]
        code = metadata["legal_status_code"]
        status_labels = [label]
        status_codes = [code]
        if record.get("legal_statuses"):
            status_labels.extend(
                value
                for value in str(record["legal_statuses"]).split(GRAPH_FIELD_SEP)
                if value
            )
        if record.get("legal_status_codes"):
            status_codes.extend(
                value
                for value in str(record["legal_status_codes"]).split(GRAPH_FIELD_SEP)
                if value
            )
        for status_label in status_labels:
            if status_label not in labels:
                labels.append(status_label)
        for status_code in status_codes:
            if status_code not in codes:
                codes.append(status_code)

        legal_document_id = record.get("legal_document_id")
        if legal_document_id and str(legal_document_id) not in document_ids:
            document_ids.append(str(legal_document_id))
        if record.get("legal_document_ids"):
            for doc_id in str(record["legal_document_ids"]).split(GRAPH_FIELD_SEP):
                if doc_id and doc_id not in document_ids:
                    document_ids.append(doc_id)

        if not best or metadata["legal_status_priority"] > best.get(
            "legal_status_priority", -1
        ):
            best = dict(metadata)

    if not best:
        return {}

    best["legal_statuses"] = GRAPH_FIELD_SEP.join(labels)
    best["legal_status_codes"] = GRAPH_FIELD_SEP.join(codes)
    if document_ids:
        best["legal_document_ids"] = GRAPH_FIELD_SEP.join(document_ids)
    return best


def legal_status_sort_priority(legal_info: dict[str, Any] | None) -> int:
    if not isinstance(legal_info, dict):
        return 0
    priority = legal_info.get("legal_status_priority")
    if isinstance(priority, (int, float)):
        return int(priority)
    return normalize_legal_document_status(
        legal_info.get("legal_status") or legal_info.get("legal_status_code")
    ).get("legal_status_priority", 0)


async def load_legal_document_statuses(
    legal_document_ids: list[Any],
) -> dict[str, dict[str, Any]]:
    ids = sorted({str(doc_id) for doc_id in legal_document_ids if doc_id})
    if not ids:
        return {}

    try:
        from lightrag.kg.postgres_impl import ClientManager

        db = await ClientManager.get_client()
        if db is None:
            return {}

        result: dict[str, dict[str, Any]] = {}

        async def _query_db(connection):
            numeric_ids = [int(doc_id) for doc_id in ids if str(doc_id).isdigit()]
            if not numeric_ids:
                return
            rows = await connection.fetch(
                """
                SELECT id, status::text AS status
                FROM tbl_legal_document
                WHERE id = ANY($1::bigint[])
                """,
                numeric_ids,
            )
            for row in rows:
                metadata = normalize_legal_document_status(row["status"])
                if metadata:
                    metadata["legal_document_id"] = str(row["id"])
                    result[str(row["id"])] = metadata

        await db._run_with_retry(_query_db)
        return result
    except Exception as e:
        logger.warning(f"Failed to load legal document statuses: {e}")
        return {}
