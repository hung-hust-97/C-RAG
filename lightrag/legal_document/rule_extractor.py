from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import date
from typing import Any

from lightrag.legal_document.extraction_mode import legal_extraction_metadata
from lightrag.utils import logger

LEGAL_DOC_TYPES = (
    "Bộ luật",
    "Luật",
    "Pháp lệnh",
    "Nghị định",
    "Thông tư liên tịch",
    "Thông tư",
    "Quyết định",
    "Nghị quyết",
    "Chỉ thị",
)

DOC_REF_RE = re.compile(
    r"\b(?P<doctype>"
    + "|".join(re.escape(doc_type) for doc_type in LEGAL_DOC_TYPES)
    + r")\s+(?:số\s+)?(?P<code>\d{1,4}/[0-9A-ZĐđ\-\/]+)",
    re.IGNORECASE,
)

ARTICLE_RE = re.compile(r"\bĐiều\s+(?P<article>\d+[a-zA-Z]?)\b", re.IGNORECASE)
CLAUSE_RE = re.compile(r"\bkhoản\s+(?P<clause>\d+)\b", re.IGNORECASE)
POINT_RE = re.compile(r"\bđiểm\s+(?P<point>[a-z])\b", re.IGNORECASE)
DATE_RE = re.compile(
    r"ngày\s+(?P<day>\d{1,2})\s+tháng\s+(?P<month>\d{1,2})\s+năm\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
AUTHORITY_RE = re.compile(
    r"\b("
    r"Quốc hội|Ủy ban thường vụ Quốc hội|Chính phủ|Thủ tướng Chính phủ|"
    r"Tòa án nhân dân tối cao|Viện kiểm sát nhân dân tối cao|"
    r"Ngân hàng Nhà nước Việt Nam|"
    r"Bộ\s+(?!luật\b)[A-ZÀ-ỸĐ][A-Za-zÀ-ỹĐđ\s\-]{2,80}|"
    r"Ủy ban nhân dân\s+[A-Za-zÀ-ỹĐđ\s\-]{2,80}|"
    r"Hội đồng nhân dân\s+[A-Za-zÀ-ỹĐđ\s\-]{2,80}"
    r")\b",
    re.IGNORECASE,
)

DOMAIN_KEYWORDS = {
    "doanh nghiệp": "Doanh nghiệp",
    "đầu tư": "Đầu tư",
    "đất đai": "Đất đai",
    "xây dựng": "Xây dựng",
    "lao động": "Lao động",
    "bảo hiểm": "Bảo hiểm",
    "thuế": "Thuế",
    "hải quan": "Hải quan",
    "môi trường": "Môi trường",
    "giao thông": "Giao thông vận tải",
    "giáo dục": "Giáo dục",
    "y tế": "Y tế",
    "ngân hàng": "Ngân hàng",
}

LEGAL_METADATA_KEYS = (
    "legal_status",
    "legal_status_code",
    "legal_status_priority",
    "legal_statuses",
    "legal_status_codes",
    "legal_document_id",
    "legal_document_ids",
    "document_type",
    "document_number",
    "issuing_authority",
    "effective_date",
)


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip(" .,:;")


def _date_to_iso(day: str, month: str, year: str) -> str | None:
    try:
        return date(int(year), int(month), int(day)).isoformat()
    except ValueError:
        return None


def _is_valid_entity_name(name: str) -> bool:
    if len(name) < 2 or len(name) > 220:
        return False
    if not any(ch.isalnum() for ch in name):
        return False
    return True


def _legal_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in LEGAL_METADATA_KEYS
        if key in metadata and metadata[key] not in (None, "")
    }


def _add_entity(
    entities: list[dict[str, Any]],
    seen: set[tuple[str, str]],
    *,
    name: str,
    entity_type: str,
    description: str,
    metadata: dict[str, Any],
) -> None:
    name = _normalize_space(name)
    entity_type = _normalize_space(entity_type).upper()
    if not _is_valid_entity_name(name):
        return
    key = (name.casefold(), entity_type)
    if key in seen:
        return
    seen.add(key)
    entity = {
        "name": name,
        "type": entity_type,
        "description": description,
        "source": "rule",
        "source_method": "rule",
    }
    entity.update(_legal_metadata(metadata))
    entities.append(entity)


def _add_relation(
    relations: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    *,
    source: str,
    target: str,
    relation_type: str,
    description: str,
    metadata: dict[str, Any],
) -> None:
    source = _normalize_space(source)
    target = _normalize_space(target)
    relation_type = _normalize_space(relation_type).upper()
    if not source or not target or source == target:
        return
    key = (source.casefold(), target.casefold(), relation_type)
    if key in seen:
        return
    seen.add(key)
    relation = {
        "source": source,
        "target": target,
        "type": relation_type,
        "description": description,
        "source_method": "rule",
    }
    relation.update(_legal_metadata(metadata))
    relations.append(relation)


def _context_node(metadata: dict[str, Any]) -> str | None:
    context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
    article = context.get("article") or metadata.get("article")
    clauses = context.get("clauses") or metadata.get("clauses")
    points = context.get("points") or metadata.get("points")

    if article:
        node = f"Điều {article}"
        if clauses:
            node = f"{node}, Khoản {clauses}"
        if points:
            node = f"{node}, Điểm {points}"
        return node

    file_path = metadata.get("file_path")
    full_doc_id = metadata.get("full_doc_id")
    if file_path:
        return str(file_path)
    if full_doc_id:
        return str(full_doc_id)
    return None


def _classify_date(match: re.Match[str], text: str) -> tuple[str, str]:
    start = max(0, match.start() - 100)
    end = min(len(text), match.end() + 40)
    window = text[start:end].casefold()
    if "hiệu lực" in window or "áp dụng" in window or "thi hành" in window:
        return "EFFECTIVE_DATE", "EFFECTIVE_ON"
    if "ban hành" in window or "ký" in window or "thông qua" in window:
        return "ISSUE_DATE", "ISSUED_ON"
    return "ISSUE_DATE", "MENTIONS_DATE"


def extract_legal_rule_entities(
    chunk_text: str,
    doc_meta: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    text = chunk_text or ""
    metadata = legal_extraction_metadata(None, doc_meta)
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    seen_entities: set[tuple[str, str]] = set()
    seen_relations: set[tuple[str, str, str]] = set()
    current_node = _context_node(metadata)

    document_names: list[str] = []
    for match in DOC_REF_RE.finditer(text):
        doc_type = _normalize_space(match.group("doctype"))
        code = _normalize_space(match.group("code").upper())
        doc_name = f"{doc_type} {code}"
        document_names.append(doc_name)

        _add_entity(
            entities,
            seen_entities,
            name=doc_name,
            entity_type="LEGAL_DOCUMENT",
            description="Văn bản pháp luật được nhận diện bằng quy tắc từ nội dung.",
            metadata={**metadata, "document_type": doc_type, "document_number": code},
        )
        _add_entity(
            entities,
            seen_entities,
            name=code,
            entity_type="DOCUMENT_CODE",
            description=f"Số hiệu văn bản của {doc_name}.",
            metadata={**metadata, "document_number": code},
        )
        _add_entity(
            entities,
            seen_entities,
            name=f"Viện dẫn {doc_name}",
            entity_type="REFERENCE",
            description=f"Thông tin viện dẫn tới {doc_name}.",
            metadata=metadata,
        )
        _add_relation(
            relations,
            seen_relations,
            source=doc_name,
            target=code,
            relation_type="HAS_CODE",
            description=f"{doc_name} có số hiệu {code}.",
            metadata=metadata,
        )
        _add_relation(
            relations,
            seen_relations,
            source=f"Viện dẫn {doc_name}",
            target=doc_name,
            relation_type="REFERS_TO",
            description=f"Thực thể viện dẫn trỏ tới {doc_name}.",
            metadata=metadata,
        )
        if current_node:
            _add_relation(
                relations,
                seen_relations,
                source=current_node,
                target=doc_name,
                relation_type="CITES",
                description=f"{current_node} viện dẫn {doc_name}.",
                metadata=metadata,
            )

    articles = []
    for match in ARTICLE_RE.finditer(text):
        article_name = f"Điều {match.group('article')}"
        articles.append(article_name)
        _add_entity(
            entities,
            seen_entities,
            name=article_name,
            entity_type="LEGAL_ARTICLE",
            description=f"Cấu trúc điều luật {article_name}.",
            metadata=metadata,
        )

    clauses = []
    for match in CLAUSE_RE.finditer(text):
        clause_name = f"Khoản {match.group('clause')}"
        clauses.append(clause_name)
        _add_entity(
            entities,
            seen_entities,
            name=clause_name,
            entity_type="LEGAL_CLAUSE",
            description=f"Cấu trúc khoản luật {clause_name}.",
            metadata=metadata,
        )

    points = []
    for match in POINT_RE.finditer(text):
        point_name = f"Điểm {match.group('point').lower()}"
        points.append(point_name)
        _add_entity(
            entities,
            seen_entities,
            name=point_name,
            entity_type="LEGAL_POINT",
            description=f"Cấu trúc điểm luật {point_name}.",
            metadata=metadata,
        )

    for article in articles:
        for clause in clauses:
            _add_relation(
                relations,
                seen_relations,
                source=article,
                target=clause,
                relation_type="HAS_CLAUSE",
                description=f"{article} bao gồm {clause}.",
                metadata=metadata,
            )
    for clause in clauses or articles:
        for point in points:
            _add_relation(
                relations,
                seen_relations,
                source=clause,
                target=point,
                relation_type="HAS_POINT",
                description=f"{clause} bao gồm {point}.",
                metadata=metadata,
            )

    authority_names = []
    for match in AUTHORITY_RE.finditer(text):
        authority_name = _normalize_space(match.group(1))
        authority_names.append(authority_name)
        _add_entity(
            entities,
            seen_entities,
            name=authority_name,
            entity_type="LEGAL_AUTHORITY",
            description="Cơ quan hoặc chủ thể có thẩm quyền được nhận diện bằng quy tắc.",
            metadata={**metadata, "issuing_authority": authority_name},
        )

    for authority_name in authority_names:
        for doc_name in document_names:
            _add_relation(
                relations,
                seen_relations,
                source=authority_name,
                target=doc_name,
                relation_type="ISSUES",
                description=f"{authority_name} có liên quan thẩm quyền ban hành {doc_name}.",
                metadata=metadata,
            )

    for match in DATE_RE.finditer(text):
        iso_date = _date_to_iso(
            match.group("day"), match.group("month"), match.group("year")
        )
        if not iso_date:
            continue
        entity_type, relation_type = _classify_date(match, text)
        description = (
            "Ngày hiệu lực được chuẩn hóa theo ISO."
            if entity_type == "EFFECTIVE_DATE"
            else "Ngày ban hành hoặc ngày được đề cập, chuẩn hóa theo ISO."
        )
        _add_entity(
            entities,
            seen_entities,
            name=iso_date,
            entity_type=entity_type,
            description=description,
            metadata={
                **metadata,
                "effective_date": iso_date
                if entity_type == "EFFECTIVE_DATE"
                else metadata.get("effective_date"),
            },
        )
        for doc_name in document_names:
            _add_relation(
                relations,
                seen_relations,
                source=doc_name,
                target=iso_date,
                relation_type=relation_type,
                description=f"{doc_name} liên quan tới ngày {iso_date}.",
                metadata=metadata,
            )

    lowered = text.casefold()
    for keyword, domain_name in DOMAIN_KEYWORDS.items():
        if keyword in lowered:
            _add_entity(
                entities,
                seen_entities,
                name=domain_name,
                entity_type="LEGAL_DOMAIN",
                description=f"Lĩnh vực pháp luật: {domain_name}.",
                metadata=metadata,
            )
            for doc_name in document_names:
                _add_relation(
                    relations,
                    seen_relations,
                    source=doc_name,
                    target=domain_name,
                    relation_type="HAS_DOMAIN",
                    description=f"{doc_name} có nội dung liên quan lĩnh vực {domain_name}.",
                    metadata=metadata,
                )

    return {"entities": entities, "relations": relations}


def extract_legal_rule_kg(
    chunk_id: str,
    chunk: dict[str, Any],
    doc_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    metadata = legal_extraction_metadata(chunk, doc_meta)
    file_path = str(metadata.get("file_path") or "unknown_source")
    timestamp = int(time.time())
    rule_output = extract_legal_rule_entities(str(chunk.get("content") or ""), metadata)

    nodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edges: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for entity in rule_output["entities"]:
        entity_name = entity["name"]
        node = {
            "entity_name": entity_name,
            "entity_type": entity["type"],
            "description": entity["description"],
            "source_id": chunk_id,
            "file_path": file_path,
            "timestamp": timestamp,
            "source_method": "rule",
        }
        node.update(_legal_metadata(entity))
        nodes[entity_name].append(node)

    for relation in rule_output["relations"]:
        src_id = relation["source"]
        tgt_id = relation["target"]
        edge = {
            "src_id": src_id,
            "tgt_id": tgt_id,
            "keywords": relation["type"],
            "description": relation["description"],
            "weight": 1.0,
            "source_id": chunk_id,
            "file_path": file_path,
            "timestamp": timestamp,
            "directed": True,
            "source_method": "rule",
        }
        edge.update(_legal_metadata(relation))
        edges[(src_id, tgt_id)].append(edge)

    logger.debug(
        "[LegalRuleExtractor] chunk_id=%s entities=%s relations=%s",
        chunk_id,
        len(nodes),
        len(edges),
    )
    return dict(nodes), dict(edges)

