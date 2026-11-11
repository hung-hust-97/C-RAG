"""Parser and matcher for Vietnamese legal documents with partial expiration (Hết hiệu lực một phần).

Extracts affected articles (Điều) and clauses (Khoản/Điểm) from 'lien_quan_hieu_luc_tung_phan'
and legal relationships to tag chunks accurately during ingestion and retrieval.
"""

from __future__ import annotations

import re
from typing import Any
from bs4 import BeautifulSoup


# Pattern to match Article references (e.g., "Điều 5", "Điều 12a", "Điều 12 bis")
ARTICLE_PATTERN = re.compile(
    r"\bĐiều\s+(\d+[a-zA-Z]?(?:\s+bis|\s+ter)?)\b",
    re.IGNORECASE,
)

# Pattern to match Clause references (e.g., "Khoản 2", "Khoản 1, 2, 3")
CLAUSE_PATTERN = re.compile(
    r"\bKhoản\s+(\d+(?:\s*,\s*\d+)*)\b",
    re.IGNORECASE,
)

# Action keywords signaling invalidation or amendment
EXPIRY_KEYWORDS = [
    "bãi bỏ",
    "hết hiệu lực",
    "bị bãi bỏ",
    "ngưng hiệu lực",
    "ngừng hiệu lực",
    "hủy bỏ",
    "vô hiệu",
]

AMEND_KEYWORDS = [
    "sửa đổi",
    "bổ sung",
    "thay thế",
    "được sửa đổi",
    "được bổ sung",
]


def _clean_html_to_text(raw_html_or_text: str) -> str:
    if not raw_html_or_text or not raw_html_or_text.strip():
        return ""
    try:
        soup = BeautifulSoup(raw_html_or_text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        # Fallback to regex tag stripping
        return re.sub(r"<[^>]+>", " ", raw_html_or_text).strip()


def normalize_article_key(article_str: str) -> str:
    """Normalize 'Điều 5' or 'Điều 5.' to standard 'điều 5'."""
    clean = article_str.strip().strip(":.").casefold()
    m = re.search(r"điều\s+(\d+[a-z]?)", clean)
    if m:
        return f"điều {m.group(1)}"
    return clean


def parse_partial_expiry_info(
    raw_content: str | None,
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Parse 'lien_quan_hieu_luc_tung_phan' HTML/text and relation records.

    Returns:
        dict containing:
            - has_partial_expiry: bool
            - expired_articles: set of normalized article strings (e.g. {'điều 5', 'điều 10'})
            - amended_articles: set of normalized article strings (e.g. {'điều 12'})
            - details: list of structured dicts describing the affected scope
    """
    expired_articles: set[str] = set()
    amended_articles: set[str] = set()
    details: list[dict[str, Any]] = []

    text = _clean_html_to_text(raw_content or "")

    if text:
        # Split text into sentences / paragraphs for contextual matching
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        for para in paragraphs:
            para_lower = para.casefold()

            is_expiry = any(kw in para_lower for kw in EXPIRY_KEYWORDS)
            is_amend = any(kw in para_lower for kw in AMEND_KEYWORDS)

            if not (is_expiry or is_amend):
                continue

            # Extract articles mentioned in this paragraph
            articles = ARTICLE_PATTERN.findall(para)
            for art in articles:
                norm_art = normalize_article_key(f"điều {art}")
                if is_expiry:
                    expired_articles.add(norm_art)
                    details.append({
                        "scope": norm_art,
                        "type": "EXPIRED",
                        "context": para[:200],
                    })
                elif is_amend:
                    amended_articles.add(norm_art)
                    details.append({
                        "scope": norm_art,
                        "type": "AMENDED",
                        "context": para[:200],
                    })

    # Also inspect relations if provided (e.g., relation_type BAI_BO_MOT_PHAN, SUA_DOI_BO_SUNG)
    if relations:
        for rel in relations:
            rel_type = str(rel.get("relation_type", "")).upper()
            scope = rel.get("scope") or rel.get("note") or ""
            if "BAI_BO" in rel_type or "HET_HIEU_LUC" in rel_type:
                for art in ARTICLE_PATTERN.findall(scope):
                    norm_art = normalize_article_key(f"điều {art}")
                    expired_articles.add(norm_art)
            elif "SUA_DOI" in rel_type:
                for art in ARTICLE_PATTERN.findall(scope):
                    norm_art = normalize_article_key(f"điều {art}")
                    amended_articles.add(norm_art)

    has_partial_expiry = bool(expired_articles or amended_articles)
    return {
        "has_partial_expiry": has_partial_expiry,
        "expired_articles": expired_articles,
        "amended_articles": amended_articles,
        "details": details,
    }


def evaluate_chunk_partial_expiry(
    context: dict[str, Any],
    partial_expiry_info: dict[str, Any] | None,
    doc_status_code: str = "",
) -> tuple[bool, str, str | None]:
    """Evaluate whether a chunk belongs to an expired clause/article.

    Args:
        context: Chunk context containing 'article', 'chapter', etc.
        partial_expiry_info: Parsed partial expiry info from parse_partial_expiry_info()
        doc_status_code: Document-level legal status code

    Returns:
        tuple (is_expired: bool, clause_status: str, expiry_warning: str | None)
    """
    if doc_status_code in ("EXPIRED", "EXPIRED_TOTALLY"):
        return True, "EXPIRED", "Toàn bộ văn bản đã hết hiệu lực."

    if not partial_expiry_info or not partial_expiry_info.get("has_partial_expiry"):
        if doc_status_code == "EXPIRED_PARTIALLY":
            # If doc is partially expired but no specific article was extracted, flag caution
            return False, "VALID_WITH_CAUTION", "Văn bản có điều khoản hết hiệu lực một phần."
        return False, "VALID", None

    article_name = str(context.get("article", "")).strip()
    norm_article = normalize_article_key(article_name) if article_name else ""

    if norm_article and norm_article in partial_expiry_info.get("expired_articles", set()):
        warning = f"Điều khoản này ({article_name}) đã HẾT HIỆU LỰC theo quy định sửa đổi, bãi bỏ."
        return True, "EXPIRED", warning

    if norm_article and norm_article in partial_expiry_info.get("amended_articles", set()):
        warning = f"Điều khoản này ({article_name}) đã được SỬA ĐỔI, BỔ SUNG bởi văn bản liên quan."
        return False, "AMENDED", warning

    if doc_status_code == "EXPIRED_PARTIALLY":
        return False, "VALID", None

    return False, "VALID", None
