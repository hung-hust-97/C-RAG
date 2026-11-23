from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup, Comment

from lightrag.utils import Tokenizer, logger

# Presentational markup (Word/TVPL inline styles, CSS, comments) can make up most
# of a legal HTML document's tokens. Structure tags and the id/name anchors used
# for citations and article lookup are kept.
_HTML_DROP_TAGS = ("script", "style", "meta", "link", "title", "head", "xml")
_HTML_UNWRAP_TAGS = ("span", "font")
_HTML_KEEP_ATTRS = frozenset({"id", "name", "colspan", "rowspan"})
_HTML_TAG_RE = re.compile(r"<(?:p|table|div|br|html|body|tr|td)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_CHUNK_DAN_CHIEU_CHARS = 4000


def looks_like_html(content: str | None) -> bool:
    """Return True when legal content is HTML rather than markdown/plain text."""
    return bool(content) and _HTML_TAG_RE.search(content[:20000]) is not None


def strip_presentational_markup(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove styling noise in place, keeping structure tags and anchors."""
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for tag in soup.find_all(_HTML_DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in _HTML_KEEP_ATTRS}
    for tag in soup.find_all(True):
        if tag.attrs:
            continue
        # Word namespaced tags (o:p, st1:...) and bare inline wrappers carry no
        # structure once attributes are gone.
        if tag.name in _HTML_UNWRAP_TAGS or tag.name == "a" or ":" in tag.name:
            tag.unwrap()
    for text in soup.find_all(string=True):
        collapsed = _WHITESPACE_RE.sub(" ", text)
        if collapsed != text:
            text.replace_with(collapsed)
    return soup

_HTML_PART_ID_RE = re.compile(r"^phan_([a-z0-9]+)(_name)?$", re.IGNORECASE)
_HTML_CHAPTER_ID_RE = re.compile(r"^chuong_([ivxlcdm0-9]+)(_name)?$", re.IGNORECASE)
_HTML_SECTION_ID_RE = re.compile(
    r"^(?:tieu_?muc|muc)_([ivxlcdm0-9]+)(_name)?$", re.IGNORECASE
)
_HTML_ARTICLE_ID_RE = re.compile(
    r"^dieu_(\d+[a-z]?(?:_\d+)?)(_name)?$", re.IGNORECASE
)

_TEXT_PART_RE = re.compile(
    r"^Phần\s+(?:thứ\s+)?([IVXLCDM0-9a-zA-ZÀ-ỹ]+)(?:[.:\s].*)?$",
    re.IGNORECASE | re.DOTALL,
)
_TEXT_CHAPTER_RE = re.compile(
    r"^Chương\s+([IVXLCDM0-9]+)(?:[.:\s].*)?$", re.IGNORECASE | re.DOTALL
)
_TEXT_SECTION_RE = re.compile(
    r"^(?:Tiểu\s+mục|Mục)\s+([IVXLCDM0-9]+)(?:[.:\s].*)?$",
    re.IGNORECASE | re.DOTALL,
)
_TEXT_ARTICLE_RE = re.compile(
    r"^Điều\s+(\d+[a-z]?)(?:[.:\s].*)?$", re.IGNORECASE | re.DOTALL
)


def _normalize_heading_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


def _merge_heading_name(base_text: str, name_text: str) -> str:
    base = _normalize_heading_text(base_text)
    name = _normalize_heading_text(name_text)
    if not name:
        return base
    if not base:
        return name
    if name.lower() in base.lower():
        return base
    if base.endswith((".", ":", "-", "–", "—")):
        return f"{base} {name}"
    return f"{base}. {name}"


def build_parent_section(
    *,
    part: str = "",
    chapter: str = "",
    section: str = "",
    article: str = "",
) -> str | None:
    """Build normalized structural path joined by ' > ' up to Article level."""
    parts = [
        _normalize_heading_text(value)
        for value in (part, chapter, section, article)
    ]
    parts = [p for p in parts if p]
    if not parts:
        return None
    return " > ".join(parts)


def _collect_element_anchor_ids(block: Any) -> list[str]:
    """Collect candidate HTML id/name attributes on block or nested <a> tags."""
    candidates: list[str] = []
    if block is None or not hasattr(block, "get"):
        return candidates

    for attr in ("id", "name"):
        val = block.get(attr)
        if val:
            s = str(val).strip()
            if s and s not in candidates:
                candidates.append(s)

    if hasattr(block, "find_all"):
        for anchor in block.find_all("a"):
            for attr in ("id", "name"):
                val = anchor.get(attr)
                if val:
                    s = str(val).strip()
                    if s and s not in candidates:
                        candidates.append(s)

    return candidates


def extract_html_structure_marker(
    block: Any,
) -> tuple[str | None, str, bool, str]:
    """Inspect HTML id/name attributes for legal structural boundaries.

    Returns:
        Tuple of (level, index, is_name_block, raw_id) where level is one of
        'part', 'chapter', 'section', 'article', or None if no structural marker.
    """
    for raw_id in _collect_element_anchor_ids(block):
        for level, pattern in (
            ("article", _HTML_ARTICLE_ID_RE),
            ("chapter", _HTML_CHAPTER_ID_RE),
            ("section", _HTML_SECTION_ID_RE),
            ("part", _HTML_PART_ID_RE),
        ):
            match = pattern.match(raw_id)
            if match:
                return level, match.group(1), bool(match.group(2)), raw_id
    return None, "", False, ""


def _extract_html_blocks(soup: BeautifulSoup) -> list[Any]:
    blocks = []
    # Some sources put article headings ("Điều 1. ...") in h1-h6 instead of <p>.
    for element in soup.find_all(["p", "table", "h1", "h2", "h3", "h4", "h5", "h6"]):
        if element.name != "table" and element.find_parent("table"):
            continue
        blocks.append(element)
    return blocks


def build_html_structure_index(html_content: str) -> dict[str, Any]:
    """Build lookup maps from HTML anchors and article numbers to parent_section.

    Uses the exact same document-level hybrid detection as ``legal_chunking_html``.
    """
    soup = BeautifulSoup(html_content or "", "html.parser")
    blocks = _extract_html_blocks(soup)

    has_part_ids = False
    has_chapter_ids = False
    has_section_ids = False
    has_article_ids = False
    for block in blocks:
        level, _, is_name, _ = extract_html_structure_marker(block)
        if is_name or not level:
            continue
        if level == "part":
            has_part_ids = True
        elif level == "chapter":
            has_chapter_ids = True
        elif level == "section":
            has_section_ids = True
        elif level == "article":
            has_article_ids = True

    current_part = ""
    current_chapter = ""
    current_section = ""
    current_article = ""
    current_article_num = ""

    id_to_parent_section: dict[str, str] = {}
    article_num_to_parent_section: dict[str, str] = {}
    current_article_anchor_ids: list[str] = []

    def _record_current_anchors(anchor_ids: list[str], parent_sec: str | None) -> None:
        if not parent_sec:
            return
        for aid in anchor_ids:
            id_to_parent_section[aid.lower()] = parent_sec

    for block in blocks:
        text = _normalize_heading_text(block.get_text(" ", strip=True))
        level, idx, is_name, _ = extract_html_structure_marker(block)
        anchor_ids = _collect_element_anchor_ids(block)

        if is_name:
            if level == "part" and current_part:
                current_part = _merge_heading_name(current_part, text)
            elif level == "chapter" and current_chapter:
                current_chapter = _merge_heading_name(current_chapter, text)
            elif level == "section" and current_section:
                current_section = _merge_heading_name(current_section, text)
            elif level == "article" and current_article:
                current_article = _merge_heading_name(current_article, text)
                parent_sec = build_parent_section(
                    part=current_part,
                    chapter=current_chapter,
                    section=current_section,
                    article=current_article,
                )
                if parent_sec:
                    _record_current_anchors(current_article_anchor_ids, parent_sec)
                    _record_current_anchors(anchor_ids, parent_sec)
                    if current_article_num:
                        article_num_to_parent_section[current_article_num] = parent_sec
                continue
            parent_sec = build_parent_section(
                part=current_part,
                chapter=current_chapter,
                section=current_section,
                article=current_article,
            )
            _record_current_anchors(anchor_ids, parent_sec)
            continue

        is_part = (level == "part") or (
            not has_part_ids and not level and bool(_TEXT_PART_RE.match(text))
        )
        is_chapter = (level == "chapter") or (
            not has_chapter_ids and not level and bool(_TEXT_CHAPTER_RE.match(text))
        )
        is_section = (level == "section") or (
            not has_section_ids and not level and bool(_TEXT_SECTION_RE.match(text))
        )
        is_article = (level == "article") or (
            not has_article_ids and not level and bool(_TEXT_ARTICLE_RE.match(text))
        )

        if is_part:
            current_part = text
            current_chapter = ""
            current_section = ""
            current_article = ""
            current_article_num = ""
            current_article_anchor_ids = []
        elif is_chapter:
            current_chapter = text
            current_section = ""
            current_article = ""
            current_article_num = ""
            current_article_anchor_ids = []
        elif is_section:
            current_section = text
            current_article = ""
            current_article_num = ""
            current_article_anchor_ids = []
        elif is_article:
            current_article = text
            if level == "article" and idx:
                current_article_num = idx.split("_", 1)[0].lower()
            else:
                m = _TEXT_ARTICLE_RE.match(text)
                current_article_num = m.group(1).lower() if m else ""
            current_article_anchor_ids = list(anchor_ids)

        parent_sec = build_parent_section(
            part=current_part,
            chapter=current_chapter,
            section=current_section,
            article=current_article,
        )
        _record_current_anchors(anchor_ids, parent_sec)
        if is_article and parent_sec and current_article_num:
            article_num_to_parent_section.setdefault(current_article_num, parent_sec)

    return {
        "id_to_parent_section": id_to_parent_section,
        "article_num_to_parent_section": article_num_to_parent_section,
        "has_article_ids": has_article_ids,
    }


def _validate_chunking_limits(
    chunk_token_size: int, chunk_overlap_token_size: int
) -> None:
    if chunk_token_size <= 0:
        raise ValueError("chunk_token_size must be greater than zero")
    if not 0 <= chunk_overlap_token_size < chunk_token_size:
        raise ValueError(
            "chunk_overlap_token_size must be non-negative and smaller "
            "than chunk_token_size"
        )


def _split_oversized_content(
    tokenizer: Tokenizer,
    content: str,
    context_prefix: str,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
) -> list[tuple[str, int, int]]:
    source_tokens = tokenizer.encode(content)
    if not source_tokens:
        return []

    # A misdetected heading can make the prefix longer than the chunk itself;
    # cap it so the content always keeps most of the token budget.
    max_prefix_tokens = max(1, chunk_token_size // 4)
    prefix_tokens = tokenizer.encode(context_prefix)
    if len(prefix_tokens) > max_prefix_tokens:
        context_prefix = (
            tokenizer.decode(prefix_tokens[:max_prefix_tokens]).rstrip() + "…\n\n"
        )

    chunks = []
    start = 0
    while start < len(source_tokens):
        end = min(start + chunk_token_size, len(source_tokens))
        while end > start:
            chunk_content = (
                f"{context_prefix}{tokenizer.decode(source_tokens[start:end])}"
            ).strip()
            if len(tokenizer.encode(chunk_content)) <= chunk_token_size:
                break
            end -= 1

        if end == start:
            raise ValueError(
                "The legal context prefix leaves no room for content within "
                f"the {chunk_token_size}-token chunk limit"
            )

        chunks.append((chunk_content, start, end))
        if end == len(source_tokens):
            break
        start = max(start + 1, end - chunk_overlap_token_size)

    return chunks


def _enrich_chunks_with_partial_expiry(
    chunks: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Enrich chunks with partial expiry evaluation if metadata is available."""
    partial_expiry_info = kwargs.get("partial_expiry_info")
    if partial_expiry_info is None and kwargs.get("lien_quan_hieu_luc_tung_phan"):
        from lightrag.legal_document.partial_expiry_parser import parse_partial_expiry_info
        partial_expiry_info = parse_partial_expiry_info(
            kwargs.get("lien_quan_hieu_luc_tung_phan"),
            kwargs.get("relations"),
        )

    doc_status_code = kwargs.get("legal_status_code") or ""
    if not doc_status_code and kwargs.get("legal_status"):
        from lightrag.legal_document.status import normalize_legal_document_status
        doc_status_code = normalize_legal_document_status(
            kwargs.get("legal_status")
        ).get("legal_status_code", "")

    if partial_expiry_info or doc_status_code:
        from lightrag.legal_document.partial_expiry_parser import evaluate_chunk_partial_expiry
        for chunk in chunks:
            is_expired, clause_status, expiry_warning = evaluate_chunk_partial_expiry(
                chunk.get("context", {}),
                partial_expiry_info,
                doc_status_code=doc_status_code,
            )
            chunk["is_expired"] = is_expired
            chunk["clause_status"] = clause_status
            if expiry_warning:
                chunk["expiry_warning"] = expiry_warning

    # lien_quan_dan_chieu is document-level HTML kept in tbl_legal_document and can
    # reach several MB. Every chunk carries a copy, so only small values are kept,
    # and only in context (the field Milvus stores and the query path reads).
    lien_quan_dan_chieu = kwargs.get("lien_quan_dan_chieu")
    if lien_quan_dan_chieu and len(lien_quan_dan_chieu) <= _MAX_CHUNK_DAN_CHIEU_CHARS:
        for chunk in chunks:
            if "context" in chunk and isinstance(chunk["context"], dict):
                chunk["context"]["lien_quan_dan_chieu"] = lien_quan_dan_chieu

    return chunks


def legal_chunking_markdown(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
    **kwargs,
) -> list[dict[str, Any]]:
    """Chunking for Vietnamese legal documents with clause-aware splitting.

    This function splits legal documents by article boundaries (Điều) while preserving
    chapter context (Chương). When an article exceeds token limits, it intelligently
    splits by clause boundaries (Khoản) and groups clauses to fit within token limits.

    Clause/point detection patterns:
    - Numbered clauses: "1.", "2.", "3." at line start
    - Lettered points: "a)", "b)", "c)" or "a.", "b.", "c." at line start

    Args:
        tokenizer: Tokenizer for counting tokens
        content: Markdown content of the legal document
        chunk_token_size: Maximum tokens per chunk (default: 1200)
        chunk_overlap_token_size: Token overlap between chunks (default: 100)

    Returns:
        List of chunk dictionaries with keys:
        - tokens: Number of tokens in the chunk
        - content: Chunk content with context prefix
        - chunk_order_index: Sequential index of the chunk
        - context: Dict with 'chapter', 'article', and optional 'clauses'/'points' keys
    """
    import re

    _validate_chunking_limits(chunk_token_size, chunk_overlap_token_size)
    chunks = []
    current_chapter = ""
    current_article = ""

    # Patterns for detecting legal structure
    chapter_pattern = re.compile(
        r"^(#+)\s*(Chương\s+[IVXLCDM]+[:\s].*?)$", re.MULTILINE
    )
    article_pattern = re.compile(r"^(#+)\s*(Điều\s+\d+[.:\s].*?)$", re.MULTILINE)

    # Clause patterns: numbered clauses (1., 2.); point patterns: lettered points (a), a.)
    clause_pattern = re.compile(r"^\s*(\d+\.)\s+", re.MULTILINE)
    point_pattern = re.compile(r"^\s*([a-z]\)|[a-z]\.)\s+", re.MULTILINE)

    # Split content into lines for processing
    lines = content.split("\n")
    current_article_lines = []

    def build_context_prefix(clause_range: str = "", point_range: str = ""):
        """Build the context prefix for chunks.

        Args:
            clause_range: Optional clause range like "Khoản 1-3" or "Khoản 1"
            point_range: Optional point range like "Điểm a-c" or "Điểm a"
        """
        prefix_parts = []
        if current_chapter:
            prefix_parts.append(f"[{current_chapter}]")
        if current_article:
            prefix_parts.append(f"[{current_article}]")
        if clause_range:
            prefix_parts.append(f"[{clause_range}]")
        if point_range:
            prefix_parts.append(f"[{point_range}]")
        return " ".join(prefix_parts) + ("\n\n" if prefix_parts else "")

    def parse_clauses(article_content: str) -> list[dict[str, Any]]:
        """Parse article content into clauses.

        Returns:
            List of dicts with clause 'identifier', 'content', and nested 'points' keys
        """
        article_lines = article_content.split("\n")
        clauses = []
        current_clause = None
        current_point = None

        for line in article_lines:
            clause_match = clause_pattern.match(line)
            point_match = point_pattern.match(line)
            if clause_match:
                # Save previous clause if any
                if current_clause is not None:
                    if current_point is not None:
                        current_clause["points"].append(current_point)
                        current_point = None
                    clauses.append(current_clause)

                # Start new clause
                current_clause = {
                    "identifier": clause_match.group(1),
                    "content": line,
                    "points": [],
                }
            elif point_match:
                if current_clause is None:
                    current_clause = {"identifier": "", "content": "", "points": []}
                if current_point is not None:
                    current_clause["points"].append(current_point)
                current_point = {"identifier": point_match.group(1), "content": line}
            else:
                # Continue current point/clause or add to preamble
                if current_point is not None:
                    current_point["content"] += "\n" + line
                elif current_clause is not None:
                    current_clause["content"] += "\n" + line
                else:
                    # Preamble before first clause
                    if clauses or line.strip():
                        if not clauses:
                            clauses.append(
                                {"identifier": "", "content": line, "points": []}
                            )
                        else:
                            clauses[0]["content"] += "\n" + line

        # Save last clause
        if current_clause is not None:
            if current_point is not None:
                current_clause["points"].append(current_point)
            clauses.append(current_clause)

        return clauses

    def clause_content_with_points(clause: dict[str, Any]) -> str:
        parts = []
        if clause.get("content", "").strip():
            parts.append(clause["content"])
        parts.extend(point["content"] for point in clause.get("points", []))
        return "\n".join(parts)

    def format_clause_range(start_idx: int, end_idx: int, clauses: list) -> str:
        """Format clause range for context prefix.

        Args:
            start_idx: Starting clause index (0-based)
            end_idx: Ending clause index (0-based, inclusive)
            clauses: List of clause dicts

        Returns:
            Formatted string like "Khoản 1-3" or "Khoản 1"
        """
        clause_ids = [
            clause["identifier"].rstrip(".)").strip()
            for clause in clauses[start_idx : end_idx + 1]
            if clause.get("identifier", "").rstrip(".)").strip()
        ]
        if not clause_ids:
            return ""
        if len(clause_ids) == 1 or clause_ids[0] == clause_ids[-1]:
            return f"Khoản {clause_ids[0]}"
        return f"Khoản {clause_ids[0]}-{clause_ids[-1]}"

    def format_point_range(start_idx: int, end_idx: int, clauses: list) -> str:
        point_ids = []
        for clause in clauses[start_idx : end_idx + 1]:
            for point in clause.get("points", []):
                point_id = point["identifier"].rstrip(".)").strip()
                if point_id:
                    point_ids.append(point_id)

        if not point_ids:
            return ""
        if len(point_ids) == 1 or point_ids[0] == point_ids[-1]:
            return f"Điểm {point_ids[0]}"
        return f"Điểm {point_ids[0]}-{point_ids[-1]}"

    def create_chunks_from_article(article_content: str):
        """Create one or more chunks from article content, respecting clause boundaries."""
        if not article_content.strip():
            return

        context_prefix = build_context_prefix()
        full_content = context_prefix + article_content.strip()

        # Tokenize the full content
        tokens = tokenizer.encode(full_content)

        # If content fits in one chunk, create single chunk
        if len(tokens) <= chunk_token_size:
            clauses = parse_clauses(article_content)
            context = {
                "chapter": current_chapter,
                "article": current_article,
            }
            if clauses:
                clause_range = format_clause_range(0, len(clauses) - 1, clauses)
                point_range = format_point_range(0, len(clauses) - 1, clauses)
                if clause_range:
                    context["clauses"] = clause_range
                if point_range:
                    context["points"] = point_range

            chunks.append(
                {
                    "tokens": len(tokens),
                    "content": full_content,
                    "chunk_order_index": len(chunks),
                    "context": context,
                }
            )
            return

        # Article is too large - try to split by clauses
        clauses = parse_clauses(article_content)

        # If no clauses detected or only one clause, fall back to token-based splitting
        if len(clauses) <= 1:
            logger.warning(
                f"Article '{current_article}' exceeds token limit ({len(tokens)} > {chunk_token_size}) "
                f"but no clause structure detected. Falling back to token-based splitting."
            )
            for chunk_content, _, _ in _split_oversized_content(
                tokenizer,
                article_content.strip(),
                context_prefix,
                chunk_token_size,
                chunk_overlap_token_size,
            ):
                chunks.append(
                    {
                        "tokens": len(tokenizer.encode(chunk_content)),
                        "content": chunk_content.strip(),
                        "chunk_order_index": len(chunks),
                        "context": {
                            "chapter": current_chapter,
                            "article": current_article,
                        },
                    }
                )
            return

        # Group clauses to fit within token limits
        current_group = []
        current_group_start_idx = 0

        for i, clause in enumerate(clauses):
            # Calculate tokens for ONLY the current clause to see if it's independently too large
            clause_content = clause_content_with_points(clause)
            clause_range = format_clause_range(i, i, clauses)
            point_range = format_point_range(i, i, clauses)
            clause_prefix = build_context_prefix(clause_range, point_range)
            clause_full = clause_prefix + clause_content
            clause_tokens = len(tokenizer.encode(clause_full))

            if clause_tokens > chunk_token_size:
                # If there's a pending group, save it first
                if current_group:
                    group_content = "\n\n".join(
                        [clause_content_with_points(c) for c in current_group]
                    )
                    group_range = format_clause_range(
                        current_group_start_idx, i - 1, clauses
                    )
                    point_range = format_point_range(
                        current_group_start_idx, i - 1, clauses
                    )
                    group_prefix = build_context_prefix(group_range, point_range)
                    group_full = group_prefix + group_content

                    chunks.append(
                        {
                            "tokens": len(tokenizer.encode(group_full)),
                            "content": group_full.strip(),
                            "chunk_order_index": len(chunks),
                            "context": {
                                "chapter": current_chapter,
                                "article": current_article,
                                "clauses": group_range,
                                "points": point_range,
                            },
                        }
                    )
                    current_group = []

                # Split the current huge clause into multiple chunks using token-based boundaries
                for chunk_content_str, _, _ in _split_oversized_content(
                    tokenizer,
                    clause_content,
                    clause_prefix,
                    chunk_token_size,
                    chunk_overlap_token_size,
                ):
                    chunks.append(
                        {
                            "tokens": len(tokenizer.encode(chunk_content_str)),
                            "content": chunk_content_str.strip(),
                            "chunk_order_index": len(chunks),
                            "context": {
                                "chapter": current_chapter,
                                "article": current_article,
                                "clauses": clause_range,
                                "points": point_range,
                            },
                        }
                    )
                current_group = []
                current_group_start_idx = i + 1
                continue

            # Build tentative chunk with current group + new clause
            tentative_group = current_group + [clause]
            tentative_content = "\n\n".join(
                [clause_content_with_points(c) for c in tentative_group]
            )

            # Determine clause range for context
            clause_range = format_clause_range(current_group_start_idx, i, clauses)
            point_range = format_point_range(current_group_start_idx, i, clauses)
            tentative_prefix = build_context_prefix(clause_range, point_range)
            tentative_full = tentative_prefix + tentative_content
            tentative_tokens = len(tokenizer.encode(tentative_full))

            # If adding this clause exceeds limit and we have clauses in current group
            if tentative_tokens > chunk_token_size and current_group:
                # Save current group as chunk
                group_content = "\n\n".join(
                    [clause_content_with_points(c) for c in current_group]
                )
                group_range = format_clause_range(
                    current_group_start_idx, i - 1, clauses
                )
                point_range = format_point_range(
                    current_group_start_idx, i - 1, clauses
                )
                group_prefix = build_context_prefix(group_range, point_range)
                group_full = group_prefix + group_content

                chunks.append(
                    {
                        "tokens": len(tokenizer.encode(group_full)),
                        "content": group_full.strip(),
                        "chunk_order_index": len(chunks),
                        "context": {
                            "chapter": current_chapter,
                            "article": current_article,
                            "clauses": group_range,
                            "points": point_range,
                        },
                    }
                )

                # Start new group with current clause
                current_group = [clause]
                current_group_start_idx = i
            else:
                # Add clause to current group
                current_group.append(clause)

        # Save remaining group
        if current_group:
            group_content = "\n\n".join(
                [clause_content_with_points(c) for c in current_group]
            )
            group_range = format_clause_range(
                current_group_start_idx, len(clauses) - 1, clauses
            )
            point_range = format_point_range(
                current_group_start_idx, len(clauses) - 1, clauses
            )
            group_prefix = build_context_prefix(group_range, point_range)
            group_full = group_prefix + group_content

            chunks.append(
                {
                    "tokens": len(tokenizer.encode(group_full)),
                    "content": group_full.strip(),
                    "chunk_order_index": len(chunks),
                    "context": {
                        "chapter": current_chapter,
                        "article": current_article,
                        "clauses": group_range,
                        "points": point_range,
                    },
                }
            )

    # Process content line by line
    for line in lines:
        # Check for chapter header
        chapter_match = chapter_pattern.match(line)
        if chapter_match:
            # Save previous article if any
            if current_article_lines:
                create_chunks_from_article("\n".join(current_article_lines))
                current_article_lines = []

            # Update current chapter
            current_chapter = chapter_match.group(2).strip()
            current_article = ""
            continue

        # Check for article header
        article_match = article_pattern.match(line)
        if article_match:
            # Save previous article if any
            if current_article_lines:
                create_chunks_from_article("\n".join(current_article_lines))
                current_article_lines = []

            # Update current article
            current_article = article_match.group(2).strip()
            continue

        # Accumulate lines for current article
        current_article_lines.append(line)

    # Process any remaining article content
    if current_article_lines:
        create_chunks_from_article("\n".join(current_article_lines))

    for chunk in chunks:
        ctx = chunk.get("context") or {}
        parent_sec = build_parent_section(
            chapter=ctx.get("chapter", ""),
            article=ctx.get("article", ""),
        )
        chunk["parent_section"] = parent_sec
        if isinstance(ctx, dict) and parent_sec:
            ctx["parent_section"] = parent_sec

    return _enrich_chunks_with_partial_expiry(chunks, **kwargs)


def legal_chunking_html(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
    **kwargs,
) -> list[dict[str, Any]]:
    """Chunking for Vietnamese legal documents in HTML format."""
    from datetime import datetime

    doc_id = kwargs.get("doc_id")
    current_chapter_anchor = None
    current_article_anchor = None

    def block_anchor(block, tag_name: str = "", index: str = "") -> dict | None:
        anchor_id = str(block.get("id") or "")
        anchor_name = str(block.get("name") or "")
        if not anchor_id and not anchor_name and hasattr(block, "find"):
            nested = block.find("a", attrs={"id": True}) or block.find(
                "a", attrs={"name": True}
            )
            if nested is not None:
                anchor_id = str(nested.get("id") or "")
                anchor_name = str(nested.get("name") or "")
        if not anchor_id and not anchor_name:
            return None

        return {
            "tag": tag_name,
            "index": str(index or ""),
            "id": anchor_id,
            "name": anchor_name,
            "text": block.get_text(" ", strip=True)[:160],
        }

    def check_missing_id_log(block, tag_name: str, index: str):
        if doc_id and block.name != "table":
            if not _collect_element_anchor_ids(block):
                prefix_map = {
                    "Phần": "phan_",
                    "Chương": "chuong_",
                    "Mục": "muc_",
                    "Điều": "dieu_",
                    "Khoản": "khoan_",
                    "Điểm": "diem_",
                }
                prefix = prefix_map.get(tag_name, "unknown_")
                new_id = f"{prefix}{index}"
                block["id"] = new_id
                block["name"] = new_id

                try:
                    import os

                    os.makedirs("data/logs", exist_ok=True)
                    with open(
                        "data/logs/missing_id_legal_docs.log", "a", encoding="utf-8"
                    ) as f:
                        text_preview = block.get_text(" ", strip=True)[:50]
                        f.write(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Doc: {doc_id} - Thiếu ID ở {tag_name}, đã bổ sung {new_id}: {text_preview}...\n"
                        )
                except Exception as e:
                    logger.warning(f"Failed to write missing ID log: {e}")
        return block_anchor(block, tag_name, index)

    def collect_block_anchors(blocks) -> list[dict]:
        anchors = []
        for block in blocks:
            anchor = block_anchor(block)
            if anchor is not None:
                anchors.append(anchor)
        return anchors

    def dedupe_anchors(anchors: list[dict]) -> list[dict]:
        unique_anchors = []
        seen = set()
        for anchor in anchors:
            key = (anchor.get("id", ""), anchor.get("name", ""), anchor.get("text", ""))
            if key in seen:
                continue
            seen.add(key)
            unique_anchors.append(anchor)
        return unique_anchors

    def collect_clause_anchors(
        clauses: list[dict], include_article: bool = False
    ) -> list[dict]:
        anchors = [current_article_anchor] if include_article else []
        for clause in clauses:
            anchors.append(clause.get("anchor"))
            anchors.extend(point.get("anchor") for point in clause.get("points", []))
        return dedupe_anchors([anchor for anchor in anchors if anchor])

    def build_citation_position(
        anchors: list[dict] | None = None,
        token_start: int | None = None,
        token_end: int | None = None,
    ) -> dict:
        anchors = anchors or []
        position = {
            "anchors": anchors,
            "html_ids": [anchor["id"] for anchor in anchors if anchor.get("id")],
            "html_names": [
                anchor["name"] for anchor in anchors if anchor.get("name")
            ],
        }
        if anchors:
            position["start"] = anchors[0]
            position["end"] = anchors[-1]
        if token_start is not None:
            position["token_start"] = token_start
        if token_end is not None:
            position["token_end"] = token_end
        return position

    def attach_citation_position(
        context: dict,
        anchors: list[dict] | None = None,
        token_start: int | None = None,
        token_end: int | None = None,
    ) -> dict:
        anchors = anchors or []
        if current_chapter_anchor:
            context["chapter_anchor"] = current_chapter_anchor
        if current_article_anchor:
            context["article_anchor"] = current_article_anchor
        if not anchors:
            anchors = [
                anchor
                for anchor in [current_article_anchor, current_chapter_anchor]
                if anchor
            ]
        if anchors:
            context["citation_position"] = build_citation_position(
                anchors, token_start, token_end
            )
        elif token_start is not None or token_end is not None:
            context["citation_position"] = build_citation_position(
                [], token_start, token_end
            )
        return context

    soup = strip_presentational_markup(BeautifulSoup(content, "html.parser"))
    blocks = _extract_html_blocks(soup)

    _validate_chunking_limits(chunk_token_size, chunk_overlap_token_size)

    # Document-level hybrid detection: if HTML already carries dieu_* (or
    # phan_/chuong_/muc_) anchor IDs/names, split that level strictly by those
    # HTML markers to avoid false splits on cross-reference paragraphs.
    has_part_ids = False
    has_chapter_ids = False
    has_section_ids = False
    has_article_ids = False
    for block in blocks:
        level, _, is_name, _ = extract_html_structure_marker(block)
        if is_name or not level:
            continue
        if level == "part":
            has_part_ids = True
        elif level == "chapter":
            has_chapter_ids = True
        elif level == "section":
            has_section_ids = True
        elif level == "article":
            has_article_ids = True

    chunks = []
    current_part = ""
    current_chapter = ""
    current_section = ""
    current_article = ""

    clause_pattern = re.compile(r"^\s*(?:Khoản\s+)?(\d+)[.:]\s+", re.IGNORECASE)
    point_pattern = re.compile(r"^\s*(?:Điểm\s+)?([a-z])[.)]\s+", re.IGNORECASE)

    current_article_blocks: list[Any] = []
    current_article_header_blocks: list[Any] = []

    def current_parent_section() -> str | None:
        return build_parent_section(
            part=current_part,
            chapter=current_chapter,
            section=current_section,
            article=current_article,
        )

    def build_base_context(
        clause_range: str = "", point_range: str = ""
    ) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            "chapter": current_chapter,
            "article": current_article,
        }
        if current_part:
            ctx["part"] = current_part
        if current_section:
            ctx["section"] = current_section
        if clause_range:
            ctx["clauses"] = clause_range
        if point_range:
            ctx["points"] = point_range
        parent_sec = current_parent_section()
        if parent_sec:
            ctx["parent_section"] = parent_sec
        return ctx

    def build_context_prefix(clause_range: str = "", point_range: str = ""):
        prefix_parts = []
        if current_chapter:
            prefix_parts.append(f"[{current_chapter}]")
        if current_article:
            prefix_parts.append(f"[{current_article}]")
        if clause_range:
            prefix_parts.append(f"[{clause_range}]")
        if point_range:
            prefix_parts.append(f"[{point_range}]")
        return " ".join(prefix_parts) + ("\n\n" if prefix_parts else "")

    def format_clause_range(start_idx: int, end_idx: int, clauses: list) -> str:
        clause_ids = [
            clause["identifier"].rstrip(".)").strip()
            for clause in clauses[start_idx : end_idx + 1]
            if clause.get("identifier", "").rstrip(".)").strip()
        ]
        if not clause_ids:
            return ""
        if len(clause_ids) == 1 or clause_ids[0] == clause_ids[-1]:
            return f"Khoản {clause_ids[0]}"
        return f"Khoản {clause_ids[0]}-{clause_ids[-1]}"

    def format_point_range(start_idx: int, end_idx: int, clauses: list) -> str:
        point_ids = []
        for clause in clauses[start_idx : end_idx + 1]:
            for point in clause.get("points", []):
                point_id = point["identifier"].rstrip(".)").strip()
                if point_id:
                    point_ids.append(point_id)

        if not point_ids:
            return ""
        if len(point_ids) == 1 or point_ids[0] == point_ids[-1]:
            return f"Điểm {point_ids[0]}"
        return f"Điểm {point_ids[0]}-{point_ids[-1]}"

    def parse_clauses(article_blocks) -> list:
        clauses = []
        current_clause = None
        current_point = None

        for block in article_blocks:
            text = block.get_text(" ", strip=True)
            clause_match = clause_pattern.match(text)
            point_match = point_pattern.match(text)

            if clause_match:
                clause_idx = clause_match.group(1)
                check_missing_id_log(block, "Khoản", clause_idx)
                if current_clause is not None:
                    if current_point is not None:
                        current_clause["points"].append(current_point)
                        current_point = None
                    clauses.append(current_clause)
                current_clause = {
                    "identifier": clause_match.group(1),
                    "content": str(block),
                    "anchor": block_anchor(block, "Khoản", clause_idx),
                    "points": [],
                }
            elif point_match:
                point_idx = point_match.group(1)
                check_missing_id_log(block, "Điểm", point_idx)
                if current_clause is None:
                    current_clause = {"identifier": "", "content": "", "points": []}
                if current_point is not None:
                    current_clause["points"].append(current_point)
                current_point = {
                    "identifier": point_match.group(1),
                    "content": str(block),
                    "anchor": block_anchor(block, "Điểm", point_idx),
                }
            else:
                if current_point is not None:
                    current_point["content"] += "\n" + str(block)
                elif current_clause is not None:
                    current_clause["content"] += "\n" + str(block)
                else:
                    if not clauses:
                        clauses.append(
                            {"identifier": "", "content": str(block), "points": []}
                        )
                    else:
                        clauses[0]["content"] += "\n" + str(block)

        if current_clause is not None:
            if current_point is not None:
                current_clause["points"].append(current_point)
            clauses.append(current_clause)

        return clauses

    def clause_content_with_points(clause: dict[str, Any]) -> str:
        parts = []
        if clause.get("content", "").strip():
            parts.append(clause["content"])
        parts.extend(point["content"] for point in clause.get("points", []))
        return "\n".join(parts)

    def create_chunks_from_article(article_blocks, article_header_blocks=None):
        if not article_blocks:
            return

        header_blocks = list(article_header_blocks or [])
        has_leading_header = (
            bool(header_blocks)
            and len(article_blocks) >= len(header_blocks)
            and article_blocks[: len(header_blocks)] == header_blocks
        )
        article_header_html = (
            "\n".join(str(b) for b in header_blocks).strip()
            if has_leading_header
            else ""
        )
        body_blocks = (
            article_blocks[len(header_blocks) :]
            if has_leading_header
            else article_blocks
        )
        parent_sec = current_parent_section()

        def build_subchunk_prefix(
            clause_range: str = "", point_range: str = ""
        ) -> str:
            base_prefix = build_context_prefix(clause_range, point_range)
            if not article_header_html:
                return base_prefix
            candidate = f"{base_prefix}{article_header_html}\n"
            if len(tokenizer.encode(candidate)) < chunk_token_size:
                return candidate
            return base_prefix

        def append_chunk(
            chunk_text: str,
            context: dict[str, Any],
            anchors: list[dict] | None = None,
            token_start: int | None = None,
            token_end: int | None = None,
            tokens_count: int | None = None,
        ) -> None:
            cleaned = chunk_text.strip()
            chunks.append(
                {
                    "tokens": (
                        tokens_count
                        if tokens_count is not None
                        else len(tokenizer.encode(cleaned))
                    ),
                    "content": cleaned,
                    "chunk_order_index": len(chunks),
                    "parent_section": parent_sec,
                    "context": attach_citation_position(
                        context, anchors, token_start, token_end
                    ),
                }
            )

        article_content = "\n".join([str(b) for b in article_blocks])
        context_prefix = build_context_prefix()
        full_content = context_prefix + article_content

        tokens = len(tokenizer.encode(full_content))

        if tokens <= chunk_token_size:
            clauses = parse_clauses(body_blocks or article_blocks)
            clause_range = (
                format_clause_range(0, len(clauses) - 1, clauses) if clauses else ""
            )
            point_range = (
                format_point_range(0, len(clauses) - 1, clauses) if clauses else ""
            )
            context = build_base_context(clause_range, point_range)
            append_chunk(
                full_content,
                context,
                anchors=collect_clause_anchors(clauses, include_article=True)
                or collect_block_anchors(article_blocks),
                tokens_count=tokens,
            )
            return

        clauses = parse_clauses(body_blocks) if body_blocks else []

        if len(clauses) <= 1:
            split_source = (
                "\n".join(str(b) for b in body_blocks)
                if body_blocks
                else article_content
            )
            split_prefix = (
                build_subchunk_prefix() if body_blocks else context_prefix
            )
            for chunk_content_str, start, end in _split_oversized_content(
                tokenizer,
                split_source,
                split_prefix,
                chunk_token_size,
                chunk_overlap_token_size,
            ):
                append_chunk(
                    chunk_content_str,
                    build_base_context(),
                    anchors=collect_block_anchors(article_blocks),
                    token_start=start,
                    token_end=end,
                )
            return

        current_group = []
        current_group_start_idx = 0

        for i, clause in enumerate(clauses):
            # Calculate tokens for ONLY the current clause to see if it's independently too large
            clause_content = clause_content_with_points(clause)
            clause_range = format_clause_range(i, i, clauses)
            point_range = format_point_range(i, i, clauses)
            clause_prefix = build_subchunk_prefix(clause_range, point_range)
            clause_full = clause_prefix + clause_content
            clause_tokens = len(tokenizer.encode(clause_full))

            if clause_tokens > chunk_token_size:
                # If there's a pending group, save it first
                if current_group:
                    group_content = "\n".join(
                        [clause_content_with_points(c) for c in current_group]
                    )
                    group_range = format_clause_range(
                        current_group_start_idx, i - 1, clauses
                    )
                    group_point_range = format_point_range(
                        current_group_start_idx, i - 1, clauses
                    )
                    group_anchors = collect_clause_anchors(
                        current_group, include_article=True
                    )
                    group_prefix = build_subchunk_prefix(
                        group_range, group_point_range
                    )
                    group_full = group_prefix + group_content

                    append_chunk(
                        group_full,
                        build_base_context(group_range, group_point_range),
                        anchors=group_anchors,
                    )
                    current_group = []

                # Split the current huge clause into multiple chunks using token-based boundaries
                for chunk_content_str, start, end in _split_oversized_content(
                    tokenizer,
                    clause_content,
                    clause_prefix,
                    chunk_token_size,
                    chunk_overlap_token_size,
                ):
                    append_chunk(
                        chunk_content_str,
                        build_base_context(clause_range, point_range),
                        anchors=collect_clause_anchors([clause], include_article=True),
                        token_start=start,
                        token_end=end,
                    )
                current_group = []
                current_group_start_idx = i + 1
                continue

            tentative_group = current_group + [clause]
            tentative_content = "\n".join(
                [clause_content_with_points(c) for c in tentative_group]
            )
            tentative_clause_range = format_clause_range(
                current_group_start_idx, i, clauses
            )
            tentative_point_range = format_point_range(
                current_group_start_idx, i, clauses
            )
            tentative_prefix = build_subchunk_prefix(
                tentative_clause_range, tentative_point_range
            )
            tentative_full = tentative_prefix + tentative_content
            tentative_tokens = len(tokenizer.encode(tentative_full))

            if tentative_tokens > chunk_token_size and current_group:
                group_content = "\n".join(
                    [clause_content_with_points(c) for c in current_group]
                )
                group_range = format_clause_range(
                    current_group_start_idx, i - 1, clauses
                )
                group_point_range = format_point_range(
                    current_group_start_idx, i - 1, clauses
                )
                group_anchors = collect_clause_anchors(
                    current_group, include_article=True
                )
                group_prefix = build_subchunk_prefix(
                    group_range, group_point_range
                )
                group_full = group_prefix + group_content

                append_chunk(
                    group_full,
                    build_base_context(group_range, group_point_range),
                    anchors=group_anchors,
                )
                current_group = [clause]
                current_group_start_idx = i
            else:
                current_group.append(clause)

        if current_group:
            group_content = "\n".join(
                [clause_content_with_points(c) for c in current_group]
            )
            group_range = format_clause_range(
                current_group_start_idx, len(clauses) - 1, clauses
            )
            group_point_range = format_point_range(
                current_group_start_idx, len(clauses) - 1, clauses
            )
            group_anchors = collect_clause_anchors(
                current_group, include_article=True
            )
            group_prefix = build_subchunk_prefix(group_range, group_point_range)
            group_full = group_prefix + group_content

            append_chunk(
                group_full,
                build_base_context(group_range, group_point_range),
                anchors=group_anchors,
            )

    def flush_current_article() -> None:
        nonlocal current_article_blocks, current_article_header_blocks
        if current_article_blocks:
            create_chunks_from_article(
                current_article_blocks, current_article_header_blocks
            )
            current_article_blocks = []
            current_article_header_blocks = []

    for block in blocks:
        text = _normalize_heading_text(block.get_text(" ", strip=True))
        level, idx, is_name, _ = extract_html_structure_marker(block)

        if is_name:
            if level == "part" and current_part:
                current_part = _merge_heading_name(current_part, text)
                continue
            if level == "chapter" and current_chapter:
                current_chapter = _merge_heading_name(current_chapter, text)
                continue
            if level == "section" and current_section:
                current_section = _merge_heading_name(current_section, text)
                continue
            if level == "article" and current_article:
                current_article = _merge_heading_name(current_article, text)
                if current_article_anchor is not None:
                    current_article_anchor["text"] = current_article[:160]
                current_article_header_blocks.append(block)
                current_article_blocks.append(block)
                continue

        part_match = (
            None if (has_part_ids or level) else _TEXT_PART_RE.match(text)
        )
        if level == "part" or part_match:
            flush_current_article()
            part_idx = idx if level == "part" else part_match.group(1)
            check_missing_id_log(block, "Phần", part_idx)
            current_part = text
            current_chapter = ""
            current_section = ""
            current_article = ""
            current_chapter_anchor = None
            current_article_anchor = None
            continue

        chapter_match = (
            None if (has_chapter_ids or level) else _TEXT_CHAPTER_RE.match(text)
        )
        if level == "chapter" or chapter_match:
            flush_current_article()
            chapter_idx = idx if level == "chapter" else chapter_match.group(1)
            current_chapter_anchor = check_missing_id_log(
                block, "Chương", chapter_idx
            )
            current_chapter = text
            current_section = ""
            current_article = ""
            current_article_anchor = None
            continue

        section_match = (
            None if (has_section_ids or level) else _TEXT_SECTION_RE.match(text)
        )
        if level == "section" or section_match:
            flush_current_article()
            section_idx = idx if level == "section" else section_match.group(1)
            check_missing_id_log(block, "Mục", section_idx)
            current_section = text
            current_article = ""
            current_article_anchor = None
            continue

        article_match = (
            None if (has_article_ids or level) else _TEXT_ARTICLE_RE.match(text)
        )
        if level == "article" or article_match:
            flush_current_article()
            article_idx = idx if level == "article" else article_match.group(1)
            current_article_anchor = check_missing_id_log(
                block, "Điều", article_idx
            )
            current_article = text
            current_article_header_blocks = [block]
            current_article_blocks = [block]
            continue

        current_article_blocks.append(block)

    flush_current_article()

    return _enrich_chunks_with_partial_expiry(chunks, **kwargs)
