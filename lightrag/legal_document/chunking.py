from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup

from lightrag.utils import Tokenizer, logger


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

    lien_quan_dan_chieu = kwargs.get("lien_quan_dan_chieu")
    if lien_quan_dan_chieu:
        for chunk in chunks:
            chunk["lien_quan_dan_chieu"] = lien_quan_dan_chieu
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
    import re
    from datetime import datetime

    doc_id = kwargs.get("doc_id")
    current_chapter_anchor = None
    current_article_anchor = None

    def block_anchor(block, tag_name: str = "", index: str = "") -> dict | None:
        anchor_id = str(block.get("id") or "")
        anchor_name = str(block.get("name") or "")
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
            if not block.get("id") and not block.get("name"):
                prefix_map = {
                    "Chương": "chuong_",
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

    soup = BeautifulSoup(content, "html.parser")

    blocks = []
    for element in soup.find_all(["p", "table"]):
        if element.name == "p" and element.find_parent("table"):
            continue
        blocks.append(element)

    _validate_chunking_limits(chunk_token_size, chunk_overlap_token_size)
    chunks = []
    current_chapter = ""
    current_article = ""

    chapter_pattern = re.compile(
        r"^Chương\s+([IVXLCDM0-9]+)(?:[.:\s].*)?$", re.IGNORECASE | re.DOTALL
    )
    article_pattern = re.compile(
        r"^Điều\s+(\d+)(?:[.:\s].*)?$", re.IGNORECASE | re.DOTALL
    )
    clause_pattern = re.compile(r"^\s*(?:Khoản\s+)?(\d+)[.:]\s+", re.IGNORECASE)
    point_pattern = re.compile(r"^\s*(?:Điểm\s+)?([a-z])[.)]\s+", re.IGNORECASE)

    current_article_blocks = []

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

    def create_chunks_from_article(article_blocks):
        if not article_blocks:
            return

        article_content = "\n".join([str(b) for b in article_blocks])
        context_prefix = build_context_prefix()
        full_content = context_prefix + article_content

        tokens = len(tokenizer.encode(full_content))

        if tokens <= chunk_token_size:
            clauses = parse_clauses(article_blocks)
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
            context = attach_citation_position(
                context,
                collect_clause_anchors(clauses, include_article=True)
                or collect_block_anchors(article_blocks),
            )

            chunks.append(
                {
                    "tokens": tokens,
                    "content": full_content.strip(),
                    "chunk_order_index": len(chunks),
                    "context": context,
                }
            )
            return

        clauses = parse_clauses(article_blocks)

        if len(clauses) <= 1:
            for chunk_content_str, start, end in _split_oversized_content(
                tokenizer,
                article_content,
                context_prefix,
                chunk_token_size,
                chunk_overlap_token_size,
            ):
                chunk_tokens = len(tokenizer.encode(chunk_content_str))
                chunks.append(
                    {
                        "tokens": chunk_tokens,
                        "content": chunk_content_str.strip(),
                        "chunk_order_index": len(chunks),
                        "context": attach_citation_position(
                            {
                                "chapter": current_chapter,
                                "article": current_article,
                            },
                            collect_block_anchors(article_blocks),
                            start,
                            end,
                        ),
                    }
                )
            return

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
                    group_content = "\n".join(
                        [clause_content_with_points(c) for c in current_group]
                    )
                    group_range = format_clause_range(
                        current_group_start_idx, i - 1, clauses
                    )
                    point_range = format_point_range(
                        current_group_start_idx, i - 1, clauses
                    )
                    group_anchors = collect_clause_anchors(current_group)
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
                            }
                            | attach_citation_position({}, group_anchors),
                        }
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
                    chunks.append(
                        {
                            "tokens": len(tokenizer.encode(chunk_content_str)),
                            "content": chunk_content_str.strip(),
                            "chunk_order_index": len(chunks),
                            "context": attach_citation_position(
                                {
                                    "chapter": current_chapter,
                                    "article": current_article,
                                    "clauses": clause_range,
                                    "points": point_range,
                                },
                                collect_clause_anchors([clause]),
                                start,
                                end,
                            ),
                        }
                    )
                current_group = []
                current_group_start_idx = i + 1
                continue

            tentative_group = current_group + [clause]
            tentative_content = "\n".join(
                [clause_content_with_points(c) for c in tentative_group]
            )
            clause_range = format_clause_range(current_group_start_idx, i, clauses)
            point_range = format_point_range(current_group_start_idx, i, clauses)
            tentative_prefix = build_context_prefix(clause_range, point_range)
            tentative_full = tentative_prefix + tentative_content
            tentative_tokens = len(tokenizer.encode(tentative_full))

            if tentative_tokens > chunk_token_size and current_group:
                group_content = "\n".join(
                    [clause_content_with_points(c) for c in current_group]
                )
                group_range = format_clause_range(
                    current_group_start_idx, i - 1, clauses
                )
                point_range = format_point_range(
                    current_group_start_idx, i - 1, clauses
                )
                group_anchors = collect_clause_anchors(current_group)
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
                        }
                        | attach_citation_position({}, group_anchors),
                    }
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
            point_range = format_point_range(
                current_group_start_idx, len(clauses) - 1, clauses
            )
            group_anchors = collect_clause_anchors(current_group)
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
                    }
                    | attach_citation_position({}, group_anchors),
                }
            )

    for block in blocks:
        text = block.get_text(" ", strip=True)

        chapter_match = chapter_pattern.match(text)
        if chapter_match:
            chapter_idx = chapter_match.group(1)
            current_chapter_anchor = check_missing_id_log(
                block, "Chương", chapter_idx
            )
            if current_article_blocks:
                create_chunks_from_article(current_article_blocks)
                current_article_blocks = []
            current_chapter = text
            continue

        article_match = article_pattern.match(text)
        if article_match:
            article_idx = article_match.group(1)
            current_article_anchor = check_missing_id_log(block, "Điều", article_idx)
            if current_article_blocks:
                create_chunks_from_article(current_article_blocks)
                current_article_blocks = []
            current_article = text

        current_article_blocks.append(block)

    if current_article_blocks:
        create_chunks_from_article(current_article_blocks)

    return _enrich_chunks_with_partial_expiry(chunks, **kwargs)
