#!/usr/bin/env python3
"""Hybrid batch backfill for ``tbl_lightrag_document_chunk.parent_section``.

By default (hybrid mode), for each legal document this script fetches the source
HTML from S3/MinIO via ``tbl_legal_document.content_path``, builds the exact
``Phần > Chương > Mục > Điều`` anchor map using ``build_html_structure_index()``,
and maps every chunk (ordered by ``order_index``) by its HTML ``id``/``name`` or
article number with stateful carry-over for sub-chunks.

If S3 is unavailable/missing for a document (or when ``--db-only`` is passed),
it automatically falls back to stateful parsing of ``tbl_lightrag_document_chunk.content``
ordered by ``order_index``.

This script ONLY updates ``tbl_lightrag_document_chunk.parent_section`` in
PostgreSQL; it never modifies ``chunk_id``, ``content``, DDL, or Milvus.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

# Ensure repository root is importable when invoked as `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightrag.legal_document.chunking import (  # noqa: E402
    _TEXT_ARTICLE_RE,
    _TEXT_CHAPTER_RE,
    _TEXT_PART_RE,
    _TEXT_SECTION_RE,
    _collect_element_anchor_ids,
    _extract_html_blocks,
    _merge_heading_name,
    _normalize_heading_text,
    build_html_structure_index,
    build_parent_section,
    extract_html_structure_marker,
)
from lightrag.utils import logger  # noqa: E402

_BRACKET_PREFIX_LINE_RE = re.compile(r"^((?:\[[^\]]+\]\s*)+)(?:\n|$)")
_BRACKET_TOKEN_RE = re.compile(r"\[([^\]]+)\]")
_ARTICLE_NUM_RE = re.compile(r"Điều\s+(\d+[a-z]?)\b", re.IGNORECASE)


def _normalize_s3_object_key(value: Any) -> str | None:
    object_key = str(value or "").strip()
    if not object_key:
        return None
    if object_key.startswith("s3://"):
        bucket_and_key = object_key[5:]
        if "/" not in bucket_and_key:
            return None
        object_key = bucket_and_key.split("/", 1)[1]
    return object_key.lstrip("/") or None


def _extract_chunk_anchor_ids(content: str) -> list[str]:
    if not content or "<" not in content:
        return []
    soup = BeautifulSoup(content, "html.parser")
    seen: set[str] = set()
    ordered: list[str] = []
    for elem in soup.find_all(["p", "table", "a"]):
        for aid in _collect_element_anchor_ids(elem):
            lower = aid.lower()
            if lower not in seen:
                seen.add(lower)
                ordered.append(lower)
    return ordered


def _extract_bracket_tokens(content: str) -> list[str]:
    if not content:
        return []
    first_line = content.lstrip().split("\n", 1)[0]
    if not _BRACKET_PREFIX_LINE_RE.match(first_line):
        return []
    return [
        _normalize_heading_text(m.group(1))
        for m in _BRACKET_TOKEN_RE.finditer(first_line)
        if _normalize_heading_text(m.group(1))
    ]


def _pick_best_anchor_parent_section(
    anchor_ids: list[str], id_to_parent_section: dict[str, str]
) -> str | None:
    """Choose the most specific structural anchor present in a chunk."""
    priority_prefixes = (
        "dieu_",
        "khoan_",
        "diem_",
        "tieumuc_",
        "tieu_muc_",
        "muc_",
        "chuong_",
        "phan_",
    )
    for prefix in priority_prefixes:
        for aid in anchor_ids:
            if aid.startswith(prefix) and aid in id_to_parent_section:
                return id_to_parent_section[aid]
    for aid in anchor_ids:
        if aid in id_to_parent_section:
            return id_to_parent_section[aid]
    return None


def _extract_article_num_from_chunk(content: str) -> str | None:
    for token in _extract_bracket_tokens(content):
        m = _TEXT_ARTICLE_RE.match(token)
        if m:
            return m.group(1).lower()
    if content:
        soup = BeautifulSoup(content, "html.parser")
        for block in _extract_html_blocks(soup):
            text = _normalize_heading_text(block.get_text(" ", strip=True))
            m = _TEXT_ARTICLE_RE.match(text)
            if m:
                return m.group(1).lower()
        first_line = _normalize_heading_text(content.splitlines()[0])
        m = _ARTICLE_NUM_RE.search(first_line)
        if m:
            return m.group(1).lower()
    return None


def _resolve_from_s3_html(
    chunks: list[dict[str, Any]], html_content: str
) -> dict[Any, str | None]:
    index = build_html_structure_index(html_content)
    id_map: dict[str, str] = index["id_to_parent_section"]
    art_map: dict[str, str] = index["article_num_to_parent_section"]

    resolved: dict[Any, str | None] = {}
    last_parent_section: str | None = None

    for chunk in chunks:
        content = chunk.get("content") or ""
        anchor_ids = _extract_chunk_anchor_ids(content)
        parent_sec = _pick_best_anchor_parent_section(anchor_ids, id_map)

        if not parent_sec:
            art_num = _extract_article_num_from_chunk(content)
            if art_num and art_num in art_map:
                parent_sec = art_map[art_num]

        if not parent_sec:
            parent_sec = last_parent_section

        if parent_sec:
            last_parent_section = parent_sec
        resolved[chunk["id"]] = parent_sec

    return resolved


def _resolve_from_db_chunks(chunks: list[dict[str, Any]]) -> dict[Any, str | None]:
    """Stateful fallback parser using chunk contents ordered by order_index."""
    current_part = ""
    current_chapter = ""
    current_section = ""
    current_article = ""

    resolved: dict[Any, str | None] = {}

    for chunk in chunks:
        content = chunk.get("content") or ""

        # 1. Inspect bracket context prefix if present (e.g. "[Chương III] [Điều 35. ...]")
        for token in _extract_bracket_tokens(content):
            if _TEXT_PART_RE.match(token):
                if token != current_part:
                    current_part = token
                    current_chapter = ""
                    current_section = ""
                    current_article = ""
            elif _TEXT_CHAPTER_RE.match(token):
                if token != current_chapter:
                    current_chapter = token
                    current_section = ""
                    current_article = ""
            elif _TEXT_SECTION_RE.match(token):
                if token != current_section:
                    current_section = token
                    current_article = ""
            elif _TEXT_ARTICLE_RE.match(token):
                current_article = token

        # 2. Inspect HTML blocks inside chunk content
        if "<" in content:
            soup = BeautifulSoup(content, "html.parser")
            blocks = _extract_html_blocks(soup)
            has_article_ids = any(
                extract_html_structure_marker(b)[0] == "article"
                and not extract_html_structure_marker(b)[2]
                for b in blocks
            )
            for block in blocks:
                text = _normalize_heading_text(block.get_text(" ", strip=True))
                level, _, is_name, _ = extract_html_structure_marker(block)
                if is_name:
                    if level == "part" and current_part:
                        current_part = _merge_heading_name(current_part, text)
                    elif level == "chapter" and current_chapter:
                        current_chapter = _merge_heading_name(current_chapter, text)
                    elif level == "section" and current_section:
                        current_section = _merge_heading_name(current_section, text)
                    elif level == "article" and current_article:
                        current_article = _merge_heading_name(current_article, text)
                    continue

                if level == "part" or (not level and _TEXT_PART_RE.match(text)):
                    current_part = text
                    current_chapter = ""
                    current_section = ""
                    current_article = ""
                elif level == "chapter" or (not level and _TEXT_CHAPTER_RE.match(text)):
                    current_chapter = text
                    current_section = ""
                    current_article = ""
                elif level == "section" or (not level and _TEXT_SECTION_RE.match(text)):
                    current_section = text
                    current_article = ""
                elif level == "article" or (
                    not has_article_ids and not level and _TEXT_ARTICLE_RE.match(text)
                ):
                    current_article = text

        resolved[chunk["id"]] = build_parent_section(
            part=current_part,
            chapter=current_chapter,
            section=current_section,
            article=current_article,
        )

    return resolved


def resolve_document_chunk_parent_sections(
    chunks: list[dict[str, Any]],
    html_content: str | None = None,
) -> tuple[dict[Any, str | None], str]:
    """Resolve ``parent_section`` for all chunks of a single legal document.

    Args:
        chunks: Chunk dicts ordered by ``order_index ASC, id ASC``.
        html_content: Optional original HTML loaded from S3.

    Returns:
        Tuple of ``(chunk_id -> parent_section, source_mode)`` where
        ``source_mode`` is ``"s3"`` or ``"db_fallback"``.
    """
    if html_content and html_content.strip():
        s3_resolved = _resolve_from_s3_html(chunks, html_content)
        if any(s3_resolved.values()):
            return s3_resolved, "s3"
    return _resolve_from_db_chunks(chunks), "db_fallback"


async def _fetch_s3_html(s3_client: Any, bucket: str, content_path: str | None) -> str | None:
    object_key = _normalize_s3_object_key(content_path)
    if not object_key or s3_client is None:
        return None
    try:
        response = await s3_client.get_object(Bucket=bucket, Key=object_key)
        async with response["Body"] as stream:
            raw = await stream.read()
        return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("S3 fetch failed for key=%s: %s", object_key, exc)
        return None


async def run_backfill(args: argparse.Namespace) -> dict[str, int]:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    import asyncpg

    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_port = int(os.getenv("POSTGRES_PORT", "5432"))
    pg_user = os.getenv("POSTGRES_USER", "postgres")
    pg_password = os.getenv("POSTGRES_PASSWORD", "")
    pg_database = os.getenv("POSTGRES_DATABASE", "lightrag")

    conn = await asyncpg.connect(
        host=pg_host,
        port=pg_port,
        user=pg_user,
        password=pg_password,
        database=pg_database,
    )

    s3_ctx = None
    s3_client = None
    s3_bucket = os.getenv("S3_BUCKET") or os.getenv("S3_BUCKET_NAME", "lightrag-markdown")
    if not args.db_only:
        endpoint_url = os.getenv("S3_ENDPOINT_URL")
        access_key = os.getenv("S3_ACCESS_KEY") or os.getenv("S3_ACCESS_KEY_ID")
        secret_key = os.getenv("S3_SECRET_KEY") or os.getenv("S3_SECRET_ACCESS_KEY")
        if all([endpoint_url, access_key, secret_key]):
            import aioboto3

            session = aioboto3.Session()
            s3_ctx = session.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
            s3_client = await s3_ctx.__aenter__()
        else:
            logger.warning(
                "S3 credentials not configured; proceeding in DB-only fallback mode."
            )

    stats = {
        "docs_processed": 0,
        "docs_s3": 0,
        "docs_db_fallback": 0,
        "chunks_scanned": 0,
        "chunks_updated": 0,
        "chunks_skipped_existing": 0,
        "chunks_unresolved": 0,
    }

    try:
        where_clauses = ["c.legal_document_id IS NOT NULL"]
        params: list[Any] = []
        if not args.overwrite:
            where_clauses.append(
                "(c.parent_section IS NULL OR BTRIM(c.parent_section) = '')"
            )
        if args.workspace_id:
            params.append(args.workspace_id)
            where_clauses.append(f"ld.workspace_id = ${len(params)}::uuid")
        if args.legal_doc_id is not None:
            params.append(args.legal_doc_id)
            where_clauses.append(f"c.legal_document_id = ${len(params)}")

        limit_sql = ""
        if args.limit is not None and args.limit > 0:
            params.append(args.limit)
            limit_sql = f" LIMIT ${len(params)}"

        doc_query = f"""
            SELECT DISTINCT c.legal_document_id, ld.content_path
            FROM tbl_lightrag_document_chunk c
            LEFT JOIN tbl_legal_document ld ON ld.id = c.legal_document_id
            WHERE {' AND '.join(where_clauses)}
            ORDER BY c.legal_document_id ASC
            {limit_sql}
        """
        doc_rows = await conn.fetch(doc_query, *params)
        logger.info(
            "Found %d legal documents to backfill (dry_run=%s, db_only=%s, overwrite=%s)",
            len(doc_rows),
            args.dry_run,
            args.db_only,
            args.overwrite,
        )

        pending_updates: list[tuple[str, Any]] = []
        sample_printed = 0

        async def flush_updates() -> None:
            if not pending_updates or args.dry_run:
                pending_updates.clear()
                return
            async with conn.transaction():
                await conn.executemany(
                    "UPDATE tbl_lightrag_document_chunk SET parent_section = $1 WHERE id = $2",
                    pending_updates,
                )
            pending_updates.clear()

        for doc_row in doc_rows:
            legal_doc_id = doc_row["legal_document_id"]
            content_path = doc_row["content_path"]

            chunk_rows = await conn.fetch(
                """
                SELECT id, order_index, content, parent_section
                FROM tbl_lightrag_document_chunk
                WHERE legal_document_id = $1
                ORDER BY order_index ASC, id ASC
                """,
                legal_doc_id,
            )
            if not chunk_rows:
                continue

            chunks = [dict(r) for r in chunk_rows]
            html_content = None
            if not args.db_only and s3_client is not None and content_path:
                html_content = await _fetch_s3_html(s3_client, s3_bucket, content_path)

            resolved_map, mode = resolve_document_chunk_parent_sections(
                chunks, html_content=html_content
            )
            stats["docs_processed"] += 1
            if mode == "s3":
                stats["docs_s3"] += 1
            else:
                stats["docs_db_fallback"] += 1

            for chunk in chunks:
                stats["chunks_scanned"] += 1
                existing = (chunk.get("parent_section") or "").strip()
                if existing and not args.overwrite:
                    stats["chunks_skipped_existing"] += 1
                    continue

                new_ps = resolved_map.get(chunk["id"])
                if not new_ps:
                    stats["chunks_unresolved"] += 1
                    continue

                if new_ps == existing:
                    stats["chunks_skipped_existing"] += 1
                    continue

                stats["chunks_updated"] += 1
                pending_updates.append((new_ps, chunk["id"]))

                if args.dry_run and sample_printed < args.sample:
                    logger.info(
                        "[DRY-RUN][%s] legal_doc_id=%s order=%s -> %s",
                        mode,
                        legal_doc_id,
                        chunk.get("order_index"),
                        new_ps,
                    )
                    sample_printed += 1

            if len(pending_updates) >= args.batch_size:
                await flush_updates()

            if stats["docs_processed"] % 100 == 0:
                logger.info(
                    "Progress: docs=%d/%d (s3=%d, db=%d) chunks_updated=%d unresolved=%d",
                    stats["docs_processed"],
                    len(doc_rows),
                    stats["docs_s3"],
                    stats["docs_db_fallback"],
                    stats["chunks_updated"],
                    stats["chunks_unresolved"],
                )

        await flush_updates()
        logger.info("Backfill complete: %s", stats)
        return stats
    finally:
        if s3_ctx is not None:
            await s3_ctx.__aexit__(None, None, None)
        await conn.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill tbl_lightrag_document_chunk.parent_section from S3 HTML or DB chunk content."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute parent_section and log samples without updating PostgreSQL.",
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="Skip S3 HTML lookup and parse directly from DB chunk contents.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite non-null parent_section values (default: only NULL/empty).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of chunk UPDATE rows per DB transaction batch (default: 500).",
    )
    parser.add_argument(
        "--workspace-id",
        type=str,
        default=None,
        help="Optional workspace UUID filter.",
    )
    parser.add_argument(
        "--legal-doc-id",
        type=int,
        default=None,
        help="Optional single legal_document_id filter for targeted testing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of legal documents to process.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=15,
        help="Maximum number of sample updates to log during --dry-run (default: 15).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(run_backfill(args))


if __name__ == "__main__":
    main()
