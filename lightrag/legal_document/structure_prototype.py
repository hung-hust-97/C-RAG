from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from lightrag.utils import Tokenizer


_HEADING_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s*)?(?P<text>.*?)\s*#*\s*$")
_STRUCTURE_RE = re.compile(
    r"^(?P<label>Phần|Chương|Mục|Tiểu mục|Điều)\s+"
    r"(?P<number>[IVXLCDM]+|\d+)(?:[.:\s]+(?P<title>.*))?$",
    re.IGNORECASE,
)
_CLAUSE_RE = re.compile(
    r"^\s*(?:(?:Khoản)\s+)?(?P<number>\d+)[.) :]\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_POINT_RE = re.compile(
    r"^\s*(?:(?:Điểm)\s+)?(?P<number>[a-z])[.)]\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_PAGE_MARKER_RE = re.compile(
    r"<!--\s*SOURCE_PAGE_START=(?P<start>\d+)\s+"
    r"SOURCE_PAGE_END=(?P<end>\d+)(?:\s+PART_INDEX=\d+)?\s*-->"
)
_LEVELS = {
    "DOCUMENT": 0,
    "PART": 1,
    "CHAPTER": 2,
    "SECTION": 3,
    "SUBSECTION": 4,
    "ARTICLE": 5,
    "CLAUSE": 6,
    "POINT": 7,
}
_LABELS = {
    "PART": "Phần",
    "CHAPTER": "Chương",
    "SECTION": "Mục",
    "SUBSECTION": "Tiểu mục",
    "ARTICLE": "Điều",
    "CLAUSE": "Khoản",
    "POINT": "Điểm",
}


@dataclass
class LegalStructureNode:
    kind: str
    number: str | None = None
    title: str | None = None
    path: str = "DOCUMENT"
    parent: LegalStructureNode | None = field(default=None, repr=False, compare=False)
    start_line: int | None = None
    end_line: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    anchor_ids: list[str] = field(default_factory=list)
    anchor_names: list[str] = field(default_factory=list)
    items: list[str | LegalStructureNode] = field(default_factory=list)
    children: list[LegalStructureNode] = field(default_factory=list)
    _child_counts: dict[str, int] = field(default_factory=dict, repr=False)

    def add_child(
        self,
        kind: str,
        *,
        number: str | None,
        title: str | None,
        line_number: int,
        anchor_id: str | None = None,
        anchor_name: str | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> LegalStructureNode:
        ordinal = self._child_counts.get(kind, 0) + 1
        self._child_counts[kind] = ordinal
        identity = number or "unlabelled"
        child = LegalStructureNode(
            kind=kind,
            number=number,
            title=title.strip() if title else None,
            path=f"{self.path}/{kind}:{identity}:{ordinal}",
            parent=self,
            start_line=line_number,
            end_line=line_number,
            page_start=page_start,
            page_end=page_end,
        )
        child.add_anchor(anchor_id, anchor_name)
        self.children.append(child)
        self.items.append(child)
        return child

    def append_line(
        self,
        text: str,
        line_number: int,
        *,
        anchor_id: str | None = None,
        anchor_name: str | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> None:
        self.items.append(text)
        if self.start_line is None:
            self.start_line = line_number
        self.end_line = line_number
        self.add_anchor(anchor_id, anchor_name)
        if page_start is not None:
            self.page_start = (
                page_start
                if self.page_start is None
                else min(self.page_start, page_start)
            )
        if page_end is not None:
            self.page_end = (
                page_end if self.page_end is None else max(self.page_end, page_end)
            )

    def add_anchor(
        self, anchor_id: str | None, anchor_name: str | None
    ) -> None:
        if anchor_id and anchor_id not in self.anchor_ids:
            self.anchor_ids.append(anchor_id)
        if anchor_name and anchor_name not in self.anchor_names:
            self.anchor_names.append(anchor_name)

    def label(self) -> str:
        if self.kind == "DOCUMENT":
            return ""
        label = _LABELS[self.kind]
        if self.number:
            label += f" {self.number}"
        if self.title:
            label += f". {self.title}"
        return label


def parse_legal_structure(content: str) -> LegalStructureNode:
    """Parse common Markdown legal headings and clause/point markers.

    Unrecognized lines are retained under the current structural node so partial
    or malformed input remains available to downstream fallback handling.
    """
    records = [
        {
            "text": line,
            "line_number": line_number,
            "page_start": None,
            "page_end": None,
            "anchor_id": None,
            "anchor_name": None,
        }
        for line_number, line in enumerate(content.splitlines(), start=1)
    ]
    return _parse_records(records)


def parse_legal_html_structure(content: str) -> LegalStructureNode:
    """Parse HTML legal blocks while preserving ids, names, tables, and page spans."""
    soup = BeautifulSoup(content, "html.parser")
    records: list[dict] = []
    for element in soup.find_all(
        ["p", "table", "h1", "h2", "h3", "h4", "h5", "h6"]
    ):
        if element.name == "p" and element.find_parent("table"):
            continue
        if element.name == "table":
            rows = []
            for row in element.find_all("tr"):
                cells = [
                    cell.get_text(" ", strip=True)
                    for cell in row.find_all(["th", "td"], recursive=False)
                ]
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")
            text = "\n".join(rows)
        else:
            text = element.get_text(" ", strip=True)
        if not text:
            continue
        page_start = element.get("data-page-start")
        page_end = element.get("data-page-end")
        records.append(
            {
                "text": text,
                "line_number": len(records) + 1,
                "page_start": int(page_start) if page_start else None,
                "page_end": int(page_end) if page_end else None,
                "anchor_id": str(element.get("id") or "") or None,
                "anchor_name": str(element.get("name") or "") or None,
            }
        )
    return _parse_records(records)


def _parse_records(records: list[dict]) -> LegalStructureNode:
    root = LegalStructureNode(kind="DOCUMENT")
    active: list[LegalStructureNode] = [root]
    containers: list[LegalStructureNode] = [root]

    current_page_start: int | None = None
    current_page_end: int | None = None
    for record in records:
        raw_line = record["text"]
        line_number = record["line_number"]
        if not raw_line.strip():
            continue
        page_marker = _PAGE_MARKER_RE.match(raw_line.strip())
        if page_marker:
            current_page_start = int(page_marker.group("start"))
            current_page_end = int(page_marker.group("end"))
            continue
        page_start = record.get("page_start") or current_page_start
        page_end = record.get("page_end") or current_page_end
        anchor_id = record.get("anchor_id")
        anchor_name = record.get("anchor_name")
        for node in active:
            node.end_line = line_number
        heading = _HEADING_RE.match(raw_line)
        heading_text = heading.group("text").strip() if heading else raw_line.strip()
        structure = _STRUCTURE_RE.match(heading_text)

        if structure:
            label = structure.group("label").casefold()
            kind = {
                "phần": "PART",
                "chương": "CHAPTER",
                "mục": "SECTION",
                "tiểu mục": "SUBSECTION",
                "điều": "ARTICLE",
            }[label]
            level = _LEVELS[kind]
            while len(containers) > 1 and _LEVELS[containers[-1].kind] >= level:
                containers.pop()
            parent = containers[-1]
            node = parent.add_child(
                kind,
                number=structure.group("number"),
                title=structure.group("title"),
                line_number=line_number,
                anchor_id=anchor_id,
                anchor_name=anchor_name,
                page_start=page_start,
                page_end=page_end,
            )
            while len(active) > 1 and _LEVELS[active[-1].kind] >= level:
                active.pop()
            active.append(node)
            containers.append(node)
            continue

        current_article = next(
            (node for node in reversed(active) if node.kind == "ARTICLE"),
            None,
        )
        current_clause = next(
            (node for node in reversed(active) if node.kind == "CLAUSE"),
            None,
        )
        point_match = _POINT_RE.match(raw_line)
        clause_match = _CLAUSE_RE.match(raw_line)

        if current_article is not None and point_match and current_clause is not None:
            while active[-1].kind == "POINT":
                active.pop()
            point = current_clause.add_child(
                "POINT",
                number=point_match.group("number"),
                title=None,
                line_number=line_number,
                anchor_id=anchor_id,
                anchor_name=anchor_name,
                page_start=page_start,
                page_end=page_end,
            )
            point.append_line(raw_line.strip(), line_number)
            active.append(point)
        elif current_article is not None and clause_match:
            while active[-1].kind in {"CLAUSE", "POINT"}:
                active.pop()
            clause = current_article.add_child(
                "CLAUSE",
                number=clause_match.group("number"),
                title=None,
                line_number=line_number,
                anchor_id=anchor_id,
                anchor_name=anchor_name,
                page_start=page_start,
                page_end=page_end,
            )
            clause.append_line(raw_line.strip(), line_number)
            active.append(clause)
        else:
            active[-1].append_line(
                raw_line.rstrip(),
                line_number,
                anchor_id=anchor_id,
                anchor_name=anchor_name,
                page_start=page_start,
                page_end=page_end,
            )

    return root


def _flatten(node: LegalStructureNode) -> str:
    parts = [
        item if isinstance(item, str) else _flatten(item)
        for item in node.items
    ]
    return "\n".join(part for part in parts if part.strip())


def _context_for(node: LegalStructureNode) -> str:
    labels: list[str] = []
    current: LegalStructureNode | None = node
    while current is not None:
        if current.kind != "DOCUMENT":
            labels.append(current.label())
        current = current.parent
    return " ".join(f"[{label}]" for label in reversed(labels))


def build_experimental_chunks(
    root: LegalStructureNode,
    tokenizer: Tokenizer,
    *,
    workspace_id: str,
    document_id: str,
    document_version_id: str,
    policy: str,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
) -> list[dict]:
    """Build isolated experimental chunks using article-first or clause-first."""
    if policy not in {"article_first", "clause_first"}:
        raise ValueError(f"Unsupported legal chunking policy: {policy}")
    if chunk_token_size <= 0:
        raise ValueError("chunk_token_size must be greater than zero")
    if not 0 <= chunk_overlap_token_size < chunk_token_size:
        raise ValueError(
            "chunk_overlap_token_size must be non-negative and smaller "
            "than chunk_token_size"
        )

    articles = [
        node for node in _walk(root) if node.kind == "ARTICLE"
    ]
    chunks: list[dict] = []

    def emit_text(
        text: str,
        node: LegalStructureNode,
        *,
        kind: str,
    ) -> None:
        context = _context_for(node)
        prefix = f"{context}\n\n" if context else ""
        source_tokens = tokenizer.encode(text.strip())
        start = 0
        piece_index = 0
        while start < len(source_tokens):
            end = min(start + chunk_token_size, len(source_tokens))
            while end > start:
                body = tokenizer.decode(source_tokens[start:end]).strip()
                rendered = f"{prefix}{body}".strip()
                if len(tokenizer.encode(rendered)) <= chunk_token_size:
                    break
                end -= 1
            if end == start:
                raise ValueError(
                    "Legal structure context leaves no room within the token limit"
                )
            chunk_identity = f"{node.path}:{piece_index}:{policy}"
            scoped_identity = (
                "legal-structure-prototype-v1:"
                f"{workspace_id}:{document_id}:{document_version_id}:"
                f"{chunk_identity}"
            )
            chunks.append(
                {
                    "chunk_id": hashlib.sha256(
                        scoped_identity.encode("utf-8")
                    ).hexdigest(),
                    "content": rendered,
                    "tokens": len(tokenizer.encode(rendered)),
                    "node_type": kind,
                    "structure_path": node.path,
                    "context": {
                        "breadcrumb": context,
                        "line_start": node.start_line,
                        "line_end": node.end_line,
                        "page_start": _inherited_page(node, "page_start"),
                        "page_end": _inherited_page(node, "page_end"),
                        "html_ids": _chunk_anchor_values(
                            node, "anchor_ids", include_descendants=kind == "ARTICLE"
                        ),
                        "html_names": _chunk_anchor_values(
                            node,
                            "anchor_names",
                            include_descendants=kind == "ARTICLE",
                        ),
                        "policy": policy,
                        "chunker_version": "legal-structure-prototype-v1",
                        "tokenizer_model": tokenizer.model_name,
                        "workspace_id": workspace_id,
                        "document_id": document_id,
                        "document_version_id": document_version_id,
                    },
                }
            )
            piece_index += 1
            if end == len(source_tokens):
                break
            start = max(start + 1, end - chunk_overlap_token_size)

    def emit_article(article: LegalStructureNode) -> None:
        clauses = [child for child in article.children if child.kind == "CLAUSE"]
        article_text = _flatten(article)
        if policy == "article_first" and len(tokenizer.encode(
            f"{_context_for(article)}\n\n{article_text}".strip()
        )) <= chunk_token_size:
            emit_text(article_text, article, kind="ARTICLE")
            return

        if clauses:
            non_clause_text = "\n".join(
                item for item in article.items if isinstance(item, str)
            ).strip()
            if non_clause_text:
                emit_text(non_clause_text, article, kind="ARTICLE")
            for clause in clauses:
                emit_text(_flatten(clause), clause, kind="CLAUSE")
        elif article_text.strip():
            emit_text(article_text, article, kind="ARTICLE")

    for article in articles:
        emit_article(article)

    def emit_unattached(node: LegalStructureNode) -> None:
        if node.kind == "ARTICLE":
            for child in node.children:
                emit_unattached(child)
            return
        if node.kind in {"CLAUSE", "POINT"}:
            return
        direct_text = "\n".join(
            item for item in node.items if isinstance(item, str)
        ).strip()
        if direct_text:
            emit_text(direct_text, node, kind=node.kind)
        for child in node.children:
            emit_unattached(child)

    emit_unattached(root)
    return chunks


def _walk(root: LegalStructureNode) -> list[LegalStructureNode]:
    result = [root]
    for child in root.children:
        result.extend(_walk(child))
    return result


def _inherited_values(node: LegalStructureNode, attribute: str) -> list[str]:
    values: list[str] = []
    current: LegalStructureNode | None = node
    while current is not None:
        for value in getattr(current, attribute):
            if value not in values:
                values.append(value)
        current = current.parent
    return values


def _chunk_anchor_values(
    node: LegalStructureNode,
    attribute: str,
    *,
    include_descendants: bool,
) -> list[str]:
    values = _inherited_values(node, attribute)
    if include_descendants:
        for descendant in _walk(node)[1:]:
            for value in getattr(descendant, attribute):
                if value not in values:
                    values.append(value)
    return values


def _inherited_page(node: LegalStructureNode, attribute: str) -> int | None:
    current: LegalStructureNode | None = node
    while current is not None:
        value = getattr(current, attribute)
        if value is not None:
            return value
        current = current.parent
    return None
