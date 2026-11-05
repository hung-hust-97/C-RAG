"""Build unreviewed retrieval gold-set candidates by joining VLegal-Bench with a legal corpus.

VLegal-Bench Task 3.1 rows provide a question, multiple-choice options (A-D) citing
an article/clause and document, and the ``ground_truth`` letter. A legal corpus
(such as ``minhnguyent546/vietnamese-legal-documents``) provides full statutory
texts. This module parses the winning option's reference, matches it against the
corpus and its AST structure nodes, and emits candidate cases compatible with
``load_legal_retrieval_cases`` marked ``review_status="unreviewed"`` for legal
reviewer sign-off.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .structure_prototype import parse_legal_structure

_OPTION_LINE_RE = re.compile(
    r"^\s*([ABCD])[\.\)\:\-]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE
)
_ARTICLE_RE = re.compile(r"\bĐiều\s+(\d+[a-zA-Z]?)\b", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"\bKhoản\s+(\d+)\b", re.IGNORECASE)
_DOC_NUMBER_RE = re.compile(
    r"\b(\d+/\d{4}/[A-ZĐa-zđ0-9\-]+|\d+/[A-ZĐa-zđ0-9\-]+)\b"
)


@dataclass(frozen=True)
class ParsedLegalReference:
    option_letter: str
    raw_option_text: str
    article_number: str
    clause_number: str | None = None
    doc_number: str | None = None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def extract_ground_truth_reference(row: Mapping[str, Any]) -> ParsedLegalReference | None:
    """Extract the cited Article/Clause/DocNumber from the ground-truth option."""
    gt = row.get("ground_truth")
    if isinstance(gt, list):
        gt = gt[0] if len(gt) == 1 else None
    if not isinstance(gt, str):
        return None
    gt_letter = gt.strip().upper()
    answers = str(row.get("answers") or "")
    option_map = {
        match.group(1).upper(): match.group(2).strip()
        for match in _OPTION_LINE_RE.finditer(answers)
    }
    option_text = option_map.get(gt_letter)
    if not option_text:
        return None
    art_match = _ARTICLE_RE.search(option_text)
    if not art_match:
        return None
    clause_match = _CLAUSE_RE.search(option_text)
    doc_num_match = _DOC_NUMBER_RE.search(option_text)
    return ParsedLegalReference(
        option_letter=gt_letter,
        raw_option_text=option_text,
        article_number=art_match.group(1).upper(),
        clause_number=clause_match.group(1) if clause_match else None,
        doc_number=doc_num_match.group(1).upper() if doc_num_match else None,
    )


def _match_corpus_document(
    ref: ParsedLegalReference,
    corpus_docs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    norm_option = _normalize_text(ref.raw_option_text)
    if ref.doc_number:
        norm_num = ref.doc_number.casefold()
        for doc in corpus_docs:
            haystack = _normalize_text(
                f"{doc.get('doc_number', '')} {doc.get('title', '')}"
            )
            if norm_num in haystack:
                return doc
    for doc in corpus_docs:
        title = _normalize_text(str(doc.get("title") or ""))
        if title and len(title) >= 8 and title in norm_option:
            return doc
    return None


def _find_provision_source(
    content: str,
    article_number: str,
    clause_number: str | None,
) -> tuple[str, str] | None:
    """Return ``(source_id, citation)`` from the AST of ``content``."""
    tree = parse_legal_structure(content)
    all_nodes: list[Any] = []

    def _walk(node: Any) -> None:
        all_nodes.append(node)
        for child in getattr(node, "children", ()):
            _walk(child)

    _walk(tree)
    article_node = None
    for node in all_nodes:
        if getattr(node, "kind", None) == "ARTICLE" and str(
            getattr(node, "number", "")
        ).upper() == article_number.upper():
            article_node = node
            break
    if article_node is None:
        return None

    art_heading = f"Điều {article_node.number}"
    if article_node.title:
        art_heading = f"{art_heading}. {article_node.title}"

    if clause_number is not None:
        for child in getattr(article_node, "children", ()):
            if getattr(child, "kind", None) == "CLAUSE" and str(
                getattr(child, "number", "")
            ) == str(clause_number):
                citation = f"Khoản {clause_number} {art_heading}".strip()
                return child.path, citation

    return article_node.path, art_heading.strip()


def build_gold_candidates_from_corpus(
    vlegal_rows: Iterable[Mapping[str, Any]],
    corpus_rows: Iterable[Mapping[str, Any]],
    *,
    max_cases: int = 100,
) -> dict[str, Any]:
    """Match VLegal-Bench rows against ``corpus_rows`` to emit candidate gold JSON."""
    corpus_docs = [
        {
            "document_id": str(
                row.get("document_id") or row.get("id") or f"doc-{idx + 1:04d}"
            ),
            "title": str(row.get("title") or row.get("ten_van_ban") or ""),
            "doc_number": str(row.get("doc_number") or row.get("so_hieu") or ""),
            "content": str(
                row.get("content") or row.get("text") or row.get("noi_dung") or ""
            ),
        }
        for idx, row in enumerate(corpus_rows)
    ]

    cases: list[dict[str, Any]] = []
    for idx, row in enumerate(vlegal_rows, start=1):
        if len(cases) >= max_cases:
            break
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        ref = extract_ground_truth_reference(row)
        if ref is None:
            continue
        matched_doc = _match_corpus_document(ref, corpus_docs)
        if matched_doc is None or not matched_doc["content"].strip():
            continue
        provision = _find_provision_source(
            matched_doc["content"], ref.article_number, ref.clause_number
        )
        if provision is None:
            continue
        source_id, citation = provision
        cases.append(
            {
                "query_id": f"vlegal-cand-{idx:04d}",
                "query": question,
                "expected_sources": [
                    {
                        "document_id": matched_doc["document_id"],
                        "source_id": source_id,
                        "citation": citation,
                    }
                ],
            }
        )

    return {
        "dataset_type": "vlegal_corpus_candidates",
        "attribution": "Dataset created by CMC-OpenAI (joined with minhnguyent546/vietnamese-legal-documents)",
        "review_status": "unreviewed",
        "reviewed_by": None,
        "cases": cases,
    }
