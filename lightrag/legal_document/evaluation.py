from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LegalRetrievalCase:
    query_id: str
    query: str
    expected_sources: tuple[str, ...]
    expected_citations: Mapping[str, str]


@dataclass(frozen=True)
class LegalRetrievalResult:
    query_id: str
    policy: str
    ranked_sources: tuple[str, ...]
    citations: Mapping[str, str]


@dataclass(frozen=True)
class RetrievalMetrics:
    query_count: int
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    citation_correctness: float
    citation_coverage: float


def _normalize_citation(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def evaluate_legal_retrieval(
    cases: Sequence[LegalRetrievalCase],
    results: Sequence[LegalRetrievalResult],
    *,
    policy: str,
    k: int = 5,
) -> RetrievalMetrics:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not cases:
        raise ValueError("At least one gold case is required")

    case_by_id = {case.query_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("Gold query_id values must be unique")

    results_by_id: dict[str, LegalRetrievalResult] = {}
    for result in results:
        if result.policy != policy:
            continue
        if result.query_id not in case_by_id:
            raise ValueError(f"Unknown retrieval query_id: {result.query_id}")
        if result.query_id in results_by_id:
            raise ValueError(
                f"Duplicate retrieval result for query_id={result.query_id} "
                f"policy={policy}"
            )
        results_by_id[result.query_id] = result

    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_values: list[float] = []
    citation_correct = 0
    citation_covered = 0
    expected_citation_count = 0

    for case in cases:
        relevant = set(case.expected_sources)
        if not relevant:
            raise ValueError(f"Gold case {case.query_id} has no expected sources")
        expected_citation_count += len(case.expected_citations)

        result = results_by_id.get(case.query_id)
        ranked = list(result.ranked_sources[:k]) if result else []
        found = relevant.intersection(ranked)
        recall_values.append(len(found) / len(relevant))

        reciprocal_rank = 0.0
        seen_sources: set[str] = set()
        for rank, source in enumerate(ranked, start=1):
            is_first_occurrence = source not in seen_sources
            seen_sources.add(source)
            if is_first_occurrence and source in relevant:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

        seen_sources.clear()
        dcg = 0.0
        for rank, source in enumerate(ranked, start=1):
            if source in seen_sources:
                continue
            seen_sources.add(source)
            if source in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_relevant_count = min(len(relevant), k)
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_relevant_count + 1)
        )
        ndcg_values.append(dcg / ideal_dcg if ideal_dcg else 0.0)

        result_citations = result.citations if result else {}
        for source in set(ranked):
            expected_citation = case.expected_citations.get(source)
            if expected_citation is None:
                continue
            citation_covered += 1
            actual_citation = result_citations.get(source)
            if actual_citation and _normalize_citation(
                actual_citation
            ) == _normalize_citation(expected_citation):
                citation_correct += 1

    total_queries = len(cases)
    return RetrievalMetrics(
        query_count=total_queries,
        recall_at_k=sum(recall_values) / total_queries,
        mean_reciprocal_rank=sum(reciprocal_ranks) / total_queries,
        ndcg_at_k=sum(ndcg_values) / total_queries,
        citation_correctness=(
            citation_correct / citation_covered if citation_covered else 0.0
        ),
        citation_coverage=(
            citation_covered / expected_citation_count
            if expected_citation_count
            else 0.0
        ),
    )


def load_legal_retrieval_cases(payload: Mapping[str, Any]) -> list[LegalRetrievalCase]:
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Gold dataset must contain a 'cases' array")

    cases: list[LegalRetrievalCase] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, Mapping):
            raise ValueError(f"Gold case at index {index} must be an object")
        query_id = str(raw_case.get("query_id") or "").strip()
        query = str(raw_case.get("query") or "").strip()
        if not query_id:
            raise ValueError(f"Gold case at index {index} requires query_id")
        if not query:
            raise ValueError(f"Gold case {query_id} requires query")
        raw_sources = raw_case.get("expected_sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(
                f"Gold case {query_id} requires non-empty expected_sources"
            )

        sources: list[str] = []
        citations: dict[str, str] = {}
        for raw_source in raw_sources:
            if not isinstance(raw_source, Mapping):
                raise ValueError(
                    f"Gold source in query_id={query_id} must be an object"
                )
            document_id = str(raw_source.get("document_id") or "").strip()
            source_id = str(
                raw_source.get("source_id") or raw_source.get("structure_path") or ""
            ).strip()
            citation = str(raw_source.get("citation") or "").strip()
            if not document_id or not source_id or not citation:
                raise ValueError(
                    "Every expected source requires document_id, "
                    "source_id (or legacy structure_path), and citation "
                    f"(query_id={query_id})"
                )
            source_key = f"{document_id}#{source_id}"
            if source_key in citations:
                raise ValueError(
                    f"Duplicate expected source {source_key} for query_id={query_id}"
                )
            sources.append(source_key)
            citations[source_key] = citation

        cases.append(
            LegalRetrievalCase(
                query_id=query_id,
                query=query,
                expected_sources=tuple(sources),
                expected_citations=citations,
            )
        )
    return cases


def load_legal_retrieval_results(
    payload: Mapping[str, Any],
) -> list[LegalRetrievalResult]:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Retrieval dataset must contain a 'results' array")

    results: list[LegalRetrievalResult] = []
    for index, raw_result in enumerate(raw_results):
        if not isinstance(raw_result, Mapping):
            raise ValueError(f"Retrieval result at index {index} must be an object")
        query_id = str(raw_result.get("query_id") or "").strip()
        policy = str(raw_result.get("policy") or "").strip()
        raw_sources = raw_result.get("ranked_sources")
        if not query_id or not policy:
            raise ValueError(
                f"Retrieval result at index {index} requires query_id and policy"
            )
        if not isinstance(raw_sources, list):
            raise ValueError(
                f"Retrieval result query_id={query_id} requires ranked_sources"
            )

        ranked_sources: list[str] = []
        citations: dict[str, str] = {}
        for raw_source in raw_sources:
            if not isinstance(raw_source, Mapping):
                raise ValueError(
                    f"Ranked source in query_id={query_id} must be an object"
                )
            document_id = str(raw_source.get("document_id") or "").strip()
            source_id = str(
                raw_source.get("source_id") or raw_source.get("structure_path") or ""
            ).strip()
            citation = str(raw_source.get("citation") or "").strip()
            if not document_id or not source_id:
                raise ValueError(
                    "Every ranked source requires document_id and source_id "
                    "(or legacy structure_path) "
                    f"(query_id={query_id})"
                )
            source_key = f"{document_id}#{source_id}"
            ranked_sources.append(source_key)
            if citation:
                citations[source_key] = citation
        results.append(
            LegalRetrievalResult(
                query_id=query_id,
                policy=policy,
                ranked_sources=tuple(ranked_sources),
                citations=citations,
            )
        )
    return results
