from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re
from typing import Any


_ANSWER_OPTIONS = frozenset({"A", "B", "C", "D"})


@dataclass(frozen=True)
class VLegalBenchMetrics:
    sample_count: int
    correct_count: int
    accuracy: float
    invalid_prediction_count: int


def build_vlegal_task31_query(row: Mapping[str, Any]) -> str:
    instruction = row.get("instruction")
    question = row.get("question")
    answers = row.get("answers")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (instruction, question, answers)
    ):
        raise ValueError(
            "VLegal-Bench Task 3.1 row requires instruction, question, and answers"
        )
    return f"{instruction.lower()}\n{question}\n{answers}"


def parse_vlegal_task31_prediction(response: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", response, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _normalize_answer(value: Any) -> str | None:
    if isinstance(value, list):
        if len(value) != 1:
            return None
        value = value[0]
    if not isinstance(value, str):
        return None
    answer = value.strip().upper()
    return answer if answer in _ANSWER_OPTIONS else None


def evaluate_vlegal_bench_predictions(
    dataset_rows: Iterable[Mapping[str, Any]],
    prediction_rows: Iterable[Mapping[str, Any]],
    *,
    row_index_offset: int = 0,
) -> VLegalBenchMetrics:
    if row_index_offset < 0:
        raise ValueError("row_index_offset must be non-negative")
    sample_count = 0
    correct_count = 0
    invalid_prediction_count = 0

    dataset_iter = iter(dataset_rows)
    prediction_iter = iter(prediction_rows)
    row_index = 0
    while True:
        dataset_row = next(dataset_iter, None)
        prediction_row = next(prediction_iter, None)
        if dataset_row is None and prediction_row is None:
            break
        row_index += 1
        if dataset_row is None or prediction_row is None:
            raise ValueError(
                "Dataset and predictions must have the same number of rows "
                f"(mismatch at row {row_index})"
            )

        prediction_index = prediction_row.get("row_index")
        expected_row_index = row_index_offset + row_index
        if prediction_index is not None and prediction_index != expected_row_index:
            raise ValueError(
                f"Prediction row_index mismatch at row {expected_row_index}: "
                f"got {prediction_index}"
            )

        expected = _normalize_answer(dataset_row.get("ground_truth"))
        if expected is None:
            raise ValueError(
                f"VLegal-Bench row {row_index} has an invalid ground_truth"
            )
        prediction = _normalize_answer(prediction_row.get("prediction"))
        sample_count += 1
        if prediction is None:
            invalid_prediction_count += 1
        elif prediction == expected:
            correct_count += 1

    if sample_count == 0:
        raise ValueError("VLegal-Bench dataset contains no rows")

    return VLegalBenchMetrics(
        sample_count=sample_count,
        correct_count=correct_count,
        accuracy=correct_count / sample_count,
        invalid_prediction_count=invalid_prediction_count,
    )
