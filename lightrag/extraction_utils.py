from __future__ import annotations

import json
from typing import Any

import json_repair


def empty_extraction_parse_stats() -> dict[str, int]:
    return {
        "json_records": 0,
        "delimiter_records": 0,
        "salvaged_records": 0,
        "dropped_records": 0,
        "parse_errors": 0,
    }


def looks_like_json_payload(result: str) -> bool:
    stripped = (result or "").strip()
    return stripped.startswith("{") or stripped.startswith("[")


def json_loads_best_effort(result: str) -> Any:
    try:
        return json.loads(result)
    except Exception:
        return json_repair.loads(result)


def is_context_length_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return any(
        marker in error_text
        for marker in (
            "maximum context length",
            "context length",
            "input_tokens",
            "too many tokens",
            "token limit",
        )
    )


def should_retry_without_response_format(error: Exception) -> bool:
    if is_context_length_error(error):
        return False

    error_text = str(error).lower()
    return any(
        marker in error_text
        for marker in (
            "response_format",
            "structured",
            "json schema",
            "schema",
            "parse",
            "finish_reason",
            "unsupported",
            "not support",
        )
    )
