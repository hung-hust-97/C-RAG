from __future__ import annotations

import os
from typing import Any

_FALSE = {"0", "false", "no", "off"}


def event_doc_type(event: Any) -> str:
    raw = (
        (event.metadata or {}).get("doc_type")
        or getattr(event, "doc_type", None)
        or ("legal" if os.environ.get("PIPELINE_DOC_TYPE") == "legal" else "general")
    )
    return str(getattr(raw, "value", raw)).lower()


def legal_kg_enabled(event: Any) -> bool:
    """False only for legal documents when PIPELINE_LEGAL_KG_ENABLED is off."""
    if event_doc_type(event) != "legal":
        return True
    return os.getenv("PIPELINE_LEGAL_KG_ENABLED", "true").strip().lower() not in _FALSE
