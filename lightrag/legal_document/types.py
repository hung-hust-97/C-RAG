from __future__ import annotations

import os

from pydantic import BaseModel, Field


LEGAL_EXTRACTION_MAX_ENTITIES = int(os.getenv("LEGAL_EXTRACTION_MAX_ENTITIES", "24"))
LEGAL_EXTRACTION_MAX_RELATIONS = int(os.getenv("LEGAL_EXTRACTION_MAX_RELATIONS", "36"))
LEGAL_EXTRACTION_MAX_NAME_CHARS = int(
    os.getenv("LEGAL_EXTRACTION_MAX_NAME_CHARS", "180")
)
LEGAL_EXTRACTION_MAX_TYPE_CHARS = int(
    os.getenv("LEGAL_EXTRACTION_MAX_TYPE_CHARS", "80")
)
LEGAL_EXTRACTION_MAX_DESCRIPTION_CHARS = int(
    os.getenv("LEGAL_EXTRACTION_MAX_DESCRIPTION_CHARS", "320")
)
LEGAL_EXTRACTION_MAX_KEYWORDS_CHARS = int(
    os.getenv("LEGAL_EXTRACTION_MAX_KEYWORDS_CHARS", "120")
)


class LegalExtractedEntity(BaseModel):
    entity_name: str = Field(max_length=LEGAL_EXTRACTION_MAX_NAME_CHARS)
    entity_type: str = Field(max_length=LEGAL_EXTRACTION_MAX_TYPE_CHARS)
    description: str = Field(max_length=LEGAL_EXTRACTION_MAX_DESCRIPTION_CHARS)


class LegalExtractedRelation(BaseModel):
    src_id: str = Field(max_length=LEGAL_EXTRACTION_MAX_NAME_CHARS)
    tgt_id: str = Field(max_length=LEGAL_EXTRACTION_MAX_NAME_CHARS)
    keywords: str = Field(max_length=LEGAL_EXTRACTION_MAX_KEYWORDS_CHARS)
    description: str = Field(max_length=LEGAL_EXTRACTION_MAX_DESCRIPTION_CHARS)
    directed: bool = True
    source_ids: list[str] = Field(default_factory=list)


class LegalExtractionResult(BaseModel):
    entities: list[LegalExtractedEntity] = Field(
        default_factory=list, max_length=LEGAL_EXTRACTION_MAX_ENTITIES
    )
    relations: list[LegalExtractedRelation] = Field(
        default_factory=list, max_length=LEGAL_EXTRACTION_MAX_RELATIONS
    )
