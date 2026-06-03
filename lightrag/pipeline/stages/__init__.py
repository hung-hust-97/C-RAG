from __future__ import annotations

from lightrag.pipeline.models import StageName

from .chunk_embedding import run_chunk_embedding_stage
from .chunking import run_chunking_stage
from .entity_extraction import run_entity_extraction_stage
from .extraction import run_extraction_stage
from .finalize import run_finalize_stage
from .kg_embedding import run_kg_embedding_stage
from .kg_merge import run_kg_merge_stage


def default_stage_handlers():
    return {
        StageName.EXTRACTION: run_extraction_stage,
        StageName.CHUNKING: run_chunking_stage,
        StageName.CHUNK_EMBEDDING: run_chunk_embedding_stage,
        StageName.ENTITY_EXTRACTION: run_entity_extraction_stage,
        StageName.KG_MERGE: run_kg_merge_stage,
        StageName.KG_EMBEDDING: run_kg_embedding_stage,
        StageName.FINALIZE: run_finalize_stage,
    }


__all__ = ["default_stage_handlers"]
