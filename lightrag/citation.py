"""Citation system for LightRAG LLM responses.

This module provides data structures and utilities for tracking, formatting,
and rendering citations in LLM-generated responses. It supports both Vietnamese
legal documents and general documents with different citation formats.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
import re

# Import metrics tracking (lazy import to avoid circular dependencies)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lightrag.citation_metrics import CitationMetricsTracker


@dataclass
class CitationMetadata:
    """Unified metadata structure for all citation types.
    
    This class represents a single citation reference, supporting both legal
    documents (with Vietnamese legal metadata) and general documents (with
    breadcrumb navigation).
    """
    
    # Core identifiers
    citation_id: int  # Sequential ID: 1, 2, 3, ...
    doc_id: str  # Document ID (e.g., "doc-550e8400-e29b-41d4-a716-446655440000")
    chunk_id: Optional[str] = None  # Chunk ID (for chunks)
    
    # Document classification
    doc_type: Literal["LEGAL", "GENERAL"] = "GENERAL"  # Document type (uppercase)
    
    # Source type
    source_type: Literal["chunk", "entity", "relation"] = "chunk"  # Type of source
    entity_name: Optional[str] = None  # Entity name (for entities)
    relation_key: Optional[str] = None  # Relation key (for relations)
    
    # Workspace and location
    workspace: str = ""  # Source workspace name
    file_path: str = ""  # Original file path
    
    # Unified hierarchical structure (flexible for both types)
    # Legal example: ["Chương II: Thuế GTGT", "Điều 8: Thuế suất", "Khoản 2"]
    # General example: ["Hạ tầng", "Quy trình Backup", "Bước 1"]
    hierarchy_path: List[str] = field(default_factory=list)  # Hierarchical path from root to current position
    
    # Relevance and content
    relevance_score: float = 0.0  # Similarity/relevance score (0.0-1.0)
    content_preview: str = ""  # First 100 chars of content
    
    # Legal-specific metadata (optional, only populated when doc_type="LEGAL")
    legal_info: Optional[Dict[str, str]] = None
    # Expected keys when doc_type="LEGAL":
    # {
    #     "article": "Điều 8",
    #     "clause": "Khoản 2", 
    #     "chapter": "Chương II",
    #     "issuing_authority": "Quốc hội",
    #     "effective_date": "2008-06-03",
    #     "document_type": "Luật",
    #     "document_number": "13/2008/QH12",
    #     "legal_status": "Còn hiệu lực"
    # }
    
    # Pre-rendered citation reference (helper field for quick display)
    # Legal: "Khoản 2 Điều 8 Luật Thuế GTGT số 13/2008/QH12"
    # General: "Mục Quy trình Backup (Tài liệu HDSD Hạ tầng)"
    formatted_ref: str = ""
    
    # Timestamps
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())  # ISO timestamp of retrieval
    
    @property
    def confidence_level(self) -> Literal["high", "medium", "low"]:
        """Compute confidence level based on relevance score.
        
        Confidence levels:
        - High: relevance_score >= 0.8 (strong match)
        - Medium: 0.5 <= relevance_score < 0.8 (moderate match)
        - Low: relevance_score < 0.5 (weak match)
        
        Returns:
            Literal["high", "medium", "low"]: Confidence level classification
        """
        return compute_confidence_level(self.relevance_score)


@dataclass
class CitationList:
    """Complete citation list for a query response.
    
    This class aggregates all citations used in a response and provides
    metadata about the citation format and sources.
    """
    
    citations: List[CitationMetadata] = field(default_factory=list)  # All citations in order
    format: Literal["inline", "footnote", "bibliography"] = "footnote"  # Citation format
    total_sources: int = 0  # Total number of unique sources
    workspaces_used: List[str] = field(default_factory=list)  # List of workspaces queried
    
    # Statistics by document type
    legal_count: int = 0  # Number of legal citations
    general_count: int = 0  # Number of general citations


def compute_confidence_level(relevance_score: float) -> Literal["high", "medium", "low"]:
    """Compute confidence level based on relevance score.
    
    This function classifies citation confidence into three levels based on
    the relevance score from vector similarity search. Higher scores indicate
    stronger semantic matches between the query and the retrieved content.
    
    Confidence thresholds:
    - High: >= 0.8 (strong semantic match, highly relevant)
    - Medium: 0.5-0.8 (moderate match, reasonably relevant)
    - Low: < 0.5 (weak match, potentially less relevant)
    
    Args:
        relevance_score: Similarity/relevance score from vector search (0.0-1.0)
    
    Returns:
        Literal["high", "medium", "low"]: Confidence level classification
    
    Example:
        >>> compute_confidence_level(0.95)
        'high'
        >>> compute_confidence_level(0.65)
        'medium'
        >>> compute_confidence_level(0.35)
        'low'
    """
    if relevance_score >= 0.8:
        return "high"
    elif relevance_score >= 0.5:
        return "medium"
    else:
        return "low"


def create_legal_citation_metadata(
    citation_id: int,
    doc_id: str,
    chunk_id: str,
    workspace: str,
    file_path: str,
    hierarchy_path: List[str],
    legal_info: Dict[str, str],
    relevance_score: float,
    content_preview: str,
) -> CitationMetadata:
    """Create citation metadata for legal documents.
    
    Factory function for creating CitationMetadata instances for Vietnamese
    legal documents with proper legal metadata and formatting.
    
    Args:
        citation_id: Sequential citation ID (1, 2, 3, ...)
        doc_id: Document identifier
        chunk_id: Chunk identifier
        workspace: Source workspace name
        file_path: Original file path
        hierarchy_path: Legal hierarchy (e.g., ["Chương II", "Điều 8", "Khoản 2"])
        legal_info: Dictionary with legal metadata fields:
            - article: Article number (e.g., "Điều 8")
            - clause: Clause number (e.g., "Khoản 2")
            - chapter: Chapter (e.g., "Chương II")
            - document_type: Type (e.g., "Luật", "Nghị định")
            - document_number: Number (e.g., "13/2008/QH12")
            - issuing_authority: Authority (e.g., "Quốc hội")
            - legal_status: Status (e.g., "Còn hiệu lực")
            - effective_date: Date (e.g., "2008-06-03")
        relevance_score: Relevance score (0.0-1.0)
        content_preview: First 100 characters of content
    
    Returns:
        CitationMetadata instance configured for legal documents
    
    Example:
        >>> metadata = create_legal_citation_metadata(
        ...     citation_id=1,
        ...     doc_id="doc-550e8400-e29b-41d4-a716-446655440000",
        ...     chunk_id="chunk-1",
        ...     workspace="legal-vn-thue",
        ...     file_path="/legal/luat_thue_gtgt_2008.pdf",
        ...     hierarchy_path=["Chương II: Thuế GTGT", "Điều 8: Thuế suất", "Khoản 2"],
        ...     legal_info={
        ...         "article": "Điều 8",
        ...         "clause": "Khoản 2",
        ...         "document_type": "Luật",
        ...         "document_number": "13/2008/QH12",
        ...         "issuing_authority": "Quốc hội",
        ...         "legal_status": "Còn hiệu lực"
        ...     },
        ...     relevance_score=0.95,
        ...     content_preview="Thuế suất thuế giá trị gia tăng là 10%..."
        ... )
        >>> metadata.formatted_ref
        'Khoản 2 Điều 8 Luật số 13/2008/QH12'
    """
    # Handle missing legal_info gracefully (Requirement 15.4)
    if not legal_info:
        legal_info = {}
    
    # Build formatted reference for legal documents
    article = legal_info.get("article", "")
    clause = legal_info.get("clause", "")
    doc_type = legal_info.get("document_type", "")
    doc_number = legal_info.get("document_number", "")
    
    parts = []
    if clause:
        parts.append(clause)
    if article:
        parts.append(article)
    if doc_type and doc_number:
        parts.append(f"{doc_type} số {doc_number}")
    elif doc_type:
        parts.append(doc_type)
    
    formatted_ref = " ".join(parts) if parts else file_path
    
    return CitationMetadata(
        citation_id=citation_id,
        doc_id=doc_id,
        chunk_id=chunk_id,
        doc_type="LEGAL",
        source_type="chunk",
        workspace=workspace,
        file_path=file_path,
        hierarchy_path=hierarchy_path,
        relevance_score=relevance_score,
        content_preview=content_preview,
        legal_info=legal_info if legal_info else None,
        formatted_ref=formatted_ref,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )


def create_general_citation_metadata(
    citation_id: int,
    doc_id: str,
    chunk_id: str,
    workspace: str,
    file_path: str,
    hierarchy_path: List[str],
    relevance_score: float,
    content_preview: str,
) -> CitationMetadata:
    """Create citation metadata for general documents.
    
    Factory function for creating CitationMetadata instances for general
    documents with breadcrumb-based navigation.
    
    Args:
        citation_id: Sequential citation ID (1, 2, 3, ...)
        doc_id: Document identifier
        chunk_id: Chunk identifier
        workspace: Source workspace name
        file_path: Original file path
        hierarchy_path: Breadcrumb path (e.g., ["Hạ tầng", "Quy trình Backup", "Bước 1"])
        relevance_score: Relevance score (0.0-1.0)
        content_preview: First 100 characters of content
    
    Returns:
        CitationMetadata instance configured for general documents
    
    Example:
        >>> metadata = create_general_citation_metadata(
        ...     citation_id=2,
        ...     doc_id="doc-660e8400-e29b-41d4-a716-446655440001",
        ...     chunk_id="chunk-5",
        ...     workspace="user-docs",
        ...     file_path="/docs/infrastructure_guide.md",
        ...     hierarchy_path=["Hạ tầng", "Quy trình Backup", "Bước 1"],
        ...     relevance_score=0.88,
        ...     content_preview="Để thực hiện backup hệ thống, trước tiên..."
        ... )
        >>> metadata.formatted_ref
        'Hạ tầng > Quy trình Backup > Bước 1 (Tài liệu infrastructure_guide)'
    """
    # Build formatted reference from hierarchy and file name
    doc_name = Path(file_path).stem
    
    if hierarchy_path:
        section = " > ".join(hierarchy_path)
        formatted_ref = f"{section} (Tài liệu {doc_name})"
    else:
        formatted_ref = f"Tài liệu {doc_name}"
    
    return CitationMetadata(
        citation_id=citation_id,
        doc_id=doc_id,
        chunk_id=chunk_id,
        doc_type="GENERAL",
        source_type="chunk",
        workspace=workspace,
        file_path=file_path,
        hierarchy_path=hierarchy_path,
        relevance_score=relevance_score,
        content_preview=content_preview,
        legal_info=None,
        formatted_ref=formatted_ref,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )


class CitationTracker:
    """Tracks citations during query execution with O(1) deduplication.
    
    This class maintains an in-memory store of all retrieved sources (chunks,
    entities, relations) during a query, assigns sequential citation IDs, and
    provides deduplication to ensure the same source gets the same ID.
    
    The tracker uses dictionaries for O(1) lookup and deduplication:
    - Chunks are deduplicated by (doc_id, chunk_id) tuple
    - Entities are deduplicated by (entity_name, workspace) tuple
    - Relations are deduplicated by (relation_key, workspace) tuple
    
    Attributes:
        _citations: Dictionary mapping citation_id to CitationMetadata
        _chunk_index: Dictionary mapping (doc_id, chunk_id) to citation_id
        _entity_index: Dictionary mapping (entity_name, workspace) to citation_id
        _relation_index: Dictionary mapping (relation_key, workspace) to citation_id
        _next_id: Counter for sequential citation ID assignment
    
    Example:
        >>> tracker = CitationTracker()
        >>> 
        >>> # Track a chunk
        >>> chunk_data = {
        ...     "doc_id": "doc-123",
        ...     "chunk_id": "chunk-1",
        ...     "workspace": "legal-vn",
        ...     "content": "Thuế suất là 10%...",
        ...     "metadata": {"doc_type": "legal", ...}
        ... }
        >>> citation = tracker.track_chunk(chunk_data)
        >>> citation.citation_id
        1
        >>> 
        >>> # Track the same chunk again - returns same ID
        >>> citation2 = tracker.track_chunk(chunk_data)
        >>> citation2.citation_id
        1
        >>> 
        >>> # Get all citations
        >>> all_citations = tracker.get_all_citations()
        >>> len(all_citations)
        1
    """
    
    def __init__(self):
        """Initialize an empty citation tracker."""
        # Main storage: citation_id -> CitationMetadata
        self._citations: Dict[int, CitationMetadata] = {}
        
        # Deduplication indexes for O(1) lookup
        self._chunk_index: Dict[tuple, int] = {}  # (doc_id, chunk_id) -> citation_id
        self._entity_index: Dict[tuple, int] = {}  # (entity_name, workspace) -> citation_id
        self._relation_index: Dict[tuple, int] = {}  # (relation_key, workspace) -> citation_id
        
        # Sequential ID counter
        self._next_id: int = 1
    
    def track_chunk(self, chunk_data: Dict[str, Any]) -> CitationMetadata:
        """Track a retrieved chunk and assign citation ID.
        
        If the chunk has already been tracked (same doc_id + chunk_id), returns
        the existing CitationMetadata with the same citation_id. Otherwise,
        creates a new CitationMetadata with a new sequential ID.
        
        Args:
            chunk_data: Dictionary containing chunk information:
                - doc_id: Document identifier (required)
                - chunk_id: Chunk identifier (required)
                - workspace: Workspace name (required)
                - content: Chunk content (required)
                - metadata: Dictionary with additional metadata:
                    - doc_type: "legal" or "general" (default: "general")
                    - file_path: Original file path
                    - hierarchy_path: List of hierarchy elements
                    - legal_info: Legal metadata (for legal documents)
                    - relevance_score: Relevance score (default: 0.0)
        
        Returns:
            CitationMetadata instance for this chunk
        
        Example:
            >>> tracker = CitationTracker()
            >>> chunk = {
            ...     "doc_id": "doc-123",
            ...     "chunk_id": "chunk-1",
            ...     "workspace": "legal-vn",
            ...     "content": "Thuế suất là 10%...",
            ...     "metadata": {
            ...         "doc_type": "legal",
            ...         "file_path": "/legal/luat_thue.pdf",
            ...         "hierarchy_path": ["Chương II", "Điều 8"],
            ...         "legal_info": {"article": "Điều 8", ...}
            ...     }
            ... }
            >>> citation = tracker.track_chunk(chunk)
            >>> citation.citation_id
            1
        """
        doc_id = chunk_data.get("doc_id", "")
        chunk_id = chunk_data.get("chunk_id", "")
        
        # Check if already tracked (deduplication)
        dedup_key = (doc_id, chunk_id)
        if dedup_key in self._chunk_index:
            existing_id = self._chunk_index[dedup_key]
            return self._citations[existing_id]
        
        # Extract metadata
        metadata = chunk_data.get("metadata", {})
        workspace = chunk_data.get("workspace", "")
        content = chunk_data.get("content", "")
        
        # Determine document type
        doc_type = metadata.get("doc_type", "general").upper()
        
        # Extract common fields
        file_path = metadata.get("file_path", "")
        hierarchy_path = metadata.get("hierarchy_path", [])
        relevance_score = metadata.get("relevance_score", 0.0)
        content_preview = content[:100] if content else ""
        
        # Create citation metadata based on document type
        if doc_type == "LEGAL":
            legal_info = metadata.get("legal_info", {})
            citation = create_legal_citation_metadata(
                citation_id=self._next_id,
                doc_id=doc_id,
                chunk_id=chunk_id,
                workspace=workspace,
                file_path=file_path,
                hierarchy_path=hierarchy_path,
                legal_info=legal_info,
                relevance_score=relevance_score,
                content_preview=content_preview,
            )
        else:
            citation = create_general_citation_metadata(
                citation_id=self._next_id,
                doc_id=doc_id,
                chunk_id=chunk_id,
                workspace=workspace,
                file_path=file_path,
                hierarchy_path=hierarchy_path,
                relevance_score=relevance_score,
                content_preview=content_preview,
            )
        
        # Store citation and update indexes
        self._citations[self._next_id] = citation
        self._chunk_index[dedup_key] = self._next_id
        
        # Increment ID counter
        self._next_id += 1
        
        return citation
    
    def track_entity(self, entity_data: Dict[str, Any]) -> CitationMetadata:
        """Track a retrieved entity and assign citation ID.
        
        If the entity has already been tracked (same entity_name + workspace),
        returns the existing CitationMetadata. Otherwise, creates a new one.
        
        Args:
            entity_data: Dictionary containing entity information:
                - entity_name: Entity name (required)
                - workspace: Workspace name (required)
                - description: Entity description (optional)
                - metadata: Additional metadata (optional)
        
        Returns:
            CitationMetadata instance for this entity
        
        Example:
            >>> tracker = CitationTracker()
            >>> entity = {
            ...     "entity_name": "Thuế GTGT",
            ...     "workspace": "legal-vn",
            ...     "description": "Thuế giá trị gia tăng...",
            ...     "metadata": {"relevance_score": 0.92}
            ... }
            >>> citation = tracker.track_entity(entity)
            >>> citation.source_type
            'entity'
        """
        entity_name = entity_data.get("entity_name", "")
        workspace = entity_data.get("workspace", "")
        
        # Check if already tracked (deduplication)
        dedup_key = (entity_name, workspace)
        if dedup_key in self._entity_index:
            existing_id = self._entity_index[dedup_key]
            return self._citations[existing_id]
        
        # Extract metadata
        metadata = entity_data.get("metadata", {})
        description = entity_data.get("description", "")
        relevance_score = metadata.get("relevance_score", 0.0)
        
        # Create citation metadata for entity
        citation = CitationMetadata(
            citation_id=self._next_id,
            doc_id="",  # Entities don't have doc_id
            chunk_id=None,
            doc_type="GENERAL",
            source_type="entity",
            entity_name=entity_name,
            workspace=workspace,
            file_path="",
            hierarchy_path=[],
            relevance_score=relevance_score,
            content_preview=description[:100] if description else "",
            formatted_ref=f"Entity: {entity_name}",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
        
        # Store citation and update indexes
        self._citations[self._next_id] = citation
        self._entity_index[dedup_key] = self._next_id
        
        # Increment ID counter
        self._next_id += 1
        
        return citation
    
    def track_relation(self, relation_data: Dict[str, Any]) -> CitationMetadata:
        """Track a retrieved relation and assign citation ID.
        
        If the relation has already been tracked (same relation_key + workspace),
        returns the existing CitationMetadata. Otherwise, creates a new one.
        
        Args:
            relation_data: Dictionary containing relation information:
                - relation_key: Relation key (required)
                - workspace: Workspace name (required)
                - description: Relation description (optional)
                - metadata: Additional metadata (optional)
        
        Returns:
            CitationMetadata instance for this relation
        
        Example:
            >>> tracker = CitationTracker()
            >>> relation = {
            ...     "relation_key": "Thuế GTGT -> áp dụng -> Hàng hóa",
            ...     "workspace": "legal-vn",
            ...     "description": "Thuế GTGT áp dụng cho hàng hóa...",
            ...     "metadata": {"relevance_score": 0.85}
            ... }
            >>> citation = tracker.track_relation(relation)
            >>> citation.source_type
            'relation'
        """
        relation_key = relation_data.get("relation_key", "")
        workspace = relation_data.get("workspace", "")
        
        # Check if already tracked (deduplication)
        dedup_key = (relation_key, workspace)
        if dedup_key in self._relation_index:
            existing_id = self._relation_index[dedup_key]
            return self._citations[existing_id]
        
        # Extract metadata
        metadata = relation_data.get("metadata", {})
        description = relation_data.get("description", "")
        relevance_score = metadata.get("relevance_score", 0.0)
        
        # Create citation metadata for relation
        citation = CitationMetadata(
            citation_id=self._next_id,
            doc_id="",  # Relations don't have doc_id
            chunk_id=None,
            doc_type="GENERAL",
            source_type="relation",
            relation_key=relation_key,
            workspace=workspace,
            file_path="",
            hierarchy_path=[],
            relevance_score=relevance_score,
            content_preview=description[:100] if description else "",
            formatted_ref=f"Relation: {relation_key}",
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
        
        # Store citation and update indexes
        self._citations[self._next_id] = citation
        self._relation_index[dedup_key] = self._next_id
        
        # Increment ID counter
        self._next_id += 1
        
        return citation
    
    def get_citation(self, citation_id: int) -> Optional[CitationMetadata]:
        """Retrieve citation metadata by citation ID.
        
        Args:
            citation_id: Citation ID to retrieve
        
        Returns:
            CitationMetadata if found, None otherwise
        
        Example:
            >>> tracker = CitationTracker()
            >>> chunk = {"doc_id": "doc-123", "chunk_id": "chunk-1", ...}
            >>> citation = tracker.track_chunk(chunk)
            >>> retrieved = tracker.get_citation(1)
            >>> retrieved.citation_id
            1
        """
        return self._citations.get(citation_id)
    
    def get_all_citations(self) -> List[CitationMetadata]:
        """Retrieve all tracked citations in order of citation ID.
        
        Returns:
            List of CitationMetadata instances sorted by citation_id
        
        Example:
            >>> tracker = CitationTracker()
            >>> tracker.track_chunk({"doc_id": "doc-1", "chunk_id": "chunk-1", ...})
            >>> tracker.track_chunk({"doc_id": "doc-2", "chunk_id": "chunk-2", ...})
            >>> citations = tracker.get_all_citations()
            >>> len(citations)
            2
            >>> [c.citation_id for c in citations]
            [1, 2]
        """
        return [self._citations[cid] for cid in sorted(self._citations.keys())]
    
    def get_all_citation_ids(self) -> List[int]:
        """Retrieve all citation IDs currently tracked.
        
        Returns:
            List of citation IDs sorted in ascending order
        
        Example:
            >>> tracker = CitationTracker()
            >>> tracker.track_chunk({"doc_id": "doc-1", "chunk_id": "chunk-1", ...})
            >>> tracker.track_entity({"entity_name": "Entity1", "workspace": "ws1"})
            >>> tracker.get_all_citation_ids()
            [1, 2]
        """
        return sorted(self._citations.keys())
    
    def get_citation_count(self) -> int:
        """Get the total number of unique citations tracked.
        
        Returns:
            Number of unique citations
        
        Example:
            >>> tracker = CitationTracker()
            >>> tracker.track_chunk({"doc_id": "doc-1", "chunk_id": "chunk-1", ...})
            >>> tracker.track_chunk({"doc_id": "doc-1", "chunk_id": "chunk-1", ...})  # Duplicate
            >>> tracker.get_citation_count()
            1
        """
        return len(self._citations)


def build_context_with_citations(
    chunks: List[Dict[str, Any]],
    citation_tracker: CitationTracker,
) -> str:
    """Build LLM context with citation source labels.
    
    This function wraps each chunk with citation markers and source labels,
    making it easy for the LLM to reference sources in its response. The format
    adapts based on document type (legal vs general) and detects tables.
    
    For each chunk, the function:
    1. Retrieves or creates citation metadata via the tracker
    2. Builds a formatted source reference based on doc_type
    3. Detects if the chunk contains a table
    4. Wraps the content with [ID: N] and [Source: ...] markers
    
    Args:
        chunks: List of chunk dictionaries, each containing:
            - content: Chunk text content (required)
            - doc_id: Document identifier (required)
            - chunk_id: Chunk identifier (required)
            - workspace: Workspace name (required)
            - metadata: Dictionary with doc_type, hierarchy_path, legal_info, etc.
        citation_tracker: CitationTracker instance to track and assign IDs
    
    Returns:
        Formatted context string with citation markers for LLM consumption
    
    Example:
        >>> tracker = CitationTracker()
        >>> chunks = [
        ...     {
        ...         "doc_id": "doc-123",
        ...         "chunk_id": "chunk-1",
        ...         "workspace": "legal-vn",
        ...         "content": "Thuế suất GTGT là 10%...",
        ...         "metadata": {
        ...             "doc_type": "legal",
        ...             "hierarchy_path": ["Chương II", "Điều 8", "Khoản 2"],
        ...             "legal_info": {
        ...                 "article": "Điều 8",
        ...                 "clause": "Khoản 2",
        ...                 "document_type": "Luật",
        ...                 "document_number": "13/2008/QH12"
        ...             }
        ...         }
        ...     }
        ... ]
        >>> context = build_context_with_citations(chunks, tracker)
        >>> print(context)
        [ID: 1]
        [Source: Khoản 2 Điều 8 Luật số 13/2008/QH12]
        Thuế suất GTGT là 10%...
    """
    context_parts = []
    
    for chunk in chunks:
        # Track the chunk and get citation metadata
        citation = citation_tracker.track_chunk(chunk)
        
        # Get the formatted reference
        formatted_ref = citation.formatted_ref
        
        # Detect if chunk contains a table
        content = chunk.get("content", "")
        has_table = _detect_table_in_content(content)
        
        # Add table indicator to source label if table detected
        if has_table:
            if citation.doc_type == "LEGAL":
                formatted_ref += " (Bảng)"
            else:
                formatted_ref += " (Table)"
        
        # Build context block with citation markers
        context_block = f"""[ID: {citation.citation_id}]
[Source: {formatted_ref}]
{content}"""
        
        context_parts.append(context_block)
    
    return "\n\n".join(context_parts)


def _detect_table_in_content(content: str) -> bool:
    """Detect if content contains HTML or Markdown tables.
    
    Args:
        content: Text content to check
    
    Returns:
        True if content contains a table, False otherwise
    
    Example:
        >>> _detect_table_in_content("<table><tr><td>Cell</td></tr></table>")
        True
        >>> _detect_table_in_content("| Header | Header |\\n|--------|--------|")
        True
        >>> _detect_table_in_content("Regular text without tables")
        False
    """
    # Check for HTML tables
    if re.search(r"<table.*?>.*?</table>", content, re.IGNORECASE | re.DOTALL):
        return True
    
    # Check for Markdown tables (at least 2 lines with pipe separators)
    lines = content.split("\n")
    pipe_lines = [line for line in lines if "|" in line]
    
    # Markdown table needs at least header + separator + 1 data row (3 lines minimum)
    if len(pipe_lines) >= 3:
        return True
    
    # Also check for separator line pattern (e.g., |---|---| or |-------|----------|)
    # This handles cases with just 2 lines (header + separator)
    if len(pipe_lines) >= 2:
        for line in pipe_lines:
            # Match lines that are mostly dashes and pipes (separator lines)
            # Pattern: optional whitespace, pipe, then dashes/colons/pipes/spaces, ending with pipe
            if re.match(r"^\s*\|[\s\-:|]+\|\s*$", line):
                return True
    
    return False


def build_citation_instruction(
    doc_type: Optional[Literal["LEGAL", "GENERAL"]] = None,
    language: str = "Vietnamese",
) -> str:
    """Build citation instruction for LLM system prompt.
    
    This function generates detailed instructions for the LLM on how to cite
    sources in its response. The instructions adapt based on document type
    (legal vs general) and language (Vietnamese vs English).
    
    The instructions specify:
    - How to cite legal documents (to article/clause level)
    - How to cite general documents (using section headers)
    - Rules for using exact citation IDs from context
    - Special guidance for citing tables as complete units
    
    Args:
        doc_type: Document type to generate instructions for:
            - "LEGAL": Vietnamese legal document instructions
            - "GENERAL": General document instructions
            - None: Instructions for both types (default)
        language: Language for instructions ("Vietnamese" or "English")
    
    Returns:
        Formatted citation instruction string for system prompt
    
    Example:
        >>> instruction = build_citation_instruction(doc_type="LEGAL", language="Vietnamese")
        >>> "Điều" in instruction
        True
        >>> "Khoản" in instruction
        True
    """
    if language == "Vietnamese":
        return _build_vietnamese_citation_instruction(doc_type)
    else:
        return _build_english_citation_instruction(doc_type)


def _build_vietnamese_citation_instruction(
    doc_type: Optional[Literal["LEGAL", "GENERAL"]] = None,
) -> str:
    """Build Vietnamese citation instructions.
    
    Args:
        doc_type: Document type ("LEGAL", "GENERAL", or None for both)
    
    Returns:
        Vietnamese citation instruction string
    """
    if doc_type == "LEGAL":
        return """
## Hướng dẫn Trích dẫn - Tài liệu Pháp luật

Khi tạo câu trả lời, bạn PHẢI trích dẫn nguồn cho mọi thông tin được lấy từ ngữ cảnh:

**Quy tắc trích dẫn tài liệu pháp luật:**
- Trích dẫn chi tiết đến cấp thấp nhất có thể (Điều, Khoản, Điểm)
- Sử dụng số ID được cung cấp trong [ID: N] của ngữ cảnh
- Sử dụng nguồn được cung cấp trong [Source: ...]
- Format: Nội dung câu trả lời[ID]
- Ví dụ: "Thuế suất GTGT là 10%[1]"

**Quy tắc trích dẫn bảng:**
- Khi trích dẫn thông tin từ bảng, trích dẫn toàn bộ bảng với một ID duy nhất
- KHÔNG tách bảng thành nhiều trích dẫn
- Ví dụ: "Theo bảng thuế suất[2], mức thuế áp dụng là..."

**Quy tắc chung:**
- Sử dụng ĐÚNG số ID được cung cấp trong ngữ cảnh
- KHÔNG tự tạo số ID mới
- Đặt [ID] ngay sau câu/cụm từ được trích dẫn
- Có thể kết hợp nhiều ID: [1,2,3]

Ví dụ đầy đủ:
"Theo quy định pháp luật, thuế GTGT được áp dụng với mức 10%[1] cho hầu hết hàng hóa và dịch vụ. 
Các trường hợp miễn thuế được quy định tại Điều 5[2]."
"""
    
    elif doc_type == "GENERAL":
        return """
## Hướng dẫn Trích dẫn - Tài liệu Thông thường

Khi tạo câu trả lời, bạn PHẢI trích dẫn nguồn cho mọi thông tin được lấy từ ngữ cảnh:

**Quy tắc trích dẫn tài liệu thông thường:**
- Sử dụng tiêu đề mục (Header) để dẫn nguồn
- Sử dụng số ID được cung cấp trong [ID: N] của ngữ cảnh
- Sử dụng nguồn được cung cấp trong [Source: ...]
- Format: Nội dung câu trả lời[ID]
- Ví dụ: "Bước đầu tiên là kiểm tra dung lượng[1]"

**Quy tắc trích dẫn bảng:**
- Khi trích dẫn thông tin từ bảng, trích dẫn toàn bộ bảng với một ID duy nhất
- KHÔNG tách bảng thành nhiều trích dẫn
- Ví dụ: "Theo bảng cấu hình[2], RAM tối thiểu là 8GB..."

**Quy tắc chung:**
- Sử dụng ĐÚNG số ID được cung cấp trong ngữ cảnh
- KHÔNG tự tạo số ID mới
- Đặt [ID] ngay sau câu/cụm từ được trích dẫn
- Có thể kết hợp nhiều ID: [1,2,3]

Ví dụ đầy đủ:
"Quy trình backup bao gồm ba bước chính[1]. Đầu tiên, kiểm tra dung lượng ổ đĩa[2]. 
Sau đó, chạy script backup tự động[3]."
"""
    
    else:  # Both types
        return """
## Hướng dẫn Trích dẫn

Khi tạo câu trả lời, bạn PHẢI trích dẫn nguồn cho mọi thông tin được lấy từ ngữ cảnh:

**1. Tài liệu Pháp luật (doc_type: LEGAL):**
   - Trích dẫn chi tiết đến cấp thấp nhất có thể (Điều, Khoản, Điểm)
   - Sử dụng số ID và nguồn được cung cấp trong [ID: N] và [Source: ...]
   - Format: Nội dung câu trả lời[ID]
   - Ví dụ: "Thuế suất GTGT là 10%[1]"

**2. Tài liệu Thông thường (doc_type: GENERAL):**
   - Sử dụng tiêu đề mục (Header) để dẫn nguồn
   - Sử dụng số ID và nguồn được cung cấp trong [ID: N] và [Source: ...]
   - Format: Nội dung câu trả lời[ID]
   - Ví dụ: "Bước đầu tiên là kiểm tra dung lượng[1]"

**3. Quy tắc trích dẫn bảng:**
   - Khi trích dẫn thông tin từ bảng, trích dẫn toàn bộ bảng với một ID duy nhất
   - KHÔNG tách bảng thành nhiều trích dẫn
   - Ví dụ: "Theo bảng thuế suất[2], mức thuế áp dụng là..."

**4. Quy tắc chung:**
   - Sử dụng ĐÚNG số ID được cung cấp trong ngữ cảnh
   - KHÔNG tự tạo số ID mới
   - Đặt [ID] ngay sau câu/cụm từ được trích dẫn
   - Có thể kết hợp nhiều ID: [1,2,3]

Ví dụ đầy đủ:
"Theo quy định pháp luật, thuế GTGT được áp dụng với mức 10%[1] cho hầu hết hàng hóa và dịch vụ. 
Quy trình kê khai được mô tả chi tiết trong tài liệu hướng dẫn[2]."
"""


def _build_english_citation_instruction(
    doc_type: Optional[Literal["LEGAL", "GENERAL"]] = None,
) -> str:
    """Build English citation instructions.
    
    Args:
        doc_type: Document type ("LEGAL", "GENERAL", or None for both)
    
    Returns:
        English citation instruction string
    """
    if doc_type == "LEGAL":
        return """
## Citation Instructions - Legal Documents

When generating your response, you MUST cite sources for all information from the context:

**Legal document citation rules:**
- Cite to the most specific level possible (Article, Clause, Point)
- Use the ID provided in [ID: N] from the context
- Use the source provided in [Source: ...]
- Format: Response content[ID]
- Example: "VAT rate is 10%[1]"

**Table citation rules:**
- When citing information from a table, cite the entire table with a single ID
- DO NOT split tables into multiple citations
- Example: "According to the tax rate table[2], the applicable rate is..."

**General rules:**
- Use EXACT IDs provided in the context
- DO NOT create new ID numbers
- Place [ID] immediately after the cited statement
- Can combine multiple IDs: [1,2,3]

Full example:
"According to regulations, VAT is applied at 10%[1] for most goods and services. 
Exemption cases are specified in Article 5[2]."
"""
    
    elif doc_type == "GENERAL":
        return """
## Citation Instructions - General Documents

When generating your response, you MUST cite sources for all information from the context:

**General document citation rules:**
- Use section headers for source attribution
- Use the ID provided in [ID: N] from the context
- Use the source provided in [Source: ...]
- Format: Response content[ID]
- Example: "First step is to check capacity[1]"

**Table citation rules:**
- When citing information from a table, cite the entire table with a single ID
- DO NOT split tables into multiple citations
- Example: "According to the configuration table[2], minimum RAM is 8GB..."

**General rules:**
- Use EXACT IDs provided in the context
- DO NOT create new ID numbers
- Place [ID] immediately after the cited statement
- Can combine multiple IDs: [1,2,3]

Full example:
"The backup procedure includes three main steps[1]. First, check disk capacity[2]. 
Then, run the automated backup script[3]."
"""
    
    else:  # Both types
        return """
## Citation Instructions

When generating your response, you MUST cite sources for all information from the context:

**1. Legal Documents (doc_type: LEGAL):**
   - Cite to the most specific level possible (Article, Clause, Point)
   - Use the ID and source provided in [ID: N] and [Source: ...]
   - Format: Response content[ID]
   - Example: "VAT rate is 10%[1]"

**2. General Documents (doc_type: GENERAL):**
   - Use section headers for source attribution
   - Use the ID and source provided in [ID: N] and [Source: ...]
   - Format: Response content[ID]
   - Example: "First step is to check capacity[1]"

**3. Table citation rules:**
   - When citing information from a table, cite the entire table with a single ID
   - DO NOT split tables into multiple citations
   - Example: "According to the tax rate table[2], the applicable rate is..."

**4. General rules:**
   - Use EXACT IDs provided in the context
   - DO NOT create new ID numbers
   - Place [ID] immediately after the cited statement
   - Can combine multiple IDs: [1,2,3]

Full example:
"According to regulations, VAT is applied at 10%[1] for most goods and services. 
The filing procedure is detailed in the guide[2]."
"""


def append_citation_instruction_to_prompt(
    system_prompt: str,
    enable_citations: bool = False,
    doc_type: Optional[Literal["LEGAL", "GENERAL"]] = None,
    language: str = "Vietnamese",
) -> str:
    """Append citation instructions to system prompt when citations are enabled.
    
    This is a convenience function for integrating citation instructions into
    existing system prompts. It appends the citation instruction at the end
    of the system prompt.
    
    Args:
        system_prompt: Original system prompt
        enable_citations: Whether to append citation instructions
        doc_type: Document type for specialized instructions (optional)
        language: Language for instructions ("Vietnamese" or "English")
    
    Returns:
        System prompt with citation instructions appended (if enabled)
    
    Example:
        >>> original_prompt = "You are a helpful assistant."
        >>> enhanced_prompt = append_citation_instruction_to_prompt(
        ...     original_prompt,
        ...     enable_citations=True,
        ...     language="Vietnamese"
        ... )
        >>> "Hướng dẫn Trích dẫn" in enhanced_prompt
        True
    """
    if not enable_citations:
        return system_prompt
    
    citation_instruction = build_citation_instruction(doc_type, language)
    
    # Append citation instruction to system prompt
    return f"{system_prompt}\n\n{citation_instruction}"


# ============================================================================
# Citation Validation and Sanitization
# ============================================================================

def validate_citation_ids(
    llm_response: str,
    citation_tracker: CitationTracker,
) -> tuple[bool, List[int]]:
    """Validate that all citation IDs in LLM response exist in tracker.
    
    This function checks that the LLM only used citation IDs that were actually
    provided in the context. This helps detect hallucinated citations where the
    LLM invents citation numbers that don't correspond to real sources.
    
    Implements error handling (Requirement 15.1, 15.5):
    - Detects invalid citation IDs
    - Logs warning with citation ID and response context
    - Returns list of invalid IDs for sanitization
    
    Args:
        llm_response: LLM response text with citation markers
        citation_tracker: CitationTracker instance with tracked citations
    
    Returns:
        Tuple of (is_valid, invalid_ids):
            - is_valid: True if all citation IDs are valid, False otherwise
            - invalid_ids: List of citation IDs that don't exist in tracker
    
    Example:
        >>> tracker = CitationTracker()
        >>> tracker.track_chunk({"doc_id": "doc-1", "chunk_id": "chunk-1", ...})
        >>> response = "This is true[1] and this is false[99]"
        >>> is_valid, invalid = validate_citation_ids(response, tracker)
        >>> is_valid
        False
        >>> invalid
        [99]
    """
    # Extract all citation IDs from response
    pattern = r'\[(\d+(?:,\d+)*)\]'
    used_ids = set()
    for match in re.finditer(pattern, llm_response):
        ids = [int(x) for x in match.group(1).split(',')]
        used_ids.update(ids)
    
    # Get valid citation IDs from tracker
    valid_ids = set(citation_tracker.get_all_citation_ids())
    
    # Find invalid IDs (used but not tracked)
    invalid_ids = used_ids - valid_ids
    
    if invalid_ids:
        # Log warning with citation ID and query context (Requirement 15.5)
        from lightrag.utils import logger
        
        # Extract context around invalid citations for better debugging
        context_snippets = []
        for invalid_id in sorted(invalid_ids):
            # Find occurrences of this invalid ID in the response
            invalid_pattern = rf'\[{invalid_id}(?:,\d+)*\]|\[\d+(?:,{invalid_id})+\]'
            for match in re.finditer(invalid_pattern, llm_response):
                # Get 50 characters before and after the citation
                start = max(0, match.start() - 50)
                end = min(len(llm_response), match.end() + 50)
                snippet = llm_response[start:end].replace('\n', ' ')
                context_snippets.append(f"  ID {invalid_id}: ...{snippet}...")
        
        logger.warning(
            f"Citation validation failed: LLM used invalid citation IDs {sorted(invalid_ids)}. "
            f"Valid IDs: {sorted(valid_ids)}. "
            f"Context:\n" + "\n".join(context_snippets[:5])  # Limit to 5 snippets
        )
        return False, sorted(list(invalid_ids))
    
    return True, []


def sanitize_invalid_citations(
    llm_response: str,
    invalid_ids: List[int],
) -> str:
    """Replace invalid citation IDs with [?] marker.
    
    This function sanitizes the LLM response by replacing citation markers
    that contain invalid IDs with [?] to indicate uncertain sources.
    
    Implements error handling (Requirement 15.1, 15.5):
    - Replaces [N] with [?] when citation ID doesn't exist
    - Logs warning for each replacement
    - Continues rendering remaining valid citations
    
    Args:
        llm_response: LLM response text with citation markers
        invalid_ids: List of invalid citation IDs to replace
    
    Returns:
        Sanitized response with invalid citations replaced
    
    Example:
        >>> response = "This is true[1] and this is false[99]"
        >>> sanitized = sanitize_invalid_citations(response, [99])
        >>> sanitized
        'This is true[1] and this is false[?]'
        >>> 
        >>> # Mixed valid/invalid IDs
        >>> response = "Text[1,99,2]"
        >>> sanitized = sanitize_invalid_citations(response, [99])
        >>> sanitized
        'Text[1,2]'
    """
    if not invalid_ids:
        return llm_response
    
    invalid_set = set(invalid_ids)
    replacement_count = 0
    
    def replace_invalid(match):
        nonlocal replacement_count
        ids = [int(x) for x in match.group(1).split(',')]
        # Filter out invalid IDs
        valid_ids = [id for id in ids if id not in invalid_set]
        
        if not valid_ids:
            # All IDs were invalid, replace with [?] (Requirement 15.1)
            replacement_count += 1
            return "[?]"
        else:
            # Some IDs were valid, keep only those
            # Count how many were removed
            removed_count = len(ids) - len(valid_ids)
            if removed_count > 0:
                replacement_count += removed_count
            return f"[{','.join(map(str, valid_ids))}]"
    
    pattern = r'\[(\d+(?:,\d+)*)\]'
    sanitized = re.sub(pattern, replace_invalid, llm_response)
    
    # Log sanitization summary (Requirement 15.5)
    if replacement_count > 0:
        from lightrag.utils import logger
        logger.warning(
            f"Sanitized {replacement_count} invalid citation reference(s). "
            f"Invalid IDs: {sorted(invalid_ids)}. "
            f"Replaced with [?] or removed from mixed citations."
        )
    
    return sanitized


# ============================================================================
# Citation Aggregation
# ============================================================================

def aggregate_citation_ids(citation_ids: List[int]) -> str:
    """Aggregate consecutive citation IDs into ranges.
    
    This function converts lists of consecutive citation IDs into compact
    range notation for better readability. For example, [1,2,3,4,5] becomes
    [1-5], while [1,3,5] stays as [1][3][5].
    
    Args:
        citation_ids: List of citation IDs (e.g., [1, 2, 3, 5, 7, 8, 9])
    
    Returns:
        Aggregated string (e.g., "[1-3][5][7-9]")
    
    Examples:
        >>> aggregate_citation_ids([1, 2, 3, 4, 5])
        '[1-5]'
        >>> aggregate_citation_ids([1, 3, 5])
        '[1][3][5]'
        >>> aggregate_citation_ids([1, 2, 3, 5, 7, 8])
        '[1-3][5][7-8]'
        >>> aggregate_citation_ids([1])
        '[1]'
        >>> aggregate_citation_ids([])
        ''
    """
    if not citation_ids:
        return ""
    
    # Sort and deduplicate IDs
    sorted_ids = sorted(set(citation_ids))
    
    # Group consecutive IDs into ranges
    ranges = []
    start = sorted_ids[0]
    end = sorted_ids[0]
    
    for i in range(1, len(sorted_ids)):
        if sorted_ids[i] == end + 1:
            # Consecutive, extend range
            end = sorted_ids[i]
        else:
            # Non-consecutive, save current range and start new one
            if start == end:
                ranges.append(f"[{start}]")
            else:
                ranges.append(f"[{start}-{end}]")
            start = sorted_ids[i]
            end = sorted_ids[i]
    
    # Save last range
    if start == end:
        ranges.append(f"[{start}]")
    else:
        ranges.append(f"[{start}-{end}]")
    
    return "".join(ranges)


# ============================================================================
# Citation Renderer
# ============================================================================

class CitationRenderer:
    """Renders citations in specified format with validation and aggregation.
    
    This class takes an LLM response with citation markers and renders them
    according to the specified format (inline, footnote, or bibliography).
    It also performs validation to detect hallucinated citations and aggregates
    consecutive citation IDs for better readability.
    
    The rendering process:
    1. Validates that all citation IDs in the response exist in the tracker
    2. Sanitizes invalid citations by replacing them with [?]
    3. Aggregates consecutive citation markers (e.g., [1][2][3] -> [1-3])
    4. Renders citations according to the specified format
    5. Returns formatted response and CitationList metadata
    
    Attributes:
        tracker: CitationTracker instance with tracked citations
        format: Citation format ("inline", "footnote", or "bibliography")
        language: Language for formatting ("Vietnamese" or "English")
    
    Example:
        >>> tracker = CitationTracker()
        >>> tracker.track_chunk({"doc_id": "doc-1", "chunk_id": "chunk-1", ...})
        >>> renderer = CitationRenderer(tracker, format="footnote", language="Vietnamese")
        >>> response = "Thuế GTGT là 10%[1]"
        >>> formatted, citation_list = renderer.render(response)
        >>> "---" in formatted  # Footnote section added
        True
    """
    
    def __init__(
        self,
        citation_tracker: CitationTracker,
        format: Literal["inline", "footnote", "bibliography"] = "footnote",
        language: str = "Vietnamese",
        citation_order: Literal["relevance", "appearance"] = "relevance",
        metrics_tracker: Optional["CitationMetricsTracker"] = None,
    ):
        """Initialize citation renderer.
        
        Args:
            citation_tracker: CitationTracker instance with tracked citations
            format: Citation format ("inline", "footnote", or "bibliography")
            language: Language for formatting ("Vietnamese" or "English")
            citation_order: Citation ordering strategy ("relevance" or "appearance")
                - "relevance": Order by relevance_score (descending) in bibliography
                - "appearance": Order by first appearance in response text
            metrics_tracker: Optional CitationMetricsTracker for observability
        """
        self.tracker = citation_tracker
        self.format = format
        self.language = language
        self.citation_order = citation_order
        self.metrics_tracker = metrics_tracker
    
    def render(self, llm_response: str) -> tuple[str, CitationList]:
        """Render citations in LLM response with validation and aggregation.
        
        This is the main entry point for citation rendering. It performs
        validation, sanitization, aggregation, and format-specific rendering.
        
        Args:
            llm_response: LLM response text with citation markers
        
        Returns:
            Tuple of (formatted_response, citation_list):
                - formatted_response: Response with rendered citations
                - citation_list: CitationList with metadata
        
        Example:
            >>> tracker = CitationTracker()
            >>> tracker.track_chunk({"doc_id": "doc-1", "chunk_id": "chunk-1", ...})
            >>> renderer = CitationRenderer(tracker, format="inline")
            >>> response = "This is true[1]"
            >>> formatted, citations = renderer.render(response)
            >>> "[1:" in formatted  # Inline format
            True
        """
        # Step 1: Validate citation IDs
        is_valid, invalid_ids = validate_citation_ids(llm_response, self.tracker)
        
        # Track validation metrics if metrics tracker is available
        if self.metrics_tracker:
            # Extract all citation IDs from response
            pattern = r'\[(\d+(?:,\d+)*)\]'
            used_ids = set()
            for match in re.finditer(pattern, llm_response):
                ids = [int(x) for x in match.group(1).split(',')]
                used_ids.update(ids)
            
            total_citations = len(used_ids)
            valid_citations = total_citations - len(invalid_ids)
            
            # Get context items count from tracker
            context_items = len(self.tracker.get_all_citations())
            cited_items = len(used_ids)
            
            # Determine failure type if validation failed
            failure_type = "hallucinated_id" if invalid_ids else None
            
            # Track validation metrics
            self.metrics_tracker.track_validation(
                total_citations=total_citations,
                valid_citations=valid_citations,
                invalid_citations=len(invalid_ids),
                context_items=context_items,
                cited_items=cited_items,
                failure_type=failure_type,
            )
        
        if not is_valid:
            from lightrag.utils import logger
            logger.warning(
                f"Citation validation failed. Invalid IDs: {invalid_ids}. "
                f"Replacing with [?] markers."
            )
            llm_response = sanitize_invalid_citations(llm_response, invalid_ids)
        
        # Step 2: Aggregate consecutive citation markers
        llm_response = self._aggregate_citation_markers(llm_response)
        
        # Step 3: Render citations according to format
        if self.format == "inline":
            return self._render_inline(llm_response)
        elif self.format == "footnote":
            return self._render_footnote(llm_response)
        elif self.format == "bibliography":
            return self._render_bibliography(llm_response)
        else:
            # Fallback to footnote format
            return self._render_footnote(llm_response)
    
    def _aggregate_citation_markers(self, response: str) -> str:
        """Aggregate consecutive citation markers in response.
        
        This method finds patterns like [1][2][3] and converts them to [1-3]
        for better readability.
        
        Args:
            response: LLM response with citation markers like [1][2][3][5]
        
        Returns:
            Response with aggregated markers like [1-3][5]
        
        Example:
            >>> renderer = CitationRenderer(CitationTracker(), "inline")
            >>> response = "Text[1][2][3] more text[5][6]"
            >>> renderer._aggregate_citation_markers(response)
            'Text[1-3] more text[5-6]'
        """
        # Pattern to match consecutive citation markers: [1][2][3]
        pattern = r'(\[\d+\])+'
        
        def aggregate_match(match):
            # Extract all IDs from consecutive markers
            marker_text = match.group(0)
            ids = [int(x) for x in re.findall(r'\[(\d+)\]', marker_text)]
            
            # Aggregate into ranges
            return aggregate_citation_ids(ids)
        
        aggregated = re.sub(pattern, aggregate_match, response)
        return aggregated
    
    def _render_inline(self, response: str) -> tuple[str, CitationList]:
        """Render citations in inline format.
        
        Inline format replaces [N] with [N: Title, Chunk X, Workspace: Y] directly in the text.
        Workspace information is included for clear attribution.
        
        Implements error handling (Requirement 15.1, 15.5):
        - Replaces invalid citation IDs with [?]
        - Logs warnings for missing citations
        - Continues rendering remaining valid citations
        
        Args:
            response: LLM response with citation markers
        
        Returns:
            Tuple of (formatted_response, citation_list)
        
        Example:
            >>> # Input: "Thuế GTGT là 10%[1]"
            >>> # Output: "Thuế GTGT là 10%[1: Luật Thuế GTGT, Điều 8, Workspace: legal-vn]"
        """
        # Pattern to match citation markers including ranges: [1], [1-3], [1,2,3]
        pattern = r'\[(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)\]'
        
        def replace_marker(match):
            marker = match.group(1)
            
            # Parse marker to get individual IDs
            ids = self._parse_citation_marker(marker)
            
            # Get citations for these IDs
            citations = []
            missing_ids = []
            for id in ids:
                cit = self.tracker.get_citation(id)
                if cit:
                    citations.append(cit)
                else:
                    missing_ids.append(id)
            
            # Log warning for missing citations (Requirement 15.5)
            if missing_ids:
                from lightrag.utils import logger
                logger.warning(
                    f"Citation IDs {missing_ids} not found in tracker during inline rendering. "
                    f"Marker: [{marker}]"
                )
            
            # If no valid citations found, return [?] (Requirement 15.1)
            if not citations:
                return "[?]"
            
            # Format: [1: Document Title, Chunk X, Workspace: Y; 2: Another Doc, Chunk Y, Workspace: Z]
            formatted_parts = []
            for cit in citations:
                title = self._get_document_title(cit)
                chunk_ref = self._get_chunk_reference(cit)
                workspace = cit.workspace or "Unknown"
                formatted_parts.append(f"{cit.citation_id}: {title}, {chunk_ref}, Workspace: {workspace}")
            
            return f"[{'; '.join(formatted_parts)}]"
        
        rendered_response = re.sub(pattern, replace_marker, response)
        citation_list = self._build_citation_list()
        
        return rendered_response, citation_list
    
    def _render_footnote(self, response: str) -> tuple[str, CitationList]:
        """Render citations in footnote format.
        
        Footnote format keeps [N] in text and adds footnotes at the bottom.
        Workspace information is clearly labeled for each citation.
        
        Args:
            response: LLM response with citation markers
        
        Returns:
            Tuple of (formatted_response, citation_list)
        
        Example:
            >>> # Input: "Thuế GTGT là 10%[1]"
            >>> # Output:
            >>> # "Thuế GTGT là 10%[1]
            >>> #
            >>> # ---
            >>> # ## Chú thích
            >>> # [1] Luật Thuế GTGT, Điều 8, Workspace: legal-vn, Score: 0.95"
        """
        # Extract all citation IDs used in response
        pattern = r'\[(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)\]'
        used_ids = set()
        for match in re.finditer(pattern, response):
            marker = match.group(1)
            ids = self._parse_citation_marker(marker)
            used_ids.update(ids)
        
        # Build footnote section
        if self.language == "Vietnamese":
            footnotes = ["\n\n---\n## Chú thích\n"]
        else:
            footnotes = ["\n\n---\n## Footnotes\n"]
        
        for id in sorted(used_ids):
            cit = self.tracker.get_citation(id)
            if cit:
                footnote = self._format_footnote(cit)
                footnotes.append(footnote)
        
        rendered_response = response + "\n".join(footnotes)
        citation_list = self._build_citation_list()
        
        return rendered_response, citation_list
    
    def _render_bibliography(self, response: str) -> tuple[str, CitationList]:
        """Render citations in bibliography format.
        
        Bibliography format keeps [N] in text and adds full bibliography at end.
        When multiple workspaces are queried, citations are grouped by workspace
        for better organization and attribution.
        
        Citations can be ordered by:
        - "relevance": Descending relevance_score (highest first)
        - "appearance": Order of first appearance in response text
        
        Args:
            response: LLM response with citation markers
        
        Returns:
            Tuple of (formatted_response, citation_list)
        
        Example:
            >>> # Input: "Thuế GTGT là 10%[1]"
            >>> # Output:
            >>> # "Thuế GTGT là 10%[1]
            >>> #
            >>> # ---
            >>> # ## Tài liệu tham khảo
            >>> # [1] Luật Thuế GTGT số 13/2008/QH12
            >>> #     - Loại: Luật
            >>> #     - Cơ quan: Quốc hội
            >>> #     ..."
        """
        # Extract all citation IDs used in response (preserving order of appearance)
        pattern = r'\[(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)\]'
        used_ids_ordered = []  # Preserves appearance order
        used_ids_set = set()
        
        for match in re.finditer(pattern, response):
            marker = match.group(1)
            ids = self._parse_citation_marker(marker)
            for id in ids:
                if id not in used_ids_set:
                    used_ids_ordered.append(id)
                    used_ids_set.add(id)
        
        # Get citations for all used IDs
        citations = []
        for id in used_ids_ordered:
            cit = self.tracker.get_citation(id)
            if cit:
                citations.append(cit)
        
        # Order citations based on citation_order parameter
        if self.citation_order == "relevance":
            # Sort by relevance_score (descending), then by citation_id for stability
            citations.sort(key=lambda c: (-c.relevance_score, c.citation_id))
        elif self.citation_order == "appearance":
            # Keep appearance order (already in used_ids_ordered)
            pass  # citations is already in appearance order
        
        # Build bibliography section
        if self.language == "Vietnamese":
            bibliography = ["\n\n---\n## Tài liệu tham khảo\n"]
        else:
            bibliography = ["\n\n---\n## References\n"]
        
        # Check if we have multiple workspaces
        workspaces = set(cit.workspace or "Unknown" for cit in citations)
        
        if len(workspaces) > 1:
            # Multiple workspaces: add workspace headers but maintain global ordering
            # Iterate through ordered citations and add workspace header when it changes
            current_workspace = None
            for cit in citations:
                workspace = cit.workspace or "Unknown"
                
                # Add workspace header when workspace changes
                if workspace != current_workspace:
                    current_workspace = workspace
                    if self.language == "Vietnamese":
                        bibliography.append(f"\n### Workspace: {workspace}\n")
                    else:
                        bibliography.append(f"\n### Workspace: {workspace}\n")
                
                # Add citation entry
                bib_entry = self._format_bibliography_entry(cit)
                bibliography.append(bib_entry)
        else:
            # Single workspace - no grouping needed
            # Citations are already ordered according to citation_order
            for cit in citations:
                bib_entry = self._format_bibliography_entry(cit)
                bibliography.append(bib_entry)
        
        rendered_response = response + "\n".join(bibliography)
        citation_list = self._build_citation_list(ordered_citations=citations)
        
        return rendered_response, citation_list
    
    def _parse_citation_marker(self, marker: str) -> List[int]:
        """Parse citation marker to extract individual IDs.
        
        Handles various formats:
        - Single ID: "1" -> [1]
        - Range: "1-3" -> [1, 2, 3]
        - Comma-separated: "1,2,3" -> [1, 2, 3]
        - Mixed: "1-3,5,7-9" -> [1, 2, 3, 5, 7, 8, 9]
        
        Args:
            marker: Citation marker string (without brackets)
        
        Returns:
            List of individual citation IDs
        
        Example:
            >>> renderer = CitationRenderer(CitationTracker(), "inline")
            >>> renderer._parse_citation_marker("1-3")
            [1, 2, 3]
            >>> renderer._parse_citation_marker("1,3,5")
            [1, 3, 5]
            >>> renderer._parse_citation_marker("1-3,5")
            [1, 2, 3, 5]
        """
        ids = []
        
        # Split by comma
        parts = marker.split(',')
        
        for part in parts:
            part = part.strip()
            
            # Check if it's a range (e.g., "1-3")
            if '-' in part:
                range_parts = part.split('-')
                if len(range_parts) == 2:
                    try:
                        start = int(range_parts[0])
                        end = int(range_parts[1])
                        ids.extend(range(start, end + 1))
                    except ValueError:
                        pass
            else:
                # Single ID
                try:
                    ids.append(int(part))
                except ValueError:
                    pass
        
        return ids
    
    def _format_footnote(self, cit: CitationMetadata) -> str:
        """Format single footnote entry.
        
        Args:
            cit: CitationMetadata instance
        
        Returns:
            Formatted footnote string
        
        Example:
            >>> # Output: "[1] Luật Thuế GTGT, Điều 8, Workspace: legal-vn, Score: 0.95"
        """
        title = self._get_document_title(cit)
        chunk_ref = self._get_chunk_reference(cit)
        workspace = cit.workspace
        score = f"{cit.relevance_score:.2f}"
        
        if self.language == "Vietnamese":
            return f"[{cit.citation_id}] {title}, {chunk_ref}, Workspace: {workspace}, Độ liên quan: {score}\n"
        else:
            return f"[{cit.citation_id}] {title}, {chunk_ref}, Workspace: {workspace}, Score: {score}\n"
    
    def _format_bibliography_entry(self, cit: CitationMetadata) -> str:
        """Format single bibliography entry with full metadata based on doc_type.
        
        Args:
            cit: CitationMetadata instance
        
        Returns:
            Formatted bibliography entry string
        
        Example:
            >>> # Legal document output:
            >>> # "[1] Khoản 2 Điều 8 Luật Thuế GTGT số 13/2008/QH12 của Quốc hội (Còn hiệu lực)
            >>> #     - Loại: Luật
            >>> #     - Cơ quan ban hành: Quốc hội
            >>> #     - Trạng thái: Còn hiệu lực
            >>> #     ..."
        """
        # Use _format_legal_citation for legal documents to get full citation format
        if cit.doc_type == "LEGAL":
            citation_ref = self._format_legal_citation(cit)
        else:
            citation_ref = cit.formatted_ref
        
        lines = [f"[{cit.citation_id}] {citation_ref}"]
        
        # Format based on document type
        if cit.doc_type == "LEGAL":
            # Vietnamese legal document format with detailed metadata
            if cit.legal_info:
                if self.language == "Vietnamese":
                    if cit.legal_info.get("document_type"):
                        lines.append(f"    - Loại: {cit.legal_info['document_type']}")
                    if cit.legal_info.get("issuing_authority"):
                        lines.append(f"    - Cơ quan ban hành: {cit.legal_info['issuing_authority']}")
                    if cit.legal_info.get("legal_status"):
                        lines.append(f"    - Trạng thái: {cit.legal_info['legal_status']}")
                    if cit.legal_info.get("effective_date"):
                        lines.append(f"    - Ngày hiệu lực: {cit.legal_info['effective_date']}")
                    if cit.legal_info.get("document_number"):
                        lines.append(f"    - Số hiệu: {cit.legal_info['document_number']}")
                else:
                    if cit.legal_info.get("document_type"):
                        lines.append(f"    - Type: {cit.legal_info['document_type']}")
                    if cit.legal_info.get("issuing_authority"):
                        lines.append(f"    - Issuing Authority: {cit.legal_info['issuing_authority']}")
                    if cit.legal_info.get("legal_status"):
                        lines.append(f"    - Status: {cit.legal_info['legal_status']}")
                    if cit.legal_info.get("effective_date"):
                        lines.append(f"    - Effective Date: {cit.legal_info['effective_date']}")
                    if cit.legal_info.get("document_number"):
                        lines.append(f"    - Number: {cit.legal_info['document_number']}")
            
            # Add hierarchy path
            if cit.hierarchy_path:
                hierarchy_str = " > ".join(cit.hierarchy_path)
                if self.language == "Vietnamese":
                    lines.append(f"    - Vị trí: {hierarchy_str}")
                else:
                    lines.append(f"    - Location: {hierarchy_str}")
        
        else:  # doc_type == "GENERAL"
            # General document format with breadcrumb
            if self.language == "Vietnamese":
                lines.append("    - Loại: Tài liệu chung")
                lines.append(f"    - File: {cit.file_path}")
            else:
                lines.append("    - Type: General Document")
                lines.append(f"    - File: {cit.file_path}")
            
            # Add hierarchy path (breadcrumb)
            if cit.hierarchy_path:
                breadcrumb_str = " > ".join(cit.hierarchy_path)
                if self.language == "Vietnamese":
                    lines.append(f"    - Đường dẫn: {breadcrumb_str}")
                else:
                    lines.append(f"    - Path: {breadcrumb_str}")
        
        # Add workspace and relevance (common for both types)
        if self.language == "Vietnamese":
            lines.append(f"    - Workspace: {cit.workspace}")
            lines.append(f"    - Độ liên quan: {cit.relevance_score:.2f}")
        else:
            lines.append(f"    - Workspace: {cit.workspace}")
            lines.append(f"    - Relevance: {cit.relevance_score:.2f}")
        
        return "\n".join(lines) + "\n"
    
    def _get_document_title(self, cit: CitationMetadata) -> str:
        """Extract document title from metadata with fallback handling.
        
        Implements graceful degradation for missing metadata (Requirement 15.2):
        - If formatted_ref is empty, falls back to file_path
        - If file_path is also empty, returns "Unknown Document"
        - Logs warning when fallback is used
        
        Args:
            cit: CitationMetadata instance
        
        Returns:
            Document title string
        
        Example:
            >>> # Returns pre-rendered formatted_ref
            >>> # Or falls back to file_path if formatted_ref is missing
        """
        # Use pre-rendered formatted_ref if available
        if cit.formatted_ref:
            return cit.formatted_ref
        
        # Fallback to file_path if formatted_ref is missing (Requirement 15.2)
        if cit.file_path:
            from lightrag.utils import logger
            logger.warning(
                f"Citation {cit.citation_id}: formatted_ref is missing, using file_path as fallback: {cit.file_path}"
            )
            from pathlib import Path
            return Path(cit.file_path).name
        
        # Final fallback if both are missing
        from lightrag.utils import logger
        logger.warning(
            f"Citation {cit.citation_id}: Both formatted_ref and file_path are missing, using 'Unknown Document'"
        )
        return "Unknown Document"
    
    def _get_chunk_reference(self, cit: CitationMetadata) -> str:
        """Get chunk reference string with hierarchy context and fallback handling.
        
        Implements graceful degradation for missing metadata (Requirement 15.3):
        - Uses hierarchy_path if available
        - Falls back to chunk_id if hierarchy_path is missing
        - Returns "Unknown chunk" if chunk_id is also missing
        - Logs warning when fallback is used
        
        Args:
            cit: CitationMetadata instance
        
        Returns:
            Chunk reference string
        
        Example:
            >>> # For legal: "Chương II > Điều 8 > Khoản 2"
            >>> # For general: "Hạ tầng > Quy trình Backup > Bước 1"
            >>> # For entity: "Entity: Thuế GTGT"
            >>> # Fallback: "Unknown chunk"
        """
        if cit.source_type == "chunk":
            # Use hierarchy path if available
            if cit.hierarchy_path:
                return " > ".join(cit.hierarchy_path)
            
            # Fallback to chunk_id if hierarchy_path is missing
            if cit.chunk_id:
                if ":chunk-" in cit.chunk_id:
                    chunk_num = cit.chunk_id.split(":chunk-")[1]
                    return f"Chunk {chunk_num}"
                return cit.chunk_id
            
            # Final fallback if chunk_id is missing (Requirement 15.3)
            from lightrag.utils import logger
            logger.warning(
                f"Citation {cit.citation_id}: chunk_id is missing, using 'Unknown chunk' as reference"
            )
            return "Unknown chunk"
        
        elif cit.source_type == "entity":
            if cit.entity_name:
                return f"Entity: {cit.entity_name}"
            else:
                from lightrag.utils import logger
                logger.warning(
                    f"Citation {cit.citation_id}: entity_name is missing for entity source"
                )
                return "Entity: Unknown"
        
        elif cit.source_type == "relation":
            if cit.relation_key:
                return f"Relation: {cit.relation_key}"
            else:
                from lightrag.utils import logger
                logger.warning(
                    f"Citation {cit.citation_id}: relation_key is missing for relation source"
                )
                return "Relation: Unknown"
        
        return "Unknown"
    
    def _build_citation_list(self, ordered_citations: Optional[List[CitationMetadata]] = None) -> CitationList:
        """Build complete citation list with metadata.
        
        Args:
            ordered_citations: Optional list of citations in desired order.
                If provided, uses this order. Otherwise, gets all citations from tracker.
        
        Returns:
            CitationList instance with all citations and statistics
        
        Example:
            >>> # Returns CitationList with citations, format, counts, etc.
        """
        if ordered_citations is not None:
            citations = ordered_citations
        else:
            citations = self.tracker.get_all_citations()
            
            # Order citations based on citation_order parameter
            if self.citation_order == "relevance":
                # Sort by relevance_score (descending), then by citation_id for stability
                citations = sorted(citations, key=lambda c: (-c.relevance_score, c.citation_id))
            # For "appearance", keep the order from tracker (which is by citation_id)
        
        workspaces = list(set(c.workspace for c in citations))
        
        # Count by document type
        legal_count = sum(1 for c in citations if c.doc_type == "LEGAL")
        general_count = sum(1 for c in citations if c.doc_type == "GENERAL")
        
        return CitationList(
            citations=citations,
            format=self.format,
            total_sources=len(citations),
            workspaces_used=workspaces,
            legal_count=legal_count,
            general_count=general_count,
        )
    
    def _format_legal_citation(self, cit: CitationMetadata) -> str:
        """Build citation string from legal_info metadata for Vietnamese legal documents.
        
        This helper formats legal citations according to Vietnamese legal citation standards,
        including chapter, article, clause, document type, number, issuing authority, and status.
        
        Implements graceful degradation for missing legal metadata (Requirement 15.4):
        - Falls back to formatted_ref if legal_info is missing
        - Falls back to file_path if formatted_ref is also missing
        - Logs warning when degradation occurs
        
        Format: "[Khoản X] [Điều Y] [Loại] số [Số hiệu] của [Cơ quan] ([Trạng thái])"
        
        Args:
            cit: CitationMetadata instance with legal_info populated
        
        Returns:
            Formatted legal citation string
        
        Examples:
            >>> # Full citation with all fields:
            >>> # "Khoản 2 Điều 8 Luật số 13/2008/QH12 của Quốc hội (Còn hiệu lực)"
            >>> 
            >>> # Citation without clause:
            >>> # "Điều 8 Luật số 13/2008/QH12 của Quốc hội (Còn hiệu lực)"
            >>> 
            >>> # Citation with chapter from hierarchy_path:
            >>> # "Chương II Điều 8 Luật số 13/2008/QH12 của Quốc hội (Còn hiệu lực)"
            >>> 
            >>> # Degraded citation (missing legal_info):
            >>> # Falls back to formatted_ref or file_path
        """
        # Graceful degradation if legal_info is missing (Requirement 15.4)
        if not cit.legal_info:
            from lightrag.utils import logger
            logger.warning(
                f"Citation {cit.citation_id}: legal_info is missing for legal document, "
                f"degrading to basic format"
            )
            # Fallback to formatted_ref if available
            if cit.formatted_ref:
                return cit.formatted_ref
            # Further fallback to file_path
            if cit.file_path:
                from pathlib import Path
                return Path(cit.file_path).name
            # Final fallback
            return "Unknown Legal Document"
        
        parts = []
        
        # Extract chapter from hierarchy_path if available
        chapter = None
        if cit.hierarchy_path:
            for item in cit.hierarchy_path:
                if item.startswith("Chương"):
                    chapter = item.split(":")[0].strip()  # Extract "Chương II" from "Chương II: Thuế GTGT"
                    break
        
        # Add chapter if found
        if chapter:
            parts.append(chapter)
        
        # Add clause (Khoản) if available
        clause = cit.legal_info.get("clause", "")
        if clause:
            parts.append(clause)
        
        # Add article (Điều) if available
        article = cit.legal_info.get("article", "")
        if article:
            parts.append(article)
        
        # Build document reference: "[Loại] số [Số hiệu]"
        doc_type = cit.legal_info.get("document_type", "")
        doc_number = cit.legal_info.get("document_number", "")
        
        if doc_type and doc_number:
            parts.append(f"{doc_type} số {doc_number}")
        elif doc_type:
            parts.append(doc_type)
        elif doc_number:
            parts.append(f"số {doc_number}")
        
        # Add issuing authority: "của [Cơ quan]"
        issuing_authority = cit.legal_info.get("issuing_authority", "")
        if issuing_authority:
            parts.append(f"của {issuing_authority}")
        
        # Add legal status in parentheses: "([Trạng thái])"
        legal_status = cit.legal_info.get("legal_status", "")
        if legal_status:
            # Join all parts first, then append status in parentheses
            citation_base = " ".join(parts)
            return f"{citation_base} ({legal_status})"
        
        # Return citation without status if not available
        # If no parts were built, fall back to formatted_ref or file_path
        if not parts:
            from lightrag.utils import logger
            logger.warning(
                f"Citation {cit.citation_id}: legal_info exists but all fields are empty, "
                f"degrading to basic format"
            )
            if cit.formatted_ref:
                return cit.formatted_ref
            if cit.file_path:
                from pathlib import Path
                return Path(cit.file_path).name
            return "Unknown Legal Document"
        
        return " ".join(parts)


# ============================================================================
# Legal Metadata Update and Propagation
# ============================================================================

async def propagate_legal_metadata_to_chunks(
    doc_id: str,
    legal_metadata_update: Dict[str, str],
    doc_status_storage: Any,
    text_chunks_storage: Any,
) -> int:
    """Propagate legal metadata updates to all chunks of a document.
    
    This function updates legal metadata fields in all chunks associated with
    a document without triggering re-chunking or re-extraction. It's used when
    legal metadata (like legal_status) changes but the document content remains
    the same.
    
    The function:
    1. Retrieves the document status to get the list of chunk IDs
    2. Fetches all chunks for the document
    3. Updates legal metadata fields in each chunk's metadata
    4. Upserts updated chunks back to storage
    
    Implements Requirement 20.5, 20.6:
    - Query all chunks for the document by doc_id
    - Update legal metadata fields in each chunk's metadata
    - Upsert updated chunks back to storage
    
    Args:
        doc_id: Document identifier (e.g., "doc-550e8400-e29b-41d4-a716-446655440000")
        legal_metadata_update: Dictionary with legal metadata fields to update:
            - legal_status: Legal status (e.g., "Còn hiệu lực", "Hết hiệu lực")
            - effective_date: Effective date (e.g., "2008-06-03")
            - issuing_authority: Issuing authority (e.g., "Quốc hội")
            - document_type: Document type (e.g., "Luật", "Nghị định")
            - document_number: Document number (e.g., "13/2008/QH12")
        doc_status_storage: Document status storage instance
        text_chunks_storage: Text chunks storage instance
    
    Returns:
        Number of chunks updated
    
    Raises:
        ValueError: If document not found or has no chunks
        Exception: If chunk update fails
    
    Example:
        >>> # Update legal status for a document
        >>> updated_count = await propagate_legal_metadata_to_chunks(
        ...     doc_id="doc-550e8400-e29b-41d4-a716-446655440000",
        ...     legal_metadata_update={
        ...         "legal_status": "Hết hiệu lực",
        ...         "effective_date": "2024-01-01"
        ...     },
        ...     doc_status_storage=rag.doc_status,
        ...     text_chunks_storage=rag.text_chunks
        ... )
        >>> print(f"Updated {updated_count} chunks")
        Updated 15 chunks
    """
    from lightrag.utils import logger
    
    # Step 1: Get document status to retrieve chunk IDs
    doc_data = await doc_status_storage.get_by_id(doc_id)
    
    if not doc_data:
        raise ValueError(f"Document {doc_id} not found in storage")
    
    # Extract chunks_list from document data
    chunks_list = doc_data.get("chunks_list", [])
    
    if not chunks_list:
        logger.warning(f"Document {doc_id} has no chunks to update")
        return 0
    
    logger.info(
        f"Propagating legal metadata to {len(chunks_list)} chunks for document {doc_id}"
    )
    
    # Step 2: Fetch all chunks for the document
    chunk_data_list = await text_chunks_storage.get_by_ids(chunks_list)
    
    if not chunk_data_list:
        raise ValueError(f"No chunks found for document {doc_id}")
    
    # Step 3: Update legal metadata in each chunk
    updated_chunks = {}
    updated_count = 0
    
    for chunk_id, chunk_data in zip(chunks_list, chunk_data_list):
        if not chunk_data:
            logger.warning(f"Chunk {chunk_id} not found, skipping")
            continue
        
        # Get or create metadata dict
        metadata = chunk_data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Get or create legal_info dict within metadata
        legal_info = metadata.get("legal_info", {})
        if not isinstance(legal_info, dict):
            legal_info = {}
        
        # Update legal metadata fields (only update provided fields)
        for field_name, field_value in legal_metadata_update.items():
            if field_value is not None:  # Only update non-None values
                legal_info[field_name] = field_value
        
        # Update metadata with new legal_info
        metadata["legal_info"] = legal_info
        chunk_data["metadata"] = metadata
        
        # Add to update batch
        updated_chunks[chunk_id] = chunk_data
        updated_count += 1
    
    # Step 4: Upsert updated chunks back to storage
    if updated_chunks:
        await text_chunks_storage.upsert(updated_chunks)
        logger.info(
            f"Successfully updated legal metadata in {updated_count} chunks for document {doc_id}"
        )
    
    return updated_count
