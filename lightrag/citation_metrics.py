"""Citation metrics and observability for LightRAG.

This module provides metrics tracking and observability for the citation system,
including usage statistics, quality metrics, and validation tracking.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Literal
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CitationUsageMetrics:
    """Metrics for citation usage tracking.
    
    Tracks citation usage patterns across queries to understand how the
    citation system is being used and identify optimization opportunities.
    
    Attributes:
        total_queries: Total number of queries processed
        queries_with_citations: Number of queries that used citations
        queries_without_citations: Number of queries without citations
        total_citations: Total number of citations across all queries
        citations_by_format: Distribution of citation formats used
        citations_by_workspace: Distribution of citations per workspace
        avg_citations_per_query: Average number of citations per query
    """
    
    total_queries: int = 0
    queries_with_citations: int = 0
    queries_without_citations: int = 0
    total_citations: int = 0
    citations_by_format: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    citations_by_workspace: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def citation_usage_rate(self) -> float:
        """Calculate percentage of queries using citations.
        
        Returns:
            Citation usage rate as percentage (0.0-100.0)
        
        Example:
            >>> metrics = CitationUsageMetrics(total_queries=100, queries_with_citations=75)
            >>> metrics.citation_usage_rate
            75.0
        """
        if self.total_queries == 0:
            return 0.0
        return (self.queries_with_citations / self.total_queries) * 100.0
    
    @property
    def avg_citations_per_query(self) -> float:
        """Calculate average number of citations per query.
        
        Returns:
            Average citations per query (0.0+)
        
        Example:
            >>> metrics = CitationUsageMetrics(total_queries=10, total_citations=50)
            >>> metrics.avg_citations_per_query
            5.0
        """
        if self.total_queries == 0:
            return 0.0
        return self.total_citations / self.total_queries


@dataclass
class CitationQualityMetrics:
    """Metrics for citation quality tracking.
    
    Tracks citation quality and validation to identify issues like hallucinated
    citations, low coverage, or validation failures.
    
    Attributes:
        total_citations_validated: Total number of citations validated
        valid_citations: Number of valid citation references
        invalid_citations: Number of invalid citation references
        total_context_items: Total number of context items provided to LLM
        cited_context_items: Number of context items actually cited
        validation_failures: Number of validation failures by type
    """
    
    total_citations_validated: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    total_context_items: int = 0
    cited_context_items: int = 0
    validation_failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def citation_accuracy(self) -> float:
        """Calculate percentage of valid citation references.
        
        Returns:
            Citation accuracy as percentage (0.0-100.0)
        
        Example:
            >>> metrics = CitationQualityMetrics(total_citations_validated=100, valid_citations=95)
            >>> metrics.citation_accuracy
            95.0
        """
        if self.total_citations_validated == 0:
            return 0.0
        return (self.valid_citations / self.total_citations_validated) * 100.0
    
    @property
    def citation_coverage(self) -> float:
        """Calculate percentage of context items cited in response.
        
        Returns:
            Citation coverage as percentage (0.0-100.0)
        
        Example:
            >>> metrics = CitationQualityMetrics(total_context_items=20, cited_context_items=15)
            >>> metrics.citation_coverage
            75.0
        """
        if self.total_context_items == 0:
            return 0.0
        return (self.cited_context_items / self.total_context_items) * 100.0


class CitationMetricsTracker:
    """Tracks citation metrics across queries for observability.
    
    This class maintains running statistics about citation usage and quality,
    providing insights for monitoring and optimization. Metrics are logged
    periodically and can be exported for external monitoring systems.
    
    The tracker is designed to be lightweight and non-blocking, with minimal
    performance overhead. All metrics are stored in memory and can be reset
    or exported as needed.
    
    Attributes:
        usage_metrics: Citation usage statistics
        quality_metrics: Citation quality statistics
        _query_history: Recent query metrics for detailed analysis
    
    Example:
        >>> tracker = CitationMetricsTracker()
        >>> 
        >>> # Track a query with citations
        >>> tracker.track_query(
        ...     enable_citations=True,
        ...     citation_count=5,
        ...     citation_format="footnote",
        ...     workspaces=["legal-vn", "user-docs"]
        ... )
        >>> 
        >>> # Track validation results
        >>> tracker.track_validation(
        ...     total_citations=5,
        ...     valid_citations=5,
        ...     invalid_citations=0,
        ...     context_items=10,
        ...     cited_items=8
        ... )
        >>> 
        >>> # Get current metrics
        >>> metrics = tracker.get_usage_metrics()
        >>> metrics.citation_usage_rate
        100.0
    """
    
    def __init__(self, log_interval: int = 100):
        """Initialize citation metrics tracker.
        
        Args:
            log_interval: Number of queries between metric logging (default: 100)
        """
        self.usage_metrics = CitationUsageMetrics()
        self.quality_metrics = CitationQualityMetrics()
        self._log_interval = log_interval
        self._query_count = 0
        
        # Store recent query metrics for detailed analysis (last 1000 queries)
        self._query_history: List[Dict] = []
        self._max_history_size = 1000
    
    def track_query(
        self,
        enable_citations: bool,
        citation_count: int = 0,
        citation_format: Optional[Literal["inline", "footnote", "bibliography"]] = None,
        workspaces: Optional[List[str]] = None,
    ) -> None:
        """Track citation usage for a query.
        
        This method records citation usage statistics for a single query,
        including whether citations were enabled, how many were used, the
        format, and which workspaces were involved.
        
        Implements Requirements 16.1, 16.2, 16.3, 16.4:
        - Tracks citation usage rate (percentage of queries using citations)
        - Tracks average number of citations per query
        - Tracks distribution of citation formats
        - Tracks distribution of citations per workspace
        
        Args:
            enable_citations: Whether citations were enabled for this query
            citation_count: Number of citations used (default: 0)
            citation_format: Citation format used (optional)
            workspaces: List of workspaces queried (optional)
        
        Example:
            >>> tracker = CitationMetricsTracker()
            >>> tracker.track_query(
            ...     enable_citations=True,
            ...     citation_count=5,
            ...     citation_format="footnote",
            ...     workspaces=["legal-vn", "user-docs"]
            ... )
            >>> tracker.usage_metrics.total_queries
            1
            >>> tracker.usage_metrics.queries_with_citations
            1
        """
        # Update query counts
        self.usage_metrics.total_queries += 1
        self._query_count += 1
        
        if enable_citations and citation_count > 0:
            # Query used citations
            self.usage_metrics.queries_with_citations += 1
            self.usage_metrics.total_citations += citation_count
            
            # Track citation format distribution
            if citation_format:
                self.usage_metrics.citations_by_format[citation_format] += citation_count
            
            # Track workspace distribution
            if workspaces:
                for workspace in workspaces:
                    self.usage_metrics.citations_by_workspace[workspace] += citation_count
        else:
            # Query did not use citations
            self.usage_metrics.queries_without_citations += 1
        
        # Store in query history for detailed analysis
        query_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "enable_citations": enable_citations,
            "citation_count": citation_count,
            "citation_format": citation_format,
            "workspaces": workspaces or [],
        }
        self._query_history.append(query_record)
        
        # Trim history if it exceeds max size
        if len(self._query_history) > self._max_history_size:
            self._query_history = self._query_history[-self._max_history_size:]
        
        # Log metrics periodically
        if self._query_count % self._log_interval == 0:
            self._log_usage_metrics()
    
    def track_validation(
        self,
        total_citations: int,
        valid_citations: int,
        invalid_citations: int,
        context_items: int,
        cited_items: int,
        failure_type: Optional[str] = None,
    ) -> None:
        """Track citation validation results.
        
        This method records citation quality metrics, including validation
        results, coverage, and failure types. This helps identify issues
        like hallucinated citations or low coverage.
        
        Implements Requirements 16.5, 16.6, 11.6:
        - Tracks citation accuracy (percentage of valid references)
        - Tracks citation coverage (percentage of context items cited)
        - Tracks citation validation failures (invalid IDs, hallucinations)
        
        Args:
            total_citations: Total number of citations validated
            valid_citations: Number of valid citation references
            invalid_citations: Number of invalid citation references
            context_items: Total number of context items provided to LLM
            cited_items: Number of context items actually cited in response
            failure_type: Type of validation failure (optional)
        
        Example:
            >>> tracker = CitationMetricsTracker()
            >>> tracker.track_validation(
            ...     total_citations=10,
            ...     valid_citations=9,
            ...     invalid_citations=1,
            ...     context_items=15,
            ...     cited_items=12,
            ...     failure_type="hallucinated_id"
            ... )
            >>> tracker.quality_metrics.citation_accuracy
            90.0
        """
        # Update validation counts
        self.quality_metrics.total_citations_validated += total_citations
        self.quality_metrics.valid_citations += valid_citations
        self.quality_metrics.invalid_citations += invalid_citations
        
        # Update coverage counts
        self.quality_metrics.total_context_items += context_items
        self.quality_metrics.cited_context_items += cited_items
        
        # Track failure type if provided
        if failure_type and invalid_citations > 0:
            self.quality_metrics.validation_failures[failure_type] += invalid_citations
        
        # Log warning if accuracy is low
        if total_citations > 0:
            accuracy = (valid_citations / total_citations) * 100.0
            if accuracy < 90.0:
                logger.warning(
                    f"Low citation accuracy detected: {accuracy:.1f}% "
                    f"({valid_citations}/{total_citations} valid). "
                    f"Invalid citations: {invalid_citations}"
                )
        
        # Log warning if coverage is low
        if context_items > 0:
            coverage = (cited_items / context_items) * 100.0
            if coverage < 50.0:
                logger.warning(
                    f"Low citation coverage detected: {coverage:.1f}% "
                    f"({cited_items}/{context_items} items cited). "
                    f"Many context items were not referenced in the response."
                )
    
    def get_usage_metrics(self) -> CitationUsageMetrics:
        """Get current citation usage metrics.
        
        Returns:
            CitationUsageMetrics instance with current statistics
        
        Example:
            >>> tracker = CitationMetricsTracker()
            >>> tracker.track_query(enable_citations=True, citation_count=5)
            >>> metrics = tracker.get_usage_metrics()
            >>> metrics.citation_usage_rate
            100.0
        """
        return self.usage_metrics
    
    def get_quality_metrics(self) -> CitationQualityMetrics:
        """Get current citation quality metrics.
        
        Returns:
            CitationQualityMetrics instance with current statistics
        
        Example:
            >>> tracker = CitationMetricsTracker()
            >>> tracker.track_validation(total_citations=10, valid_citations=9, invalid_citations=1, context_items=15, cited_items=12)
            >>> metrics = tracker.get_quality_metrics()
            >>> metrics.citation_accuracy
            90.0
        """
        return self.quality_metrics
    
    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary.
        
        Returns:
            Dictionary with all metrics and statistics
        
        Example:
            >>> tracker = CitationMetricsTracker()
            >>> summary = tracker.get_summary()
            >>> "usage" in summary
            True
            >>> "quality" in summary
            True
        """
        return {
            "usage": {
                "total_queries": self.usage_metrics.total_queries,
                "queries_with_citations": self.usage_metrics.queries_with_citations,
                "queries_without_citations": self.usage_metrics.queries_without_citations,
                "citation_usage_rate": self.usage_metrics.citation_usage_rate,
                "total_citations": self.usage_metrics.total_citations,
                "avg_citations_per_query": self.usage_metrics.avg_citations_per_query,
                "citations_by_format": dict(self.usage_metrics.citations_by_format),
                "citations_by_workspace": dict(self.usage_metrics.citations_by_workspace),
            },
            "quality": {
                "total_citations_validated": self.quality_metrics.total_citations_validated,
                "valid_citations": self.quality_metrics.valid_citations,
                "invalid_citations": self.quality_metrics.invalid_citations,
                "citation_accuracy": self.quality_metrics.citation_accuracy,
                "total_context_items": self.quality_metrics.total_context_items,
                "cited_context_items": self.quality_metrics.cited_context_items,
                "citation_coverage": self.quality_metrics.citation_coverage,
                "validation_failures": dict(self.quality_metrics.validation_failures),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics to zero.
        
        This is useful for testing or when starting a new monitoring period.
        
        Example:
            >>> tracker = CitationMetricsTracker()
            >>> tracker.track_query(enable_citations=True, citation_count=5)
            >>> tracker.reset_metrics()
            >>> tracker.usage_metrics.total_queries
            0
        """
        self.usage_metrics = CitationUsageMetrics()
        self.quality_metrics = CitationQualityMetrics()
        self._query_count = 0
        self._query_history = []
        logger.info("Citation metrics reset")
    
    def _log_usage_metrics(self) -> None:
        """Log current usage metrics for monitoring.
        
        This method is called periodically (every log_interval queries) to
        provide visibility into citation usage patterns.
        """
        logger.info(
            f"Citation Usage Metrics (last {self._log_interval} queries): "
            f"Usage Rate: {self.usage_metrics.citation_usage_rate:.1f}%, "
            f"Avg Citations/Query: {self.usage_metrics.avg_citations_per_query:.1f}, "
            f"Total Queries: {self.usage_metrics.total_queries}, "
            f"Total Citations: {self.usage_metrics.total_citations}"
        )
        
        # Log format distribution if available
        if self.usage_metrics.citations_by_format:
            format_dist = ", ".join(
                f"{fmt}: {count}" 
                for fmt, count in sorted(self.usage_metrics.citations_by_format.items())
            )
            logger.info(f"Citation Format Distribution: {format_dist}")
        
        # Log workspace distribution if available
        if self.usage_metrics.citations_by_workspace:
            # Show top 5 workspaces
            top_workspaces = sorted(
                self.usage_metrics.citations_by_workspace.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            workspace_dist = ", ".join(
                f"{ws}: {count}" 
                for ws, count in top_workspaces
            )
            logger.info(f"Top Workspaces: {workspace_dist}")
        
        # Log quality metrics if available
        if self.quality_metrics.total_citations_validated > 0:
            logger.info(
                f"Citation Quality Metrics: "
                f"Accuracy: {self.quality_metrics.citation_accuracy:.1f}%, "
                f"Coverage: {self.quality_metrics.citation_coverage:.1f}%, "
                f"Invalid Citations: {self.quality_metrics.invalid_citations}"
            )


# Global metrics tracker instance
_global_metrics_tracker: Optional[CitationMetricsTracker] = None


def get_global_metrics_tracker() -> CitationMetricsTracker:
    """Get or create the global citation metrics tracker.
    
    This function provides access to a singleton metrics tracker that can be
    used across the application for consistent metrics collection.
    
    Returns:
        Global CitationMetricsTracker instance
    
    Example:
        >>> tracker = get_global_metrics_tracker()
        >>> tracker.track_query(enable_citations=True, citation_count=5)
    """
    global _global_metrics_tracker
    
    if _global_metrics_tracker is None:
        _global_metrics_tracker = CitationMetricsTracker()
        logger.info("Initialized global citation metrics tracker")
    
    return _global_metrics_tracker


def reset_global_metrics_tracker() -> None:
    """Reset the global metrics tracker.
    
    This is useful for testing or when starting a new monitoring period.
    
    Example:
        >>> reset_global_metrics_tracker()
        >>> tracker = get_global_metrics_tracker()
        >>> tracker.usage_metrics.total_queries
        0
    """
    global _global_metrics_tracker
    
    if _global_metrics_tracker is not None:
        _global_metrics_tracker.reset_metrics()
    else:
        _global_metrics_tracker = CitationMetricsTracker()
        logger.info("Initialized new global citation metrics tracker")
