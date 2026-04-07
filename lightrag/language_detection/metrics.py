"""
Metrics and monitoring capabilities for FastText language detection.

This module provides comprehensive monitoring, metrics collection, and health check
functionality for the FastText language detection system. It supports integration
with monitoring systems like Prometheus and Grafana.

Key Features:
- Performance metrics (accuracy, latency, throughput)
- Health check endpoints for FastText model availability
- Detection method usage tracking
- Prometheus-compatible metrics export
- Comprehensive logging and observability
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from threading import Lock

from lightrag.language_detector import SupportedLanguage
from .logger import get_fasttext_logger

# Use dedicated FastText logger
logger = get_fasttext_logger("metrics")


@dataclass
class DetectionMetric:
    """Single detection event metric."""
    
    timestamp: float
    language: SupportedLanguage
    confidence: float
    method_used: str  # "fasttext", "unicode", "hybrid"
    latency_ms: float
    text_length: int
    cached: bool
    success: bool = True
    error: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Throughput metrics
    total_detections: int = 0
    detections_per_second: float = 0.0
    
    # Accuracy metrics (if ground truth available)
    accuracy_percent: Optional[float] = None
    vietnamese_accuracy_percent: Optional[float] = None
    english_accuracy_percent: Optional[float] = None
    
    # Cache metrics
    cache_hit_rate_percent: float = 0.0
    cache_miss_rate_percent: float = 0.0
    
    # Method usage distribution
    fasttext_usage_percent: float = 0.0
    unicode_usage_percent: float = 0.0
    hybrid_usage_percent: float = 0.0
    
    # Error metrics
    error_rate_percent: float = 0.0
    total_errors: int = 0


@dataclass
class HealthStatus:
    """Health check status for monitoring systems."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: float
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Optional[PerformanceMetrics] = None
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "status": self.status,
            "timestamp": self.timestamp,
            "components": self.components,
            "message": self.message
        }
        if self.metrics:
            result["metrics"] = asdict(self.metrics)
        return result


class MetricsCollector:
    """
    Collects and aggregates metrics for language detection operations.
    
    This class provides thread-safe metrics collection with support for:
    - Real-time performance tracking
    - Historical metrics with configurable retention
    - Percentile calculations for latency
    - Method usage distribution
    - Error tracking and reporting
    """
    
    def __init__(self, retention_window: int = 3600, max_samples: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            retention_window: Time window in seconds to retain metrics (default: 1 hour)
            max_samples: Maximum number of samples to retain for percentile calculations
        """
        self.retention_window = retention_window
        self.max_samples = max_samples
        
        # Thread-safe metrics storage
        self._lock = Lock()
        self._metrics: deque[DetectionMetric] = deque(maxlen=max_samples)
        
        # Aggregated counters
        self._total_detections = 0
        self._total_errors = 0
        self._method_counts = defaultdict(int)
        self._language_counts = defaultdict(int)
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance tracking
        self._start_time = time.time()
        self._last_reset_time = time.time()
        
        # Accuracy tracking (optional, requires ground truth)
        self._accuracy_samples: List[Tuple[SupportedLanguage, SupportedLanguage]] = []
        
        logger.info(f"MetricsCollector initialized with retention_window={retention_window}s, "
                   f"max_samples={max_samples}")
    
    def record_detection(self, 
                        language: SupportedLanguage,
                        confidence: float,
                        method_used: str,
                        latency_ms: float,
                        text_length: int,
                        cached: bool,
                        success: bool = True,
                        error: Optional[str] = None) -> None:
        """
        Record a detection event.
        
        Args:
            language: Detected language
            confidence: Detection confidence score
            method_used: Detection method ("fasttext", "unicode", "hybrid")
            latency_ms: Detection latency in milliseconds
            text_length: Length of input text
            cached: Whether result was served from cache
            success: Whether detection succeeded
            error: Error message if detection failed
        """
        with self._lock:
            # Create metric record
            metric = DetectionMetric(
                timestamp=time.time(),
                language=language,
                confidence=confidence,
                method_used=method_used,
                latency_ms=latency_ms,
                text_length=text_length,
                cached=cached,
                success=success,
                error=error
            )
            
            # Store metric
            self._metrics.append(metric)
            
            # Update counters
            self._total_detections += 1
            self._method_counts[method_used] += 1
            self._language_counts[language.value] += 1
            
            if cached:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            
            if not success:
                self._total_errors += 1
            
            # Clean up old metrics
            self._cleanup_old_metrics()
    
    def record_accuracy(self, predicted: SupportedLanguage, actual: SupportedLanguage) -> None:
        """
        Record accuracy sample for evaluation.
        
        Args:
            predicted: Predicted language
            actual: Actual/ground truth language
        """
        with self._lock:
            self._accuracy_samples.append((predicted, actual))
            
            # Limit accuracy samples to max_samples
            if len(self._accuracy_samples) > self.max_samples:
                self._accuracy_samples = self._accuracy_samples[-self.max_samples:]
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention window."""
        if not self._metrics:
            return
        
        cutoff_time = time.time() - self.retention_window
        
        # Remove old metrics from the front of the deque
        while self._metrics and self._metrics[0].timestamp < cutoff_time:
            self._metrics.popleft()
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Calculate and return aggregated performance metrics.
        
        Returns:
            PerformanceMetrics with current statistics
        """
        with self._lock:
            self._cleanup_old_metrics()
            
            # Calculate accuracy if samples available (independent of detection metrics)
            accuracy = None
            vietnamese_accuracy = None
            english_accuracy = None
            
            if self._accuracy_samples:
                correct = sum(1 for pred, actual in self._accuracy_samples if pred == actual)
                accuracy = (correct / len(self._accuracy_samples)) * 100
                
                # Calculate per-language accuracy
                vietnamese_samples = [(pred, actual) for pred, actual in self._accuracy_samples 
                                     if actual == SupportedLanguage.VIETNAMESE]
                if vietnamese_samples:
                    vietnamese_correct = sum(1 for pred, actual in vietnamese_samples if pred == actual)
                    vietnamese_accuracy = (vietnamese_correct / len(vietnamese_samples)) * 100
                
                english_samples = [(pred, actual) for pred, actual in self._accuracy_samples 
                                  if actual == SupportedLanguage.ENGLISH]
                if english_samples:
                    english_correct = sum(1 for pred, actual in english_samples if pred == actual)
                    english_accuracy = (english_correct / len(english_samples)) * 100
            
            # If no detection metrics, return basic metrics with accuracy
            if not self._metrics:
                return PerformanceMetrics(
                    accuracy_percent=round(accuracy, 2) if accuracy is not None else None,
                    vietnamese_accuracy_percent=round(vietnamese_accuracy, 2) if vietnamese_accuracy is not None else None,
                    english_accuracy_percent=round(english_accuracy, 2) if english_accuracy is not None else None
                )
            
            # Calculate latency statistics
            latencies = [m.latency_ms for m in self._metrics if m.success]
            
            if latencies:
                latencies_sorted = sorted(latencies)
                n = len(latencies_sorted)
                
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                p50_latency = latencies_sorted[int(n * 0.50)]
                p95_latency = latencies_sorted[int(n * 0.95)]
                p99_latency = latencies_sorted[int(n * 0.99)]
            else:
                avg_latency = min_latency = max_latency = 0.0
                p50_latency = p95_latency = p99_latency = 0.0
            
            # Calculate throughput
            time_window = time.time() - self._last_reset_time
            detections_per_second = self._total_detections / time_window if time_window > 0 else 0.0
            
            # Calculate cache metrics
            total_cache_requests = self._cache_hits + self._cache_misses
            cache_hit_rate = (self._cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0.0
            cache_miss_rate = 100.0 - cache_hit_rate
            
            # Calculate method usage distribution
            total_method_counts = sum(self._method_counts.values())
            fasttext_usage = (self._method_counts.get("fasttext", 0) / total_method_counts * 100) if total_method_counts > 0 else 0.0
            unicode_usage = (self._method_counts.get("unicode", 0) / total_method_counts * 100) if total_method_counts > 0 else 0.0
            hybrid_usage = (self._method_counts.get("hybrid", 0) / total_method_counts * 100) if total_method_counts > 0 else 0.0
            
            # Calculate error rate
            error_rate = (self._total_errors / self._total_detections * 100) if self._total_detections > 0 else 0.0
            
            return PerformanceMetrics(
                avg_latency_ms=round(avg_latency, 2),
                min_latency_ms=round(min_latency, 2),
                max_latency_ms=round(max_latency, 2),
                p50_latency_ms=round(p50_latency, 2),
                p95_latency_ms=round(p95_latency, 2),
                p99_latency_ms=round(p99_latency, 2),
                total_detections=self._total_detections,
                detections_per_second=round(detections_per_second, 2),
                accuracy_percent=round(accuracy, 2) if accuracy is not None else None,
                vietnamese_accuracy_percent=round(vietnamese_accuracy, 2) if vietnamese_accuracy is not None else None,
                english_accuracy_percent=round(english_accuracy, 2) if english_accuracy is not None else None,
                cache_hit_rate_percent=round(cache_hit_rate, 2),
                cache_miss_rate_percent=round(cache_miss_rate, 2),
                fasttext_usage_percent=round(fasttext_usage, 2),
                unicode_usage_percent=round(unicode_usage, 2),
                hybrid_usage_percent=round(hybrid_usage, 2),
                error_rate_percent=round(error_rate, 2),
                total_errors=self._total_errors
            )
    
    def get_method_distribution(self) -> Dict[str, int]:
        """
        Get detection method usage distribution.
        
        Returns:
            Dictionary mapping method names to usage counts
        """
        with self._lock:
            return dict(self._method_counts)
    
    def get_language_distribution(self) -> Dict[str, int]:
        """
        Get detected language distribution.
        
        Returns:
            Dictionary mapping language names to detection counts
        """
        with self._lock:
            return dict(self._language_counts)
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent detection errors.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of error records with timestamps and details
        """
        with self._lock:
            errors = [
                {
                    "timestamp": m.timestamp,
                    "method_used": m.method_used,
                    "text_length": m.text_length,
                    "error": m.error
                }
                for m in self._metrics
                if not m.success and m.error
            ]
            return errors[-limit:]
    
    def reset_metrics(self) -> None:
        """Reset all metrics and counters."""
        with self._lock:
            self._metrics.clear()
            self._total_detections = 0
            self._total_errors = 0
            self._method_counts.clear()
            self._language_counts.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._accuracy_samples.clear()
            self._last_reset_time = time.time()
            
            logger.info("Metrics reset successfully")
    
    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus text format.
        
        Returns:
            String containing Prometheus-formatted metrics
        """
        metrics = self.get_performance_metrics()
        method_dist = self.get_method_distribution()
        language_dist = self.get_language_distribution()
        
        lines = [
            "# HELP lightrag_language_detection_total Total number of language detections",
            "# TYPE lightrag_language_detection_total counter",
            f"lightrag_language_detection_total {metrics.total_detections}",
            "",
            "# HELP lightrag_language_detection_errors_total Total number of detection errors",
            "# TYPE lightrag_language_detection_errors_total counter",
            f"lightrag_language_detection_errors_total {metrics.total_errors}",
            "",
            "# HELP lightrag_language_detection_latency_seconds Detection latency in seconds",
            "# TYPE lightrag_language_detection_latency_seconds summary",
            f"lightrag_language_detection_latency_seconds{{quantile=\"0.5\"}} {metrics.p50_latency_ms / 1000}",
            f"lightrag_language_detection_latency_seconds{{quantile=\"0.95\"}} {metrics.p95_latency_ms / 1000}",
            f"lightrag_language_detection_latency_seconds{{quantile=\"0.99\"}} {metrics.p99_latency_ms / 1000}",
            f"lightrag_language_detection_latency_seconds_sum {metrics.avg_latency_ms * metrics.total_detections / 1000}",
            f"lightrag_language_detection_latency_seconds_count {metrics.total_detections}",
            "",
            "# HELP lightrag_language_detection_throughput_per_second Detection throughput per second",
            "# TYPE lightrag_language_detection_throughput_per_second gauge",
            f"lightrag_language_detection_throughput_per_second {metrics.detections_per_second}",
            "",
            "# HELP lightrag_language_detection_cache_hit_rate Cache hit rate percentage",
            "# TYPE lightrag_language_detection_cache_hit_rate gauge",
            f"lightrag_language_detection_cache_hit_rate {metrics.cache_hit_rate_percent}",
            "",
            "# HELP lightrag_language_detection_method_total Detection count by method",
            "# TYPE lightrag_language_detection_method_total counter",
        ]
        
        for method, count in method_dist.items():
            lines.append(f"lightrag_language_detection_method_total{{method=\"{method}\"}} {count}")
        
        lines.extend([
            "",
            "# HELP lightrag_language_detection_language_total Detection count by language",
            "# TYPE lightrag_language_detection_language_total counter",
        ])
        
        for language, count in language_dist.items():
            lines.append(f"lightrag_language_detection_language_total{{language=\"{language}\"}} {count}")
        
        if metrics.accuracy_percent is not None:
            lines.extend([
                "",
                "# HELP lightrag_language_detection_accuracy Overall detection accuracy percentage",
                "# TYPE lightrag_language_detection_accuracy gauge",
                f"lightrag_language_detection_accuracy {metrics.accuracy_percent}",
            ])
        
        if metrics.vietnamese_accuracy_percent is not None:
            lines.extend([
                "",
                "# HELP lightrag_language_detection_vietnamese_accuracy Vietnamese detection accuracy percentage",
                "# TYPE lightrag_language_detection_vietnamese_accuracy gauge",
                f"lightrag_language_detection_vietnamese_accuracy {metrics.vietnamese_accuracy_percent}",
            ])
        
        if metrics.english_accuracy_percent is not None:
            lines.extend([
                "",
                "# HELP lightrag_language_detection_english_accuracy English detection accuracy percentage",
                "# TYPE lightrag_language_detection_english_accuracy gauge",
                f"lightrag_language_detection_english_accuracy {metrics.english_accuracy_percent}",
            ])
        
        lines.extend([
            "",
            "# HELP lightrag_language_detection_error_rate Error rate percentage",
            "# TYPE lightrag_language_detection_error_rate gauge",
            f"lightrag_language_detection_error_rate {metrics.error_rate_percent}",
            ""
        ])
        
        return "\n".join(lines)


class HealthChecker:
    """
    Provides health check functionality for monitoring systems.
    
    This class implements health checks for all components of the language
    detection system and provides status information for monitoring integration.
    """
    
    def __init__(self, service):
        """
        Initialize health checker.
        
        Args:
            service: LanguageDetectionService instance to monitor
        """
        self.service = service
        self.metrics_collector = MetricsCollector()
        
        logger.info("HealthChecker initialized")
    
    def check_health(self) -> HealthStatus:
        """
        Perform comprehensive health check.
        
        Returns:
            HealthStatus with detailed component status
        """
        timestamp = time.time()
        components = {}
        overall_status = "healthy"
        messages = []
        
        # Check Unicode detector
        unicode_status = self._check_unicode_detector()
        components["unicode_detector"] = unicode_status
        
        if not unicode_status.get("available", False):
            overall_status = "unhealthy"
            messages.append("Unicode detector unavailable")
        
        # Check FastText detector
        fasttext_status = self._check_fasttext_detector()
        components["fasttext_detector"] = fasttext_status
        
        if self.service.config.enabled and not fasttext_status.get("available", False):
            overall_status = "degraded"
            messages.append("FastText enabled but unavailable")
        
        # Check hybrid strategy
        hybrid_status = self._check_hybrid_strategy()
        components["hybrid_strategy"] = hybrid_status
        
        # Check error handler and circuit breaker
        error_handler_status = self._check_error_handler()
        components["error_handler"] = error_handler_status
        
        if error_handler_status.get("circuit_breaker_open", False):
            overall_status = "degraded"
            messages.append("Circuit breaker open")
        
        # Get performance metrics
        performance_metrics = self.metrics_collector.get_performance_metrics()
        
        # Check error rate
        if performance_metrics.error_rate_percent > 10.0:
            overall_status = "degraded"
            messages.append(f"High error rate: {performance_metrics.error_rate_percent}%")
        
        # Check latency
        if performance_metrics.p95_latency_ms > 200.0:
            if overall_status == "healthy":
                overall_status = "degraded"
            messages.append(f"High latency: P95={performance_metrics.p95_latency_ms}ms")
        
        message = "; ".join(messages) if messages else "All systems operational"
        
        return HealthStatus(
            status=overall_status,
            timestamp=timestamp,
            components=components,
            metrics=performance_metrics,
            message=message
        )
    
    def _check_unicode_detector(self) -> Dict[str, Any]:
        """Check Unicode detector health."""
        try:
            available = (hasattr(self.service, 'unicode_detector') and 
                        self.service.unicode_detector is not None)
            
            status = {
                "available": available,
                "status": "healthy" if available else "unavailable"
            }
            
            if available:
                cache_stats = self.service.unicode_detector.get_cache_stats()
                status["cache_stats"] = cache_stats
            
            return status
            
        except Exception as e:
            logger.error(f"Unicode detector health check failed: {e}")
            return {
                "available": False,
                "status": "error",
                "error": str(e)
            }
    
    def _check_fasttext_detector(self) -> Dict[str, Any]:
        """Check FastText detector health."""
        try:
            available = self.service.is_fasttext_available()
            
            status = {
                "available": available,
                "enabled": self.service.config.enabled,
                "status": "healthy" if available else ("disabled" if not self.service.config.enabled else "unavailable")
            }
            
            if available and hasattr(self.service, 'fasttext_detector'):
                model_info = self.service.fasttext_detector.get_model_info()
                status["model_info"] = model_info
                status["cache_stats"] = self.service.fasttext_detector.get_cache_stats()
            
            return status
            
        except Exception as e:
            logger.error(f"FastText detector health check failed: {e}")
            return {
                "available": False,
                "enabled": self.service.config.enabled,
                "status": "error",
                "error": str(e)
            }
    
    def _check_hybrid_strategy(self) -> Dict[str, Any]:
        """Check hybrid strategy health."""
        try:
            available = (hasattr(self.service, 'hybrid_strategy') and 
                        self.service.hybrid_strategy is not None)
            
            status = {
                "available": available,
                "status": "healthy" if available else "unavailable"
            }
            
            if available:
                metrics = self.service.hybrid_strategy.get_detection_metrics()
                status["metrics"] = metrics
            
            return status
            
        except Exception as e:
            logger.error(f"Hybrid strategy health check failed: {e}")
            return {
                "available": False,
                "status": "error",
                "error": str(e)
            }
    
    def _check_error_handler(self) -> Dict[str, Any]:
        """Check error handler and circuit breaker health."""
        try:
            available = (hasattr(self.service, 'error_handler') and 
                        self.service.error_handler is not None)
            
            status = {
                "available": available,
                "status": "healthy" if available else "unavailable"
            }
            
            if available:
                status["circuit_breaker_open"] = self.service.error_handler.is_circuit_open()
                status["health_status"] = self.service.error_handler.get_health_status()
                status["error_stats"] = self.service.error_handler.get_error_stats()
            
            return status
            
        except Exception as e:
            logger.error(f"Error handler health check failed: {e}")
            return {
                "available": False,
                "status": "error",
                "error": str(e)
            }
    
    def check_model_availability(self) -> Dict[str, Any]:
        """
        Check FastText model availability specifically.
        
        Returns:
            Dictionary with model availability status
        """
        try:
            if not hasattr(self.service, 'fasttext_detector') or not self.service.fasttext_detector:
                return {
                    "available": False,
                    "reason": "FastText detector not initialized"
                }
            
            detector = self.service.fasttext_detector
            
            return {
                "available": detector.is_available(),
                "model_loaded": detector._model_loaded,
                "model_info": detector.get_model_info() if detector.is_available() else None,
                "config_enabled": self.service.config.enabled
            }
            
        except Exception as e:
            logger.error(f"Model availability check failed: {e}")
            return {
                "available": False,
                "error": str(e)
            }
