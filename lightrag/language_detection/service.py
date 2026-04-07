"""
High-level language detection service coordinating all detection methods.

This module implements the enhanced LanguageDetectionService that serves as the main
interface for language detection in LightRAG. It coordinates between FastText and
Unicode detection methods using a hybrid strategy while maintaining backward
compatibility with existing API signatures.
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict

from lightrag.language_detector import SupportedLanguage, LanguageDetector
from .config import FastTextConfig
from .fasttext_detector import FastTextDetector
from .hybrid_strategy import HybridDetectionStrategy
from .model_manager import FastTextModelManager
from .error_handler import ResilientErrorHandler, ErrorHandlingConfig
from .logger import get_fasttext_logger
from .metrics import MetricsCollector, HealthChecker, HealthStatus
from .security import RateLimiter, RateLimitConfig

# Use dedicated FastText logger
logger = get_fasttext_logger("service")


@dataclass
class DetectionResult:
    """Comprehensive detection result with metadata."""
    
    language: SupportedLanguage
    confidence: float
    method_used: str  # "fasttext", "unicode", "hybrid"
    processing_time_ms: float
    text_length: int
    cached: bool
    
    # Additional metadata for monitoring
    fasttext_available: bool
    unicode_fallback_used: bool
    hybrid_strategy_applied: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "language": self.language.value,
            "confidence": self.confidence,
            "method_used": self.method_used,
            "processing_time_ms": self.processing_time_ms,
            "text_length": self.text_length,
            "cached": self.cached,
            "fasttext_available": self.fasttext_available,
            "unicode_fallback_used": self.unicode_fallback_used,
            "hybrid_strategy_applied": self.hybrid_strategy_applied,
        }


class LanguageDetectionService:
    """
    High-level service coordinating language detection.
    
    This service provides the main interface for language detection in LightRAG,
    coordinating between FastText and Unicode detection methods through a hybrid
    strategy. It maintains backward compatibility with existing API while adding
    enhanced functionality and monitoring capabilities.
    
    Key Features:
    - Hybrid detection strategy combining FastText and Unicode methods
    - Backward compatible API preserving existing method signatures
    - Extended methods for detailed detection results and health checks
    - Integration with PromptManager and SupportedLanguage enum
    - Feature flag support for gradual rollout
    - Comprehensive logging and metrics collection
    """
    
    def __init__(self, config: Optional[FastTextConfig] = None):
        """
        Initialize service with configuration.
        
        Args:
            config: FastText configuration. If None, loads from environment.
        """
        self.config = config or FastTextConfig.from_env()
        
        # Validate configuration
        try:
            self.config.validate()
            logger.info("Configuration validation passed")
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            # Continue with invalid config but log the issue
            # This allows graceful degradation to Unicode-only mode
        
        # Initialize rate limiter (Requirement 10.5)
        rate_limit_config = RateLimitConfig(
            max_requests=100,  # Default: 100 requests per minute
            window_seconds=60,
            enabled=True,
            burst_allowance=10
        )
        self.rate_limiter = RateLimiter(rate_limit_config)
        logger.info("Rate limiter initialized for abuse prevention")
        
        # Initialize components
        self._initialize_components()
        
        # Initialize metrics collector and health checker
        self.metrics_collector = MetricsCollector(
            retention_window=3600,  # 1 hour retention
            max_samples=10000
        )
        self.health_checker = HealthChecker(self)
        
        # Performance tracking (legacy, kept for backward compatibility)
        self._total_detections = 0
        self._total_processing_time = 0.0
        self._method_usage_counts = {
            "fasttext": 0,
            "unicode": 0,
            "hybrid": 0,
            "fallback": 0
        }
        
        logger.info(f"LanguageDetectionService initialized successfully. "
                   f"FastText enabled: {self.config.enabled}, "
                   f"FastText available: {self.is_fasttext_available()}, "
                   f"Rate limiting enabled: {rate_limit_config.enabled}")
    
    def _initialize_components(self) -> None:
        """Initialize all detection components with error handling."""
        try:
            # Initialize Unicode detector (always available as fallback)
            self.unicode_detector = LanguageDetector(
                default_threshold=0.3,
                cache_size=self.config.cache_size
            )
            logger.debug("Unicode detector initialized successfully")
            
            # Initialize error handler
            error_config = ErrorHandlingConfig()
            self.error_handler = ResilientErrorHandler(error_config)
            logger.debug("Error handler initialized successfully")
            
            # Initialize FastText components if enabled
            if self.config.enabled:
                self._initialize_fasttext_components()
            else:
                logger.info("FastText detection disabled by configuration")
                self.fasttext_detector = None
                self.hybrid_strategy = None
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            # Ensure we have at least Unicode detection available
            if not hasattr(self, 'unicode_detector'):
                self.unicode_detector = LanguageDetector()
            self.fasttext_detector = None
            self.hybrid_strategy = None
    
    def _initialize_fasttext_components(self) -> None:
        """Initialize FastText-specific components."""
        try:
            # Initialize FastText detector
            self.fasttext_detector = FastTextDetector(
                config=self.config,
                error_handler=self.error_handler
            )
            
            # Initialize hybrid strategy
            self.hybrid_strategy = HybridDetectionStrategy(
                fasttext_detector=self.fasttext_detector,
                unicode_detector=self.unicode_detector,
                error_handler=self.error_handler
            )
            
            logger.info("FastText components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FastText components: {e}")
            self.fasttext_detector = None
            self.hybrid_strategy = None
    
    def detect_language(self, text: str, client_id: Optional[str] = None) -> SupportedLanguage:
        """
        Main detection interface maintaining backward compatibility.
        
        This method preserves the exact API signature of the original LanguageDetector
        to ensure backward compatibility with existing LightRAG components.
        
        Args:
            text: Input text to analyze
            client_id: Optional client identifier for rate limiting (e.g., IP address, user ID)
            
        Returns:
            SupportedLanguage enum value (Vietnamese or English)
            
        Raises:
            ValueError: If rate limit is exceeded
        """
        # Apply rate limiting if client_id is provided (Requirement 10.5)
        if client_id:
            allowed, retry_after = self.rate_limiter.check_rate_limit(client_id)
            if not allowed:
                logger.warning(f"Rate limit exceeded for client, retry after {retry_after}s")
                raise ValueError(f"Rate limit exceeded. Please retry after {retry_after} seconds.")
        
        start_time = time.time()
        self._total_detections += 1
        
        method_used = "unknown"
        success = True
        error_msg = None
        result = SupportedLanguage.ENGLISH  # Default
        
        try:
            # Handle empty or whitespace-only text
            if not text or not text.strip():
                logger.debug("Empty text detected, returning English default")
                method_used = "default"
                result = SupportedLanguage.ENGLISH
                return result
            
            # Use hybrid strategy if available, otherwise fall back to Unicode
            if self.hybrid_strategy and self.config.enabled:
                result = self.hybrid_strategy.detect(text)
                method_used = "hybrid"
                self._method_usage_counts["hybrid"] += 1
                
                if self.config.log_detection_method:
                    logger.debug(f"Hybrid detection result: {result.value}")
                
            else:
                # Fall back to Unicode detection
                result = self.unicode_detector.detect(text)
                method_used = "unicode"
                self._method_usage_counts["unicode"] += 1
                
                if self.config.log_detection_method:
                    logger.debug(f"Unicode fallback result: {result.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            # Emergency fallback to English
            method_used = "fallback"
            success = False
            error_msg = str(e)
            self._method_usage_counts["fallback"] += 1
            result = SupportedLanguage.ENGLISH
            return result
            
        finally:
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self._total_processing_time += processing_time
            
            # Record metrics
            self.metrics_collector.record_detection(
                language=result,
                confidence=0.5,  # Default confidence for backward compatible method
                method_used=method_used,
                latency_ms=processing_time,
                text_length=len(text),
                cached=False,  # Cannot determine from this method
                success=success,
                error=error_msg
            )
            
            if self.config.log_performance_metrics:
                logger.debug(f"Detection completed in {processing_time:.2f}ms")
    
    def get_detection_details(self, text: str) -> DetectionResult:
        """
        Extended interface returning comprehensive detection results.
        
        This method provides detailed information about the detection process
        including confidence scores, method used, and performance metrics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectionResult with comprehensive metadata
        """
        start_time = time.time()
        self._total_detections += 1
        
        method_used = "unknown"
        success = True
        error_msg = None
        
        try:
            # Handle empty or whitespace-only text
            if not text or not text.strip():
                result = DetectionResult(
                    language=SupportedLanguage.ENGLISH,
                    confidence=1.0,
                    method_used="default",
                    processing_time_ms=0.0,
                    text_length=len(text),
                    cached=False,
                    fasttext_available=self.is_fasttext_available(),
                    unicode_fallback_used=False,
                    hybrid_strategy_applied=False
                )
                
                # Record metrics
                self.metrics_collector.record_detection(
                    language=result.language,
                    confidence=result.confidence,
                    method_used=result.method_used,
                    latency_ms=result.processing_time_ms,
                    text_length=result.text_length,
                    cached=result.cached,
                    success=True
                )
                
                return result
            
            # Determine detection method and execute
            if self.hybrid_strategy and self.config.enabled and self.is_fasttext_available():
                # Use hybrid strategy with detailed tracking
                result = self._execute_hybrid_detection_detailed(text, start_time)
                self._method_usage_counts["hybrid"] += 1
                method_used = result.method_used
                
            elif self.fasttext_detector and self.config.enabled and self.is_fasttext_available():
                # Use FastText directly if hybrid strategy not available
                result = self._execute_fasttext_detection_detailed(text, start_time)
                self._method_usage_counts["fasttext"] += 1
                method_used = result.method_used
                
            else:
                # Fall back to Unicode detection
                result = self._execute_unicode_detection_detailed(text, start_time)
                self._method_usage_counts["unicode"] += 1
                method_used = result.method_used
            
            # Record metrics
            self.metrics_collector.record_detection(
                language=result.language,
                confidence=result.confidence,
                method_used=result.method_used,
                latency_ms=result.processing_time_ms,
                text_length=result.text_length,
                cached=result.cached,
                success=True
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Detailed detection failed: {e}")
            # Emergency fallback with error information
            processing_time = (time.time() - start_time) * 1000
            self._method_usage_counts["fallback"] += 1
            success = False
            error_msg = str(e)
            
            result = DetectionResult(
                language=SupportedLanguage.ENGLISH,
                confidence=0.1,
                method_used="error_fallback",
                processing_time_ms=processing_time,
                text_length=len(text),
                cached=False,
                fasttext_available=False,
                unicode_fallback_used=True,
                hybrid_strategy_applied=False
            )
            
            # Record error metrics
            self.metrics_collector.record_detection(
                language=result.language,
                confidence=result.confidence,
                method_used=result.method_used,
                latency_ms=result.processing_time_ms,
                text_length=result.text_length,
                cached=result.cached,
                success=False,
                error=error_msg
            )
            
            return result
        finally:
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self._total_processing_time += processing_time
    
    def _execute_hybrid_detection_detailed(self, text: str, start_time: float) -> DetectionResult:
        """Execute hybrid detection with detailed result tracking."""
        # Get FastText detection details first
        fasttext_language, fasttext_confidence = self.fasttext_detector.detect(text)
        
        # Check if result was cached
        cache_stats_before = self.fasttext_detector.get_cache_stats()
        
        # Execute hybrid strategy
        final_language = self.hybrid_strategy.detect(text)
        
        # Check if result was cached (approximate)
        cache_stats_after = self.fasttext_detector.get_cache_stats()
        was_cached = cache_stats_after['hits'] > cache_stats_before['hits']
        
        # Determine which method was actually used by comparing results
        if final_language == fasttext_language and fasttext_confidence >= self.config.high_confidence_threshold:
            method_used = "fasttext_exclusive"
            confidence = fasttext_confidence
            unicode_fallback_used = False
        elif fasttext_confidence < self.config.low_confidence_threshold:
            method_used = "unicode_fallback"
            confidence = self._estimate_unicode_confidence(text, final_language)
            unicode_fallback_used = True
        else:
            method_used = "hybrid_comparison"
            confidence = max(fasttext_confidence, self._estimate_unicode_confidence(text, final_language))
            unicode_fallback_used = final_language != fasttext_language
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            language=final_language,
            confidence=confidence,
            method_used=method_used,
            processing_time_ms=processing_time,
            text_length=len(text),
            cached=was_cached,
            fasttext_available=True,
            unicode_fallback_used=unicode_fallback_used,
            hybrid_strategy_applied=True
        )
    
    def _execute_fasttext_detection_detailed(self, text: str, start_time: float) -> DetectionResult:
        """Execute FastText detection with detailed result tracking."""
        cache_stats_before = self.fasttext_detector.get_cache_stats()
        
        language, confidence = self.fasttext_detector.detect(text)
        
        cache_stats_after = self.fasttext_detector.get_cache_stats()
        was_cached = cache_stats_after['hits'] > cache_stats_before['hits']
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            language=language,
            confidence=confidence,
            method_used="fasttext",
            processing_time_ms=processing_time,
            text_length=len(text),
            cached=was_cached,
            fasttext_available=True,
            unicode_fallback_used=False,
            hybrid_strategy_applied=False
        )
    
    def _execute_unicode_detection_detailed(self, text: str, start_time: float) -> DetectionResult:
        """Execute Unicode detection with detailed result tracking."""
        cache_stats_before = self.unicode_detector.get_cache_stats()
        
        language = self.unicode_detector.detect(text)
        confidence = self._estimate_unicode_confidence(text, language)
        
        cache_stats_after = self.unicode_detector.get_cache_stats()
        was_cached = cache_stats_after['hits'] > cache_stats_before['hits']
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            language=language,
            confidence=confidence,
            method_used="unicode",
            processing_time_ms=processing_time,
            text_length=len(text),
            cached=was_cached,
            fasttext_available=self.is_fasttext_available(),
            unicode_fallback_used=True,
            hybrid_strategy_applied=False
        )
    
    def _estimate_unicode_confidence(self, text: str, result: SupportedLanguage) -> float:
        """
        Estimate confidence for Unicode detection results.
        
        Args:
            text: Input text
            result: Unicode detection result
            
        Returns:
            Estimated confidence score between 0.0 and 1.0
        """
        try:
            # Count Vietnamese and total characters
            vietnamese_chars = 0
            total_chars = 0
            
            for char in text:
                if char.isalpha():
                    total_chars += 1
                    if self._is_vietnamese_character(char):
                        vietnamese_chars += 1
            
            if total_chars == 0:
                return 0.6  # Neutral confidence for non-letter text
            
            vietnamese_ratio = vietnamese_chars / total_chars
            threshold = self.unicode_detector.default_threshold
            
            if result == SupportedLanguage.VIETNAMESE:
                # Confidence increases with Vietnamese character ratio
                if vietnamese_ratio >= threshold:
                    normalized_ratio = (vietnamese_ratio - threshold) / (1.0 - threshold)
                    confidence = 0.5 + (normalized_ratio * 0.45)
                else:
                    confidence = 0.3
            else:
                # Confidence increases with lower Vietnamese ratio
                if vietnamese_ratio < threshold:
                    normalized_ratio = (threshold - vietnamese_ratio) / threshold
                    confidence = 0.5 + (normalized_ratio * 0.45)
                else:
                    confidence = 0.3
            
            return min(0.95, max(0.1, confidence))
            
        except Exception as e:
            logger.warning(f"Failed to estimate Unicode confidence: {e}")
            return 0.6
    
    def _is_vietnamese_character(self, char: str) -> bool:
        """Check if character is Vietnamese (replicates LanguageDetector logic)."""
        vietnamese_ranges = [
            (0x00C0, 0x00C3), (0x00C8, 0x00CA), (0x00CC, 0x00CD),
            (0x00D2, 0x00D5), (0x00D9, 0x00DA), (0x00DD, 0x00DD),
            (0x00E0, 0x00E3), (0x00E8, 0x00EA), (0x00EC, 0x00ED),
            (0x00F2, 0x00F5), (0x00F9, 0x00FA), (0x00FD, 0x00FD),
            (0x0102, 0x0103), (0x0110, 0x0111), (0x0128, 0x0129),
            (0x0168, 0x0169), (0x01A0, 0x01A1), (0x01AF, 0x01B0),
            (0x1EA0, 0x1EF9),
        ]
        
        code_point = ord(char)
        for start, end in vietnamese_ranges:
            if start <= code_point <= end:
                return True
        return False
    
    def detect_batch(self, texts: List[str]) -> List[SupportedLanguage]:
        """
        Detect language for multiple texts efficiently.
        
        This method maintains backward compatibility while leveraging batch
        processing capabilities of the underlying detectors.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of SupportedLanguage results in same order as input
        """
        if not texts:
            return []
        
        try:
            # Use FastText batch processing if available
            if self.fasttext_detector and self.config.enabled and self.is_fasttext_available():
                batch_results = self.fasttext_detector.detect_batch(texts)
                results = [result[0] for result in batch_results]  # Extract just the language
                self._method_usage_counts["fasttext"] += len(texts)
                
                if self.config.log_detection_method:
                    logger.debug(f"FastText batch detection: {len(texts)} texts processed")
                
                return results
            else:
                # Fall back to Unicode batch processing
                results = self.unicode_detector.detect_batch(texts)
                self._method_usage_counts["unicode"] += len(texts)
                
                if self.config.log_detection_method:
                    logger.debug(f"Unicode batch detection: {len(texts)} texts processed")
                
                return results
                
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            # Emergency fallback to individual detection
            return [self.detect_language(text) for text in texts]
    
    def is_vietnamese(self, text: str) -> bool:
        """
        Quick check if text is Vietnamese (backward compatibility method).
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if text is classified as Vietnamese, False otherwise
        """
        return self.detect_language(text) == SupportedLanguage.VIETNAMESE
    
    def health_check(self) -> Dict[str, Any]:
        """
        Health status for monitoring integration.
        
        Returns:
            Dictionary with comprehensive health information
        """
        try:
            # Use the dedicated health checker
            health_status = self.health_checker.check_health()
            return health_status.to_dict()
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "timestamp": time.time(),
                "service_available": False,
                "error": str(e),
                "message": "Health check failed"
            }
    
    def is_fasttext_available(self) -> bool:
        """
        Check if FastText detection is available.
        
        Returns:
            True if FastText is available and functional
        """
        return (hasattr(self, 'fasttext_detector') and 
                self.fasttext_detector is not None and 
                self.fasttext_detector.is_available())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        avg_processing_time = (self._total_processing_time / self._total_detections) if self._total_detections > 0 else 0.0
        
        # Calculate method usage percentages
        method_percentages = {}
        if self._total_detections > 0:
            for method, count in self._method_usage_counts.items():
                method_percentages[f"{method}_percent"] = round((count / self._total_detections) * 100, 2)
        
        metrics = {
            "total_detections": self._total_detections,
            "total_processing_time_ms": round(self._total_processing_time, 2),
            "average_processing_time_ms": round(avg_processing_time, 2),
            "method_usage_counts": self._method_usage_counts.copy(),
            "method_usage_percentages": method_percentages,
            "service_health": {
                "fasttext_available": self.is_fasttext_available(),
                "unicode_available": hasattr(self, 'unicode_detector') and self.unicode_detector is not None,
                "hybrid_strategy_available": hasattr(self, 'hybrid_strategy') and self.hybrid_strategy is not None
            }
        }
        
        # Add component-specific metrics
        if self.is_fasttext_available():
            metrics["fasttext_metrics"] = self.fasttext_detector.get_performance_metrics()
        
        if hasattr(self, 'unicode_detector') and self.unicode_detector:
            metrics["unicode_metrics"] = self.unicode_detector.get_performance_metrics()
        
        if hasattr(self, 'hybrid_strategy') and self.hybrid_strategy:
            metrics["hybrid_strategy_metrics"] = self.hybrid_strategy.get_detection_metrics()
        
        if hasattr(self, 'error_handler') and self.error_handler:
            metrics["error_handler_metrics"] = self.error_handler.get_error_stats()
        
        # Add metrics collector data
        if hasattr(self, 'metrics_collector'):
            metrics["detailed_metrics"] = asdict(self.metrics_collector.get_performance_metrics())
            metrics["method_distribution"] = self.metrics_collector.get_method_distribution()
            metrics["language_distribution"] = self.metrics_collector.get_language_distribution()
        
        # Add rate limiter statistics (Requirement 10.5)
        if hasattr(self, 'rate_limiter'):
            metrics["rate_limiter"] = self.rate_limiter.get_stats()
        
        return metrics
    
    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus text format.
        
        This method provides Prometheus-compatible metrics export for integration
        with monitoring systems like Prometheus and Grafana.
        
        Returns:
            String containing Prometheus-formatted metrics
        """
        if hasattr(self, 'metrics_collector'):
            return self.metrics_collector.export_prometheus_metrics()
        else:
            logger.warning("Metrics collector not available, returning empty metrics")
            return "# No metrics available\n"
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent detection errors for debugging.
        
        Args:
            limit: Maximum number of errors to return
            
        Returns:
            List of recent error records
        """
        if hasattr(self, 'metrics_collector'):
            return self.metrics_collector.get_recent_errors(limit)
        else:
            return []
    
    def record_accuracy_sample(self, predicted: SupportedLanguage, actual: SupportedLanguage) -> None:
        """
        Record an accuracy sample for evaluation.
        
        This method allows external systems to provide ground truth labels
        for accuracy tracking and monitoring.
        
        Args:
            predicted: Predicted language from detection
            actual: Actual/ground truth language
        """
        if hasattr(self, 'metrics_collector'):
            self.metrics_collector.record_accuracy(predicted, actual)
            logger.debug(f"Accuracy sample recorded: predicted={predicted.value}, actual={actual.value}")
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics and statistics."""
        logger.info("Resetting LanguageDetectionService metrics")
        
        self._total_detections = 0
        self._total_processing_time = 0.0
        self._method_usage_counts = {method: 0 for method in self._method_usage_counts}
        
        # Reset component metrics
        if hasattr(self, 'unicode_detector') and self.unicode_detector:
            self.unicode_detector.clear_cache()
        
        if self.is_fasttext_available():
            self.fasttext_detector.reset_cache_stats()
        
        if hasattr(self, 'hybrid_strategy') and self.hybrid_strategy:
            self.hybrid_strategy.reset_metrics()
        
        if hasattr(self, 'error_handler') and self.error_handler:
            self.error_handler.reset_stats()
        
        # Reset metrics collector
        if hasattr(self, 'metrics_collector'):
            self.metrics_collector.reset_metrics()
    
    def update_configuration(self, **kwargs) -> None:
        """
        Update runtime configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Raises:
            ValueError: If invalid parameters or values provided
        """
        try:
            # Update configuration
            self.config.update_runtime_config(**kwargs)
            
            # Apply configuration changes to components
            if hasattr(self, 'fasttext_detector') and self.fasttext_detector:
                # Update FastText detector thresholds
                if 'vietnamese_threshold' in kwargs or 'english_threshold' in kwargs:
                    self.fasttext_detector.confidence_thresholds = {
                        'vietnamese': self.config.vietnamese_threshold,
                        'english': self.config.english_threshold
                    }
            
            logger.info(f"Configuration updated successfully: {kwargs}")
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            raise
    
    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current configuration parameters.
        
        Returns:
            Dictionary with current configuration
        """
        config_dict = self.config.to_dict()
        
        # Add runtime status information
        config_dict.update({
            "runtime_status": {
                "fasttext_available": self.is_fasttext_available(),
                "unicode_available": hasattr(self, 'unicode_detector') and self.unicode_detector is not None,
                "hybrid_strategy_enabled": hasattr(self, 'hybrid_strategy') and self.hybrid_strategy is not None,
                "circuit_breaker_open": (hasattr(self, 'error_handler') and 
                                       self.error_handler and 
                                       self.error_handler.is_circuit_open()),
            }
        })
        
        return config_dict