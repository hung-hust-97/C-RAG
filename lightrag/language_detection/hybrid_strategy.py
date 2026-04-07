"""
Hybrid detection strategy coordinating FastText and Unicode detection methods.

This module implements the hybrid detection logic that intelligently combines
FastText and Unicode detection methods based on confidence thresholds and text
characteristics to achieve optimal accuracy across different text types.
"""

import time
from typing import Dict, Any, Tuple, Optional
from lightrag.language_detector import SupportedLanguage, LanguageDetector
from .fasttext_detector import FastTextDetector
from .config import FastTextConfig
from .error_handler import ResilientErrorHandler, ErrorHandlingConfig
from .logger import get_fasttext_logger

# Use dedicated FastText logger
logger = get_fasttext_logger("hybrid_strategy")


class HybridDetectionStrategy:
    """
    Coordinates between FastText and Unicode detection methods.
    
    This class implements the hybrid detection strategy that combines FastText
    and Unicode detection methods based on confidence thresholds and text
    characteristics. The strategy follows these rules:
    
    1. When FastText confidence > 0.8: Use FastText result exclusively
    2. When FastText confidence 0.5-0.8: Compare with Unicode and use higher confidence
    3. When FastText confidence < 0.5: Fall back to Unicode detection
    4. For text < 20 characters: Prefer Unicode for Vietnamese character-heavy text
    5. Log detection method selection and performance metrics
    """
    
    def __init__(self, 
                 fasttext_detector: FastTextDetector,
                 unicode_detector: LanguageDetector,
                 strategy_config: Optional[Dict[str, Any]] = None,
                 error_handler: Optional[ResilientErrorHandler] = None):
        """
        Initialize hybrid strategy with both detectors and error handling.
        
        Args:
            fasttext_detector: FastText-based language detector
            unicode_detector: Unicode-based language detector (existing)
            strategy_config: Optional configuration overrides for strategy parameters
            error_handler: Error handler for resilient operations
        """
        self.fasttext_detector = fasttext_detector
        self.unicode_detector = unicode_detector
        self.error_handler = error_handler or ResilientErrorHandler(ErrorHandlingConfig())
        
        # Get configuration from FastText detector or use defaults
        self.config = fasttext_detector.config if hasattr(fasttext_detector, 'config') else FastTextConfig.from_env()
        
        # Apply strategy-specific configuration overrides
        if strategy_config:
            for key, value in strategy_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.debug(f"Strategy config override: {key} = {value}")
        
        # Performance tracking
        self._detection_counts = {
            'fasttext_exclusive': 0,
            'unicode_fallback': 0,
            'hybrid_comparison': 0,
            'short_text_unicode': 0,
            'fasttext_unavailable': 0,
            'circuit_breaker_fallback': 0,
            'total_detections': 0
        }
        
        self._performance_metrics = {
            'total_time_ms': 0.0,
            'fasttext_time_ms': 0.0,
            'unicode_time_ms': 0.0,
            'hybrid_time_ms': 0.0
        }
        
        logger.info(f"HybridDetectionStrategy initialized with thresholds: "
                   f"high={self.config.high_confidence_threshold}, "
                   f"low={self.config.low_confidence_threshold}, "
                   f"resilient error handling enabled")
    
    def detect(self, text: str) -> SupportedLanguage:
        """
        Apply hybrid detection logic based on confidence thresholds.
        
        The hybrid strategy follows this decision tree:
        1. Check text length - prefer Unicode for very short text with Vietnamese chars
        2. Try FastText detection if available
        3. Apply confidence-based decision logic:
           - High confidence (>0.8): Use FastText exclusively
           - Medium confidence (0.5-0.8): Compare with Unicode, use higher confidence
           - Low confidence (<0.5): Fall back to Unicode
        4. Log detection method and performance metrics
        
        Args:
            text: Input text to analyze
            
        Returns:
            SupportedLanguage enum value (Vietnamese or English)
        """
        start_time = time.time()
        self._detection_counts['total_detections'] += 1
        
        try:
            # Handle empty or whitespace-only text
            if not text or not text.strip():
                logger.debug("Empty text detected, returning English default")
                return SupportedLanguage.ENGLISH
            
            # Check for very short text with Vietnamese characters
            if len(text) < 20:
                return self._handle_short_text(text, start_time)
            
            # Check if FastText is available or circuit breaker is open
            if not self.fasttext_detector.is_available() or self.error_handler.is_circuit_open():
                return self._handle_fasttext_unavailable(text, start_time)
            
            # Perform FastText detection
            fasttext_start = time.time()
            fasttext_language, fasttext_confidence = self.fasttext_detector.detect(text)
            fasttext_time = (time.time() - fasttext_start) * 1000
            self._performance_metrics['fasttext_time_ms'] += fasttext_time
            
            # Apply hybrid decision logic based on confidence thresholds
            result = self._apply_hybrid_logic(text, fasttext_language, fasttext_confidence, start_time)
            
            # Log detection method and performance if enabled
            if self.config.log_detection_method:
                total_time = (time.time() - start_time) * 1000
                logger.debug(f"Hybrid detection: {result.value} "
                           f"(FastText: {fasttext_language.value}, conf: {fasttext_confidence:.3f}, "
                           f"time: {total_time:.1f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid detection failed: {e}")
            # Emergency fallback to Unicode detection
            return self._handle_detection_error(text, start_time)
        finally:
            # Update total performance metrics
            total_time = (time.time() - start_time) * 1000
            self._performance_metrics['total_time_ms'] += total_time
    
    def _handle_short_text(self, text: str, start_time: float) -> SupportedLanguage:
        """
        Handle short text detection using Unicode method for Vietnamese character-heavy text.
        
        Args:
            text: Input text (< 20 characters)
            start_time: Detection start time for metrics
            
        Returns:
            SupportedLanguage based on Unicode detection
        """
        self._detection_counts['short_text_unicode'] += 1
        
        unicode_start = time.time()
        result = self.unicode_detector.detect(text)
        unicode_time = (time.time() - unicode_start) * 1000
        self._performance_metrics['unicode_time_ms'] += unicode_time
        
        if self.config.log_detection_method:
            logger.debug(f"Short text detection: {result.value} (Unicode method, length: {len(text)})")
        
        return result
    
    def _handle_fasttext_unavailable(self, text: str, start_time: float) -> SupportedLanguage:
        """
        Handle detection when FastText is unavailable or circuit breaker is open.
        
        Args:
            text: Input text
            start_time: Detection start time for metrics
            
        Returns:
            SupportedLanguage based on Unicode detection
        """
        if self.error_handler.is_circuit_open():
            self._detection_counts['circuit_breaker_fallback'] += 1
            logger.debug("Circuit breaker open, using Unicode fallback")
        else:
            self._detection_counts['fasttext_unavailable'] += 1
            logger.debug("FastText unavailable, using Unicode fallback")
        
        unicode_start = time.time()
        result = self.unicode_detector.detect(text)
        unicode_time = (time.time() - unicode_start) * 1000
        self._performance_metrics['unicode_time_ms'] += unicode_time
        
        if self.config.log_detection_method:
            reason = "circuit_breaker_open" if self.error_handler.is_circuit_open() else "fasttext_unavailable"
            logger.debug(f"FastText unavailable ({reason}), using Unicode: {result.value}")
        
        return result
    
    def _apply_hybrid_logic(self, text: str, fasttext_language: SupportedLanguage, 
                           fasttext_confidence: float, start_time: float) -> SupportedLanguage:
        """
        Apply the core hybrid detection logic based on confidence thresholds.
        
        Args:
            text: Input text
            fasttext_language: Language detected by FastText
            fasttext_confidence: Confidence score from FastText
            start_time: Detection start time for metrics
            
        Returns:
            Final language detection result
        """
        # High confidence: Use FastText exclusively
        if fasttext_confidence >= self.config.high_confidence_threshold:
            self._detection_counts['fasttext_exclusive'] += 1
            
            if self.config.log_detection_method:
                logger.debug(f"High confidence FastText: {fasttext_language.value} "
                           f"(conf: {fasttext_confidence:.3f})")
            
            return fasttext_language
        
        # Low confidence: Fall back to Unicode
        elif fasttext_confidence < self.config.low_confidence_threshold:
            self._detection_counts['unicode_fallback'] += 1
            
            unicode_start = time.time()
            unicode_result = self.unicode_detector.detect(text)
            unicode_time = (time.time() - unicode_start) * 1000
            self._performance_metrics['unicode_time_ms'] += unicode_time
            
            if self.config.log_detection_method:
                logger.debug(f"Low confidence FastText, using Unicode: {unicode_result.value} "
                           f"(FastText conf: {fasttext_confidence:.3f})")
            
            return unicode_result
        
        # Medium confidence: Compare with Unicode and use method with higher confidence
        else:
            return self._compare_methods(text, fasttext_language, fasttext_confidence, start_time)
    
    def _compare_methods(self, text: str, fasttext_language: SupportedLanguage, 
                        fasttext_confidence: float, start_time: float) -> SupportedLanguage:
        """
        Compare FastText and Unicode methods and use the one with higher confidence.
        
        Args:
            text: Input text
            fasttext_language: Language detected by FastText
            fasttext_confidence: Confidence score from FastText
            start_time: Detection start time for metrics
            
        Returns:
            Language from the method with higher confidence
        """
        self._detection_counts['hybrid_comparison'] += 1
        
        hybrid_start = time.time()
        
        # Get Unicode detection result
        unicode_start = time.time()
        unicode_result = self.unicode_detector.detect(text)
        unicode_time = (time.time() - unicode_start) * 1000
        self._performance_metrics['unicode_time_ms'] += unicode_time
        
        # Calculate Unicode confidence based on character ratio
        # This is an approximation since the original detector doesn't return confidence
        unicode_confidence = self._estimate_unicode_confidence(text, unicode_result)
        
        # Choose method with higher confidence
        if fasttext_confidence >= unicode_confidence:
            chosen_method = "FastText"
            result = fasttext_language
            final_confidence = fasttext_confidence
        else:
            chosen_method = "Unicode"
            result = unicode_result
            final_confidence = unicode_confidence
        
        hybrid_time = (time.time() - hybrid_start) * 1000
        self._performance_metrics['hybrid_time_ms'] += hybrid_time
        
        if self.config.log_detection_method:
            logger.debug(f"Hybrid comparison: {result.value} via {chosen_method} "
                       f"(FastText: {fasttext_confidence:.3f}, Unicode: {unicode_confidence:.3f})")
        
        return result
    
    def _estimate_unicode_confidence(self, text: str, unicode_result: SupportedLanguage) -> float:
        """
        Estimate confidence score for Unicode detection based on character analysis.
        
        This provides an approximate confidence score for Unicode detection to enable
        fair comparison with FastText confidence scores.
        
        Args:
            text: Input text
            unicode_result: Result from Unicode detection
            
        Returns:
            Estimated confidence score between 0.0 and 1.0
        """
        try:
            # Count Vietnamese and total letter characters
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
            
            if unicode_result == SupportedLanguage.VIETNAMESE:
                # For Vietnamese detection, confidence increases with Vietnamese character ratio
                # Scale from 0.5 (at threshold) to 0.95 (at 100% Vietnamese chars)
                threshold = self.unicode_detector.default_threshold
                if vietnamese_ratio >= threshold:
                    # Map ratio above threshold to confidence 0.5-0.95
                    normalized_ratio = (vietnamese_ratio - threshold) / (1.0 - threshold)
                    confidence = 0.5 + (normalized_ratio * 0.45)
                else:
                    # This shouldn't happen if Unicode detector is working correctly
                    confidence = 0.3
            else:
                # For English detection, confidence increases with lower Vietnamese ratio
                # Scale from 0.5 (at threshold) to 0.95 (at 0% Vietnamese chars)
                threshold = self.unicode_detector.default_threshold
                if vietnamese_ratio < threshold:
                    # Map ratio below threshold to confidence 0.5-0.95
                    normalized_ratio = (threshold - vietnamese_ratio) / threshold
                    confidence = 0.5 + (normalized_ratio * 0.45)
                else:
                    # This shouldn't happen if Unicode detector is working correctly
                    confidence = 0.3
            
            return min(0.95, max(0.1, confidence))
            
        except Exception as e:
            logger.warning(f"Failed to estimate Unicode confidence: {e}")
            return 0.6  # Default moderate confidence
    
    def _is_vietnamese_character(self, char: str) -> bool:
        """
        Check if a character is Vietnamese (replicates logic from LanguageDetector).
        
        Args:
            char: Single character to check
            
        Returns:
            True if character is Vietnamese, False otherwise
        """
        # Use the same Vietnamese ranges as the original LanguageDetector
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
    
    def _handle_detection_error(self, text: str, start_time: float) -> SupportedLanguage:
        """
        Handle detection errors by falling back to Unicode detection.
        
        Args:
            text: Input text
            start_time: Detection start time for metrics
            
        Returns:
            SupportedLanguage from Unicode fallback
        """
        try:
            unicode_start = time.time()
            result = self.unicode_detector.detect(text)
            unicode_time = (time.time() - unicode_start) * 1000
            self._performance_metrics['unicode_time_ms'] += unicode_time
            
            logger.warning(f"Detection error fallback to Unicode: {result.value}")
            return result
            
        except Exception as e:
            logger.error(f"Unicode fallback also failed: {e}")
            # Ultimate fallback to English
            return SupportedLanguage.ENGLISH
    
    def get_detection_metrics(self) -> Dict[str, Any]:
        """
        Return comprehensive performance metrics for monitoring.
        
        Returns:
            Dictionary containing detection counts, performance metrics, and configuration
        """
        total_detections = self._detection_counts['total_detections']
        
        # Calculate percentages
        method_percentages = {}
        if total_detections > 0:
            for method, count in self._detection_counts.items():
                if method != 'total_detections':
                    method_percentages[f"{method}_percent"] = round((count / total_detections) * 100, 2)
        
        # Calculate average times
        avg_times = {}
        if total_detections > 0:
            for metric, total_time in self._performance_metrics.items():
                avg_times[f"avg_{metric}"] = round(total_time / total_detections, 2)
        
        return {
            'detection_counts': self._detection_counts.copy(),
            'method_distribution': method_percentages,
            'performance_metrics': self._performance_metrics.copy(),
            'average_performance': avg_times,
            'configuration': {
                'high_confidence_threshold': self.config.high_confidence_threshold,
                'low_confidence_threshold': self.config.low_confidence_threshold,
                'log_detection_method': self.config.log_detection_method,
                'log_performance_metrics': self.config.log_performance_metrics,
            },
            'detector_status': {
                'fasttext_available': self.fasttext_detector.is_available(),
                'unicode_available': True,  # Always available
                'circuit_breaker_open': self.error_handler.is_circuit_open(),
                'error_handler_stats': self.error_handler.get_error_stats(),
                'fasttext_model_info': self.fasttext_detector.get_model_info() if self.fasttext_detector.is_available() else None,
                'unicode_cache_stats': self.unicode_detector.get_cache_stats(),
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics and detection counts."""
        self._detection_counts = {key: 0 for key in self._detection_counts}
        self._performance_metrics = {key: 0.0 for key in self._performance_metrics}
        logger.info("Hybrid detection strategy metrics reset")
    
    def get_strategy_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze the effectiveness of the hybrid strategy.
        
        Returns:
            Dictionary with strategy effectiveness analysis
        """
        total = self._detection_counts['total_detections']
        if total == 0:
            return {'status': 'no_data', 'message': 'No detections performed yet'}
        
        # Calculate method usage distribution
        fasttext_usage = (
            self._detection_counts['fasttext_exclusive'] + 
            self._detection_counts['hybrid_comparison']
        ) / total * 100
        
        unicode_usage = (
            self._detection_counts['unicode_fallback'] + 
            self._detection_counts['short_text_unicode'] +
            self._detection_counts['fasttext_unavailable']
        ) / total * 100
        
        # Calculate performance efficiency
        avg_total_time = self._performance_metrics['total_time_ms'] / total
        
        effectiveness = {
            'total_detections': total,
            'method_usage': {
                'fasttext_usage_percent': round(fasttext_usage, 2),
                'unicode_usage_percent': round(unicode_usage, 2),
                'hybrid_comparison_percent': round(self._detection_counts['hybrid_comparison'] / total * 100, 2),
            },
            'performance': {
                'avg_detection_time_ms': round(avg_total_time, 2),
                'fasttext_available_percent': round((total - self._detection_counts['fasttext_unavailable']) / total * 100, 2),
            },
            'strategy_health': {
                'high_confidence_rate': round(self._detection_counts['fasttext_exclusive'] / total * 100, 2),
                'fallback_rate': round(self._detection_counts['unicode_fallback'] / total * 100, 2),
                'short_text_rate': round(self._detection_counts['short_text_unicode'] / total * 100, 2),
            }
        }
        
        return effectiveness