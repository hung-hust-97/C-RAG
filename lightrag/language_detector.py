"""
Language detection module for Vietnamese and English text.

This module provides language detection capabilities for LightRAG's prompt localization
system. It uses Unicode character range analysis to determine if text is Vietnamese or English.
"""

import hashlib
import logging
import unicodedata
from collections import OrderedDict
from enum import Enum
from typing import Optional


class SupportedLanguage(Enum):
    """Enumeration of supported languages for prompt localization."""
    
    ENGLISH = "English"
    VIETNAMESE = "Vietnamese"


# Configure logger for this module
logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Detects language from text input using Unicode character analysis.
    
    The detector analyzes the ratio of Vietnamese characters to total letter characters
    in the input text. Vietnamese characters are identified by their Unicode ranges,
    which include Latin characters with Vietnamese diacritical marks and the special
    consonant đ (d with stroke).
    
    Vietnamese Unicode Ranges:
    - U+00C0-U+00C3: À, Á, Â, Ã
    - U+00C8-U+00CA: È, É, Ê
    - U+00CC-U+00CD: Ì, Í
    - U+00D2-U+00D5: Ò, Ó, Ô, Õ
    - U+00D9-U+00DA: Ù, Ú
    - U+00DD: Ý
    - U+0102-U+0103: Ă, ă
    - U+0110-U+0111: Đ, đ
    - U+0128-U+0129: Ĩ, ĩ
    - U+0168-U+0169: Ũ, ũ
    - U+01A0-U+01A1: Ơ, ơ
    - U+01AF-U+01B0: Ư, ư
    - U+1EA0-U+1EF9: Extended Vietnamese characters (all vowels with tone marks)
    
    Attributes:
        default_threshold (float): Default threshold for Vietnamese character ratio (0.3)
    """
    
    # Vietnamese Unicode character ranges
    VIETNAMESE_RANGES = [
        (0x00C0, 0x00C3),  # À, Á, Â, Ã (uppercase)
        (0x00C8, 0x00CA),  # È, É, Ê (uppercase)
        (0x00CC, 0x00CD),  # Ì, Í (uppercase)
        (0x00D2, 0x00D5),  # Ò, Ó, Ô, Õ (uppercase)
        (0x00D9, 0x00DA),  # Ù, Ú (uppercase)
        (0x00DD, 0x00DD),  # Ý (uppercase)
        (0x00E0, 0x00E3),  # à, á, â, ã (lowercase)
        (0x00E8, 0x00EA),  # è, é, ê (lowercase)
        (0x00EC, 0x00ED),  # ì, í (lowercase)
        (0x00F2, 0x00F5),  # ò, ó, ô, õ (lowercase)
        (0x00F9, 0x00FA),  # ù, ú (lowercase)
        (0x00FD, 0x00FD),  # ý (lowercase)
        (0x0102, 0x0103),  # Ă, ă
        (0x0110, 0x0111),  # Đ, đ
        (0x0128, 0x0129),  # Ĩ, ĩ
        (0x0168, 0x0169),  # Ũ, ũ
        (0x01A0, 0x01A1),  # Ơ, ơ
        (0x01AF, 0x01B0),  # Ư, ư
        (0x1EA0, 0x1EF9),  # Extended Vietnamese characters (all tone marks)
    ]
    
    def __init__(self, default_threshold: float = 0.3, cache_size: int = 10000):
        """
        Initialize the LanguageDetector.
        
        Args:
            default_threshold: Default minimum ratio of Vietnamese characters to classify
                             text as Vietnamese. Must be between 0.0 and 1.0.
            cache_size: Maximum number of detection results to cache. Set to 0 to disable
                       caching. Default is 10000 (per Requirement 4.1).
        
        Raises:
            ValueError: If default_threshold is not between 0.0 and 1.0, or if cache_size
                       is negative
        """
        if not 0.0 <= default_threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {default_threshold}")
        if cache_size < 0:
            raise ValueError(f"Cache size must be non-negative, got {cache_size}")
        
        self.default_threshold = default_threshold
        self.cache_size = cache_size
        
        # LRU cache for detection results - maps text hash to (SupportedLanguage, threshold)
        # We include threshold in the cache key to handle different threshold values
        self._cache: OrderedDict[str, tuple[SupportedLanguage, float]] = OrderedDict()
        
        # Cache statistics for monitoring
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _get_text_hash(self, text: str, threshold: float) -> str:
        """
        Generate a hash key for caching detection results.
        
        The hash includes both the text content and threshold to ensure cache correctness
        when different thresholds are used.
        
        Args:
            text: Input text to hash
            threshold: Detection threshold used
            
        Returns:
            SHA-256 hash string for cache key
        """
        # Combine text and threshold for cache key
        cache_input = f"{text}|{threshold}"
        return hashlib.sha256(cache_input.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, cache_key: str, threshold: float) -> Optional[SupportedLanguage]:
        """
        Retrieve detection result from cache if available and valid.
        
        Args:
            cache_key: Hash key for the cached result
            threshold: Detection threshold to validate against cached result
            
        Returns:
            Cached SupportedLanguage if found and threshold matches, None otherwise
        """
        if self.cache_size == 0:
            return None
            
        if cache_key in self._cache:
            cached_result, cached_threshold = self._cache[cache_key]
            
            # Validate that the cached result was computed with the same threshold
            if cached_threshold == threshold:
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                self._cache_hits += 1
                return cached_result
            else:
                # Threshold mismatch, remove invalid cache entry
                del self._cache[cache_key]
        
        self._cache_misses += 1
        return None
    
    def _store_in_cache(self, cache_key: str, result: SupportedLanguage, threshold: float) -> None:
        """
        Store detection result in cache with LRU eviction.
        
        Args:
            cache_key: Hash key for the result
            result: Detection result to cache
            threshold: Detection threshold used for this result
        """
        if self.cache_size == 0:
            return
            
        # Store result with threshold for validation
        self._cache[cache_key] = (result, threshold)
        
        # Move to end (most recently used)
        self._cache.move_to_end(cache_key)
        
        # Evict oldest entries if cache exceeds size limit
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # Remove oldest (first) item
    
    def get_cache_stats(self) -> dict[str, int]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache hits, misses, size, and hit rate
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'size': len(self._cache),
            'max_size': self.cache_size,
            'hit_rate_percent': round(hit_rate, 2)
        }
    
    def clear_cache(self) -> None:
        """Clear all cached detection results and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text to NFC (Canonical Decomposition followed by Canonical Composition).
        
        This ensures consistent language detection results regardless of whether the input
        text uses NFC or NFD Unicode normalization forms. Vietnamese text can be represented
        in different normalization forms, and we need consistent detection results.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Text normalized to NFC form
        """
        try:
            return unicodedata.normalize('NFC', text)
        except (UnicodeError, ValueError) as e:
            logger.warning(f"Failed to normalize Unicode text: {e}")
            return text  # Return original text if normalization fails
        """
        Normalize text to NFC (Canonical Decomposition followed by Canonical Composition).
        
        This ensures consistent language detection results regardless of whether the input
        text uses NFC or NFD Unicode normalization forms. Vietnamese text can be represented
        in different normalization forms, and we need consistent detection results.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Text normalized to NFC form
        """
        try:
            return unicodedata.normalize('NFC', text)
        except (UnicodeError, ValueError) as e:
            logger.warning(f"Failed to normalize Unicode text: {e}")
            return text  # Return original text if normalization fails
    
    def _sample_text_for_detection(self, text: str) -> str:
        """
        Sample representative sections from long documents for language detection.
        
        For documents longer than 10,000 characters, this method samples:
        - First 2,000 characters (beginning)
        - 2,000 characters from the middle
        - Last 2,000 characters (end)
        
        This sampling strategy maintains detection accuracy while significantly
        reducing processing time for very long documents.
        
        Args:
            text: Input text to sample from
            
        Returns:
            Sampled text containing representative sections, or original text if <= 10,000 chars
        """
        # If text is not long enough, return as-is
        if len(text) <= 10000:
            return text
        
        # Sample size for each section
        sample_size = 2000
        
        # Beginning sample (first 2,000 characters)
        beginning = text[:sample_size]
        
        # Middle sample (2,000 characters from the center)
        middle_start = (len(text) - sample_size) // 2
        middle_end = middle_start + sample_size
        middle = text[middle_start:middle_end]
        
        # End sample (last 2,000 characters)
        end = text[-sample_size:]
        
        # Combine samples with space separators to maintain word boundaries
        sampled_text = f"{beginning} {middle} {end}"
        
        logger.debug(f"Sampled {len(sampled_text)} characters from {len(text)} character document")
        
        return sampled_text

    def _is_vietnamese_character(self, char: str) -> bool:
        """
        Check if a character is a Vietnamese character.
        
        Args:
            char: Single character to check
            
        Returns:
            True if the character is in Vietnamese Unicode ranges, False otherwise
        """
        code_point = ord(char)
        for start, end in self.VIETNAMESE_RANGES:
            if start <= code_point <= end:
                return True
        return False
    
    def detect(self, text: str, threshold: Optional[float] = None) -> SupportedLanguage:
        """
        Detect language from text using Vietnamese character ratio analysis.
        
        The algorithm:
        1. Check cache for existing result with same text and threshold
        2. If not cached, validate text length (reject if > 100,000 characters)
        3. For documents > 10,000 characters, use sampling strategy for performance
        4. Count total letter characters in the text (or sample)
        5. Count Vietnamese characters (from defined Unicode ranges)
        6. Calculate ratio = vietnamese_chars / total_chars
        7. If ratio >= threshold, classify as Vietnamese; otherwise English
        8. Store result in cache for future queries
        
        Sampling Strategy (for texts > 10,000 characters):
        - Sample from beginning (first 2,000 chars), middle (2,000 chars), and end (last 2,000 chars)
        - This maintains O(n) complexity but reduces constant factor significantly
        - Representative sampling preserves language detection accuracy
        
        Edge cases:
        - Empty or whitespace-only text returns English
        - Text with no letter characters returns English
        - Invalid Unicode sequences are handled gracefully (returns English)
        - Text exceeding 100,000 characters is rejected to prevent DoS attacks
        
        Args:
            text: Input text to analyze
            threshold: Minimum ratio of Vietnamese characters to classify as Vietnamese.
                      If None, uses default_threshold. Must be between 0.0 and 1.0.
        
        Returns:
            SupportedLanguage.VIETNAMESE if Vietnamese character ratio >= threshold,
            SupportedLanguage.ENGLISH otherwise
        
        Raises:
            ValueError: If threshold is provided and not between 0.0 and 1.0, or if
                       text length exceeds 100,000 characters
        """
        # Use default threshold if not provided
        if threshold is None:
            threshold = self.default_threshold
        
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        # Check cache first (before expensive validation and processing)
        cache_key = self._get_text_hash(text, threshold)
        cached_result = self._get_from_cache(cache_key, threshold)
        if cached_result is not None:
            return cached_result
        
        # Validate text length to prevent DoS attacks (Requirement 8.4)
        if len(text) > 100000:
            logger.warning(f"Text length {len(text)} exceeds maximum allowed length of 100,000 characters")
            raise ValueError(f"Text length {len(text)} exceeds maximum allowed length of 100,000 characters")
        
        # Handle empty or whitespace-only text
        if not text or not text.strip():
            result = SupportedLanguage.ENGLISH
            self._store_in_cache(cache_key, result, threshold)
            return result
        
        # Normalize text to ensure consistent results across NFC/NFD forms (Requirement 11.1)
        normalized_text = self._normalize_text(text)
        
        # Apply sampling strategy for long documents (Requirement 9.4)
        text_to_analyze = self._sample_text_for_detection(normalized_text)

        try:
            vietnamese_chars = 0
            total_chars = 0
            
            # Count Vietnamese characters and total letter characters
            for char in text_to_analyze:
                if char.isalpha():
                    total_chars += 1
                    if self._is_vietnamese_character(char):
                        vietnamese_chars += 1
            
            # If no letter characters found, default to English
            if total_chars == 0:
                result = SupportedLanguage.ENGLISH
            else:
                # Calculate ratio and determine language
                ratio = vietnamese_chars / total_chars
                
                if ratio >= threshold:
                    result = SupportedLanguage.VIETNAMESE
                else:
                    result = SupportedLanguage.ENGLISH
            
            # Store result in cache
            self._store_in_cache(cache_key, result, threshold)
            return result
                
        except (UnicodeDecodeError, UnicodeError) as e:
            # Handle invalid Unicode sequences gracefully (Requirement 8.2)
            logger.warning(f"Invalid Unicode sequence encountered during language detection: {e}")
            result = SupportedLanguage.ENGLISH
            self._store_in_cache(cache_key, result, threshold)
            return result
    
    def is_vietnamese(self, text: str, threshold: Optional[float] = None) -> bool:
        """
        Quick check if text is Vietnamese.
        
        This is a convenience method that returns a boolean instead of the enum.
        
        Args:
            text: Input text to analyze
            threshold: Minimum ratio of Vietnamese characters to classify as Vietnamese.
                      If None, uses default_threshold.
        
        Returns:
            True if text is classified as Vietnamese, False otherwise
        """
        return self.detect(text, threshold) == SupportedLanguage.VIETNAMESE

    def detect_batch(self, texts: list[str], threshold: Optional[float] = None) -> list[SupportedLanguage]:
        """
        Detect language for multiple texts efficiently using batch processing.
        
        This method processes multiple texts in a single call, which can improve
        throughput by reducing function call overhead and enabling better cache
        utilization patterns.
        
        Args:
            texts: List of input texts to analyze
            threshold: Minimum ratio of Vietnamese characters to classify as Vietnamese.
                      If None, uses default_threshold.
        
        Returns:
            List of SupportedLanguage results in the same order as input texts
        
        Raises:
            ValueError: If threshold is provided and not between 0.0 and 1.0
        """
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        results = []
        for text in texts:
            results.append(self.detect(text, threshold))
        
        return results

    def get_performance_metrics(self) -> dict[str, any]:
        """
        Get comprehensive performance metrics for monitoring.
        
        Returns performance statistics including cache metrics and processing
        information for monitoring and optimization purposes.
        
        Returns:
            Dictionary containing performance metrics:
            - cache_stats: Cache hit/miss statistics
            - total_detections: Total number of detections performed
            - cache_efficiency: Cache hit rate as percentage
        """
        cache_stats = self.get_cache_stats()
        total_detections = cache_stats['hits'] + cache_stats['misses']
        
        return {
            'cache_stats': cache_stats,
            'total_detections': total_detections,
            'cache_efficiency': cache_stats['hit_rate_percent'],
            'cache_utilization': (cache_stats['size'] / cache_stats['max_size'] * 100) if cache_stats['max_size'] > 0 else 0.0
        }


# Global singleton instance for easy access
_global_detector = LanguageDetector()


def detect_language(text: str, threshold: Optional[float] = None) -> str:
    """
    Convenience function for language detection.
    
    Args:
        text: Input text to analyze
        threshold: Detection threshold
        
    Returns:
        String name of the detected language ("Vietnamese" or "English")
    """
    return _global_detector.detect(text, threshold).value
