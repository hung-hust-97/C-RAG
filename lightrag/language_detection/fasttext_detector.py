"""
FastText-based language detection for Vietnamese and English.

This module implements the core FastText detection functionality using Facebook's
FastText library for accurate machine learning-based language detection. It provides
text preprocessing, confidence scoring, and handles edge cases like short text.
"""

import hashlib
import re
import time
import unicodedata
from collections import OrderedDict
from typing import Tuple, Optional, Dict, Any, List

from lightrag.language_detector import SupportedLanguage
from .config import FastTextConfig
from .model_manager import FastTextModelManager
from .error_handler import ResilientErrorHandler, ErrorHandlingConfig
from .logger import get_fasttext_logger
from .security import InputSanitizer, PrivacyFilter

# Use dedicated FastText logger
logger = get_fasttext_logger("detector")

# Import FastText with graceful fallback
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
    logger.info("FastText library successfully imported")
except ImportError as e:
    FASTTEXT_AVAILABLE = False
    logger.warning(f"FastText library not available: {e}")
    fasttext = None


class FastTextDetector:
    """FastText-based language detection for Vietnamese and English."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_thresholds: Optional[Dict[str, float]] = None,
                 cache_size: int = 10000,
                 config: Optional[FastTextConfig] = None,
                 error_handler: Optional[ResilientErrorHandler] = None):
        """
        Initialize FastText detector with configurable parameters.
        
        Args:
            model_path: Custom path to FastText model file
            confidence_thresholds: Custom confidence thresholds for languages
            cache_size: Size of LRU cache for detection results
            config: FastText configuration object
            error_handler: Error handler for resilience patterns
        """
        self.config = config or FastTextConfig.from_env()
        
        # Override config with provided parameters
        if model_path is not None:
            self.config.model_path = model_path
        if cache_size != 10000:  # Only override if different from default
            self.config.cache_size = cache_size
        
        # Set up confidence thresholds
        self.confidence_thresholds = confidence_thresholds or {
            'vietnamese': self.config.vietnamese_threshold,
            'english': self.config.english_threshold
        }
        
        # Initialize error handler with resilience patterns
        self.error_handler = error_handler or ResilientErrorHandler(ErrorHandlingConfig())
        
        # Initialize security components (Requirement 10.1, 10.2)
        self.input_sanitizer = InputSanitizer(max_length=self.config.max_text_length)
        self.privacy_filter = PrivacyFilter()
        
        # Initialize model manager and model
        self.model_manager = FastTextModelManager(self.config)
        self._model = None
        self._model_loaded = False
        
        # Initialize LRU cache for detection results
        self._cache: OrderedDict[str, Tuple[SupportedLanguage, float, float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"FastTextDetector initialized with cache_size={self.config.cache_size}, "
                   f"error_handler enabled, circuit_breaker configured, security protections enabled")
        
        # Try to load model if FastText is available
        if FASTTEXT_AVAILABLE and self.config.enabled:
            self._load_model()
        else:
            if not FASTTEXT_AVAILABLE:
                logger.warning("FastText library not available - detector will not be functional")
            if not self.config.enabled:
                logger.info("FastText detection disabled by configuration")
    
    def _load_model(self) -> bool:
        """
        Load the FastText model with resilient error handling.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self._model_loaded:
            return True
        
        def _load_model_operation():
            """Internal model loading operation."""
            # Ensure model is available
            model_path = self.model_manager.ensure_model_available()
            
            # Load the model
            logger.info(f"Loading FastText model from: {model_path}")
            start_time = time.time()
            
            self._model = fasttext.load_model(model_path)
            
            load_time = time.time() - start_time
            logger.info(f"FastText model loaded successfully in {load_time:.2f}s")
            
            self._model_loaded = True
            return True
        
        def _fallback_operation():
            """Fallback when model loading fails."""
            logger.warning("Model loading failed, FastText will be unavailable")
            self._model = None
            self._model_loaded = False
            return False
        
        try:
            return self.error_handler.execute_with_resilience(
                operation=_load_model_operation,
                fallback_operation=_fallback_operation,
                operation_name="model_loading"
            )
        except Exception as e:
            logger.error(f"Failed to load FastText model after all resilience attempts: {e}")
            self._model = None
            self._model_loaded = False
            return False
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess and normalize text for detection.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed and normalized text
        """
        if not text:
            return ""
        
        # Normalize Unicode to NFC form for consistency
        try:
            normalized = unicodedata.normalize('NFC', text)
        except (UnicodeError, ValueError):
            logger.warning("Failed to normalize Unicode text")
            normalized = text
        
        # Remove excessive whitespace and normalize
        # Replace multiple whitespace characters with single space
        cleaned = re.sub(r'\s+', ' ', normalized.strip())
        
        # Remove control characters but keep basic punctuation
        cleaned = ''.join(char for char in cleaned if not unicodedata.category(char).startswith('C'))
        
        return cleaned
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Input text
            
        Returns:
            SHA-256 hash of the text for caching
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Tuple[SupportedLanguage, float]]:
        """
        Retrieve detection result from cache.
        
        Args:
            cache_key: Cache key for the text
            
        Returns:
            Cached result tuple (language, confidence) or None if not found
        """
        if cache_key in self._cache:
            language, confidence, timestamp = self._cache[cache_key]
            
            # Check if cache entry is still valid (TTL)
            if time.time() - timestamp <= self.config.cache_ttl:
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                self._cache_hits += 1
                return language, confidence
            else:
                # Remove expired entry
                del self._cache[cache_key]
        
        self._cache_misses += 1
        return None
    
    def _store_in_cache(self, cache_key: str, language: SupportedLanguage, confidence: float) -> None:
        """
        Store detection result in cache.
        
        Args:
            cache_key: Cache key for the text
            language: Detected language
            confidence: Detection confidence
        """
        if self.config.cache_size <= 0:
            return
        
        # Store with timestamp for TTL
        self._cache[cache_key] = (language, confidence, time.time())
        
        # Move to end (most recently used)
        self._cache.move_to_end(cache_key)
        
        # Evict oldest entries if cache exceeds size limit
        while len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)  # Remove oldest (first) item
    
    def _fasttext_predict(self, text: str) -> Tuple[str, float]:
        """
        Perform FastText prediction on preprocessed text with resilient error handling.
        
        Args:
            text: Preprocessed text for prediction
            
        Returns:
            Tuple of (language_code, confidence)
        """
        def _predict_operation():
            """Internal prediction operation."""
            if not self._model:
                raise RuntimeError("FastText model not loaded")
            
            # FastText expects text without newlines for language detection
            text_for_prediction = text.replace('\n', ' ').strip()
            
            if not text_for_prediction:
                return 'en', 0.0
            
            # Predict with k=2 to get top 2 predictions
            predictions = self._model.predict(text_for_prediction, k=2)
            
            if not predictions[0]:  # No predictions returned
                return 'en', 0.0
            
            # Extract language code and confidence
            top_prediction = predictions[0][0]  # First prediction
            confidence = float(predictions[1][0])  # First confidence score
            
            # FastText returns labels like '__label__en' or '__label__vi'
            language_code = top_prediction.replace('__label__', '')
            
            return language_code, confidence
        
        def _predict_fallback():
            """Fallback when FastText prediction fails."""
            logger.debug("FastText prediction failed, using English default")
            return 'en', 0.4
        
        try:
            return self.error_handler.execute_with_resilience(
                operation=_predict_operation,
                fallback_operation=_predict_fallback,
                operation_name="fasttext_prediction"
            )
        except Exception as e:
            logger.warning(f"FastText prediction failed with fallback: {e}")
            return 'en', 0.4
    
    def _map_language_code(self, language_code: str) -> SupportedLanguage:
        """
        Map FastText language code to SupportedLanguage enum.
        
        Args:
            language_code: Language code from FastText (e.g., 'vi', 'en')
            
        Returns:
            SupportedLanguage enum value
        """
        # Map common Vietnamese language codes
        if language_code in ('vi', 'vie', 'vietnamese'):
            return SupportedLanguage.VIETNAMESE
        
        # Default to English for all other cases
        return SupportedLanguage.ENGLISH
    
    def detect_batch(self, texts: List[str]) -> List[Tuple[SupportedLanguage, float]]:
        """
        Detect language for multiple texts in batch for improved throughput.
        
        This method processes multiple texts efficiently by:
        1. Sanitizing all inputs for security (Requirement 10.2)
        2. Checking cache for all texts first
        3. Batching uncached texts for FastText prediction
        4. Processing results and updating cache
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of tuples (detected_language, confidence_score) in same order as input
        """
        if not texts:
            return []
        
        results = []
        uncached_indices = []
        uncached_texts = []
        
        # First pass: sanitize and check cache for all texts
        for i, text in enumerate(texts):
            # Sanitize input (Requirement 10.2)
            try:
                sanitized_text = self.input_sanitizer.sanitize(text, strict=False)
            except ValueError as e:
                # Input validation failed, return low confidence English
                logger.warning(f"Batch text {i} sanitization failed: {e}")
                results.append((SupportedLanguage.ENGLISH, 0.2))
                continue
            
            # Handle empty or whitespace-only text
            if not sanitized_text or not sanitized_text.strip():
                results.append((SupportedLanguage.ENGLISH, 1.0))
                continue
            
            # Check text length limits
            if len(sanitized_text) > self.config.max_text_length:
                logger.warning(f"Batch text {i} length {len(sanitized_text)} exceeds maximum {self.config.max_text_length}")
                results.append((SupportedLanguage.ENGLISH, 0.4))
                continue
            
            # Preprocess text
            processed_text = self._preprocess_text(sanitized_text)
            
            # Check cache
            cache_key = self._get_cache_key(processed_text)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result is not None:
                results.append(cached_result)
            else:
                # Mark for batch processing
                results.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_texts.append(processed_text)
        
        # Second pass: batch process uncached texts
        if uncached_texts:
            batch_results = self._detect_batch_uncached(uncached_texts)
            
            # Fill in the results and update cache
            for idx, (original_idx, processed_text) in enumerate(zip(uncached_indices, uncached_texts)):
                detection_result = batch_results[idx]
                results[original_idx] = detection_result
                
                # Update cache
                cache_key = self._get_cache_key(processed_text)
                self._store_in_cache(cache_key, detection_result[0], detection_result[1])
        
        return results
    
    def _detect_batch_uncached(self, processed_texts: List[str]) -> List[Tuple[SupportedLanguage, float]]:
        """
        Process a batch of uncached, preprocessed texts.
        
        Args:
            processed_texts: List of preprocessed texts
            
        Returns:
            List of detection results
        """
        results = []
        
        # Check if FastText is available
        if not self.is_available():
            logger.debug("FastText not available for batch detection, using defaults")
            return [(SupportedLanguage.ENGLISH, 0.5) for _ in processed_texts]
        
        # Process texts in batches according to config.batch_size
        batch_size = self.config.batch_size
        
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i + batch_size]
            batch_results = self._process_fasttext_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_fasttext_batch(self, batch_texts: List[str]) -> List[Tuple[SupportedLanguage, float]]:
        """
        Process a single batch of texts with FastText.
        
        Args:
            batch_texts: List of preprocessed texts (up to batch_size)
            
        Returns:
            List of detection results for the batch
        """
        results = []
        
        try:
            start_time = time.time()
            
            for text in batch_texts:
                # Handle short text
                if len(text) < self.config.min_text_length_for_fasttext:
                    logger.debug(f"Text too short ({len(text)} chars) for reliable FastText detection")
                    results.append((SupportedLanguage.ENGLISH, 0.3))
                    continue
                
                # Perform FastText prediction
                language_code, raw_confidence = self._fasttext_predict(text)
                
                # Map language code to SupportedLanguage
                detected_language = self._map_language_code(language_code)
                
                # Apply confidence thresholds and adjustments
                if detected_language == SupportedLanguage.VIETNAMESE:
                    # For Vietnamese, apply threshold
                    if raw_confidence >= self.confidence_thresholds['vietnamese']:
                        final_confidence = raw_confidence
                    else:
                        # Below threshold, classify as English with adjusted confidence
                        detected_language = SupportedLanguage.ENGLISH
                        final_confidence = 1.0 - raw_confidence
                else:
                    # For English or other languages, apply English threshold
                    if raw_confidence >= self.confidence_thresholds['english']:
                        final_confidence = raw_confidence
                    else:
                        # Low confidence, reduce it further
                        final_confidence = raw_confidence * 0.8
                
                # Ensure confidence is in valid range [0.0, 1.0]
                final_confidence = max(0.0, min(1.0, final_confidence))
                
                results.append((detected_language, final_confidence))
            
            batch_time = time.time() - start_time
            
            if self.config.log_performance_metrics:
                logger.debug(f"Batch FastText detection: {len(batch_texts)} texts in {batch_time*1000:.1f}ms "
                           f"({batch_time*1000/len(batch_texts):.1f}ms per text)")
            
        except Exception as e:
            logger.error(f"Batch FastText detection failed: {e}")
            # Fallback to English with low confidence for all texts in batch
            results = [(SupportedLanguage.ENGLISH, 0.4) for _ in batch_texts]
        
        return results
    
    def detect(self, text: str) -> Tuple[SupportedLanguage, float]:
        """
        Detect language and return confidence score with resilient error handling.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (detected_language, confidence_score)
        """
        # Sanitize input to prevent injection attacks (Requirement 10.2)
        try:
            sanitized_text = self.input_sanitizer.sanitize(text, strict=False)
        except ValueError as e:
            # Input validation failed (e.g., too long)
            logger.error(f"Input sanitization failed: {e}")
            raise
        
        # Log detection request without actual text content (Requirement 10.1)
        log_info = self.privacy_filter.get_safe_log_info(sanitized_text)
        logger.debug(f"Detection request: length={log_info['length']}, "
                    f"hash={log_info['text_hash']}, empty={log_info['is_empty']}")
        
        # Handle empty or whitespace-only text
        if not sanitized_text or not sanitized_text.strip():
            logger.debug("Empty or whitespace-only text, returning English default")
            return SupportedLanguage.ENGLISH, 1.0
        
        # Check text length limits (already validated by sanitizer, but double-check)
        if len(sanitized_text) > self.config.max_text_length:
            logger.warning(f"Text length {len(sanitized_text)} exceeds maximum {self.config.max_text_length}")
            raise ValueError(f"Text length exceeds maximum allowed length of {self.config.max_text_length}")
        
        # Preprocess text
        processed_text = self._preprocess_text(sanitized_text)
        
        # Check cache first
        cache_key = self._get_cache_key(processed_text)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug("Returning cached detection result")
            return cached_result
        
        # Handle short text - return low confidence or defer to Unicode detection
        if len(processed_text) < self.config.min_text_length_for_fasttext:
            logger.debug(f"Text too short ({len(processed_text)} chars) for reliable FastText detection")
            # Return English with low confidence for short text
            result = (SupportedLanguage.ENGLISH, 0.3)
            self._store_in_cache(cache_key, result[0], result[1])
            return result
        
        # Check if FastText is available and model is loaded
        if not self.is_available():
            logger.debug("FastText not available, returning English default")
            result = (SupportedLanguage.ENGLISH, 0.5)
            self._store_in_cache(cache_key, result[0], result[1])
            return result
        
        def _detection_operation():
            """Internal detection operation with FastText."""
            # Perform FastText prediction
            start_time = time.time()
            language_code, raw_confidence = self._fasttext_predict(processed_text)
            detection_time = time.time() - start_time
            
            # Map language code to SupportedLanguage
            detected_language = self._map_language_code(language_code)
            
            # Apply confidence thresholds and adjustments
            if detected_language == SupportedLanguage.VIETNAMESE:
                # For Vietnamese, apply threshold
                if raw_confidence >= self.confidence_thresholds['vietnamese']:
                    final_confidence = raw_confidence
                else:
                    # Below threshold, classify as English with adjusted confidence
                    detected_language = SupportedLanguage.ENGLISH
                    final_confidence = 1.0 - raw_confidence
            else:
                # For English or other languages, apply English threshold
                if raw_confidence >= self.confidence_thresholds['english']:
                    final_confidence = raw_confidence
                else:
                    # Low confidence, reduce it further
                    final_confidence = raw_confidence * 0.8
            
            # Ensure confidence is in valid range [0.0, 1.0]
            final_confidence = max(0.0, min(1.0, final_confidence))
            
            if self.config.log_performance_metrics:
                # Log without text content (Requirement 10.1)
                logger.debug(f"FastText detection: {detected_language.value} (confidence: {final_confidence:.3f}, "
                           f"raw: {raw_confidence:.3f}, time: {detection_time*1000:.1f}ms, "
                           f"text_hash: {log_info['text_hash']})")
            
            return detected_language, final_confidence
        
        def _detection_fallback():
            """Fallback detection using English default."""
            logger.debug("FastText detection failed, using English fallback")
            return SupportedLanguage.ENGLISH, 0.4
        
        try:
            # Execute detection with resilience patterns
            result = self.error_handler.execute_with_resilience(
                operation=_detection_operation,
                fallback_operation=_detection_fallback,
                operation_name="language_detection"
            )
            
            # Store in cache
            self._store_in_cache(cache_key, result[0], result[1])
            
            return result
            
        except Exception as e:
            logger.error(f"Language detection failed completely: {e}")
            # Ultimate fallback to English with low confidence
            result = (SupportedLanguage.ENGLISH, 0.2)
            self._store_in_cache(cache_key, result[0], result[1])
            return result
    
    def is_available(self) -> bool:
        """
        Check if FastText model is loaded and ready.
        
        Returns:
            bool: True if FastText is available and model is loaded
        """
        return (FASTTEXT_AVAILABLE and 
                self.config.enabled and 
                self._model_loaded and 
                self._model is not None)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata and status information.
        
        Returns:
            Dict containing model information and status
        """
        info = {
            "fasttext_available": FASTTEXT_AVAILABLE,
            "config_enabled": self.config.enabled,
            "model_loaded": self._model_loaded,
            "is_available": self.is_available(),
            "confidence_thresholds": self.confidence_thresholds.copy(),
            "cache_size": self.config.cache_size,
            "cache_stats": self.get_cache_stats(),
        }
        
        # Add model manager info
        if hasattr(self, 'model_manager'):
            info.update(self.model_manager.get_model_info())
        
        return info
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dict with cache performance metrics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0.0
        
        # Calculate cache efficiency metrics
        cache_efficiency = {
            'hit_rate_percent': round(hit_rate, 2),
            'miss_rate_percent': round(100 - hit_rate, 2) if total_requests > 0 else 0.0,
            'total_requests': total_requests,
        }
        
        # Calculate cache utilization
        utilization_percent = (len(self._cache) / self.config.cache_size * 100) if self.config.cache_size > 0 else 0.0
        
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'size': len(self._cache),
            'max_size': self.config.cache_size,
            'utilization_percent': round(utilization_percent, 2),
            'efficiency': cache_efficiency,
            'ttl_seconds': self.config.cache_ttl,
        }
    
    def clear_cache(self) -> None:
        """Clear detection cache and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("FastText detection cache cleared")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring.
        
        Returns:
            Dict with performance metrics including cache stats, model info, and configuration
        """
        metrics = {
            'cache_stats': self.get_cache_stats(),
            'model_info': {
                'is_available': self.is_available(),
                'model_loaded': self._model_loaded,
                'fasttext_library_available': FASTTEXT_AVAILABLE,
                'config_enabled': self.config.enabled,
            },
            'configuration': {
                'cache_size': self.config.cache_size,
                'cache_ttl': self.config.cache_ttl,
                'batch_size': self.config.batch_size,
                'confidence_thresholds': self.confidence_thresholds.copy(),
                'min_text_length_for_fasttext': self.config.min_text_length_for_fasttext,
                'max_text_length': self.config.max_text_length,
            },
            'thresholds': {
                'vietnamese_threshold': self.confidence_thresholds['vietnamese'],
                'english_threshold': self.confidence_thresholds['english'],
                'high_confidence': self.config.high_confidence_threshold,
                'low_confidence': self.config.low_confidence_threshold,
            }
        }
        
        # Add model manager metrics if available
        if hasattr(self, 'model_manager'):
            try:
                metrics['model_manager'] = self.model_manager.get_model_info()
            except Exception as e:
                logger.warning(f"Failed to get model manager info: {e}")
                metrics['model_manager'] = {'error': str(e)}
        
        # Add error handler metrics
        if hasattr(self, 'error_handler'):
            try:
                metrics['error_handler'] = self.error_handler.get_error_stats()
                metrics['circuit_breaker'] = {
                    'is_open': self.error_handler.is_circuit_open(),
                    'health_status': self.error_handler.get_health_status()
                }
            except Exception as e:
                logger.warning(f"Failed to get error handler stats: {e}")
                metrics['error_handler'] = {'error': str(e)}
        
        return metrics
    
    def reset_cache_stats(self) -> None:
        """Reset cache statistics without clearing the cache contents."""
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("FastText cache statistics reset")