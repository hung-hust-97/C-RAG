"""
Language detection module for LightRAG.

This module provides comprehensive language detection capabilities including:
- FastText-based machine learning detection
- Unicode-based character analysis detection  
- Hybrid detection strategies combining multiple methods
- Caching and performance optimization
- Configuration management
- Metrics collection and monitoring
- Health checks and observability

The module maintains backward compatibility with existing LightRAG components
while adding enhanced accuracy through FastText integration.
"""

from .fasttext_detector import FastTextDetector
from .model_manager import FastTextModelManager
from .hybrid_strategy import HybridDetectionStrategy
from .config import FastTextConfig
from .service import LanguageDetectionService, DetectionResult
from .error_handler import ResilientErrorHandler, ErrorHandlingConfig
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .metrics import MetricsCollector, HealthChecker, HealthStatus, PerformanceMetrics

# Re-export existing types for backward compatibility
from lightrag.language_detector import SupportedLanguage, LanguageDetector

__all__ = [
    "FastTextDetector",
    "FastTextModelManager", 
    "HybridDetectionStrategy",
    "FastTextConfig",
    "LanguageDetectionService",
    "DetectionResult",
    "ResilientErrorHandler",
    "ErrorHandlingConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "MetricsCollector",
    "HealthChecker",
    "HealthStatus",
    "PerformanceMetrics",
    "SupportedLanguage",
    "LanguageDetector",
]