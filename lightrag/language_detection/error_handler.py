"""
Comprehensive error handling and resilience patterns for FastText language detection.

This module provides centralized error handling, retry logic with exponential backoff,
graceful fallback mechanisms, and integration with the circuit breaker pattern.
"""

import time
import random
from typing import Any, Callable, Optional, Dict, List, Type, Union
from dataclasses import dataclass
from enum import Enum
from .circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, CircuitBreakerConfig
from .logger import get_fasttext_logger

# Use dedicated FastText logger
logger = get_fasttext_logger("error_handler")


class ErrorSeverity(Enum):
    """Error severity levels for different types of failures."""
    LOW = "low"           # Temporary issues, retry recommended
    MEDIUM = "medium"     # Significant issues, limited retries
    HIGH = "high"         # Critical issues, immediate fallback
    CRITICAL = "critical" # System-level issues, disable component


class ErrorCategory(Enum):
    """Categories of errors for different handling strategies."""
    NETWORK = "network"           # Network connectivity issues
    MODEL_LOADING = "model_loading"  # Model file loading problems
    MODEL_CORRUPTION = "model_corruption"  # Model file integrity issues
    FASTTEXT_RUNTIME = "fasttext_runtime"  # FastText library runtime errors
    CONFIGURATION = "configuration"  # Configuration validation errors
    RESOURCE = "resource"         # Resource exhaustion (memory, disk)
    UNKNOWN = "unknown"           # Unclassified errors


@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""
    
    # Maximum number of retry attempts
    max_attempts: int = 3
    
    # Base delay between retries (seconds)
    base_delay: float = 1.0
    
    # Maximum delay between retries (seconds)
    max_delay: float = 60.0
    
    # Exponential backoff multiplier
    backoff_multiplier: float = 2.0
    
    # Random jitter factor (0.0 to 1.0)
    jitter_factor: float = 0.1
    
    # Timeout for individual attempts (seconds)
    attempt_timeout: float = 30.0


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling behavior."""
    
    # Retry configuration
    retry_config: RetryConfig
    
    # Circuit breaker configuration
    circuit_breaker_config: CircuitBreakerConfig
    
    # Enable automatic fallback to Unicode detection
    enable_unicode_fallback: bool = True
    
    # Enable detailed error logging
    enable_detailed_logging: bool = True
    
    # Enable error metrics collection
    enable_metrics_collection: bool = True
    
    def __init__(self):
        self.retry_config = RetryConfig()
        self.circuit_breaker_config = CircuitBreakerConfig()


class ErrorClassifier:
    """Classifies errors into categories and severity levels for appropriate handling."""
    
    # Error classification rules
    ERROR_PATTERNS = {
        # Network-related errors
        ErrorCategory.NETWORK: [
            "URLError", "HTTPError", "ConnectionError", "TimeoutError",
            "socket.timeout", "requests.exceptions", "urllib.error"
        ],
        
        # Model loading errors
        ErrorCategory.MODEL_LOADING: [
            "FileNotFoundError", "PermissionError", "IsADirectoryError",
            "OSError", "IOError", "model not found", "cannot load model"
        ],
        
        # Model corruption errors
        ErrorCategory.MODEL_CORRUPTION: [
            "checksum", "validation failed", "corrupted", "invalid model",
            "model format", "unexpected EOF", "truncated"
        ],
        
        # FastText runtime errors
        ErrorCategory.FASTTEXT_RUNTIME: [
            "fasttext", "prediction failed", "model prediction",
            "RuntimeError", "ValueError", "AttributeError"
        ],
        
        # Configuration errors
        ErrorCategory.CONFIGURATION: [
            "configuration", "invalid parameter", "validation failed",
            "threshold", "config", "environment variable"
        ],
        
        # Resource exhaustion errors
        ErrorCategory.RESOURCE: [
            "MemoryError", "disk space", "no space left", "out of memory",
            "resource temporarily unavailable", "too many open files"
        ]
    }
    
    # Severity mapping based on error category
    SEVERITY_MAPPING = {
        ErrorCategory.NETWORK: ErrorSeverity.LOW,
        ErrorCategory.MODEL_LOADING: ErrorSeverity.MEDIUM,
        ErrorCategory.MODEL_CORRUPTION: ErrorSeverity.HIGH,
        ErrorCategory.FASTTEXT_RUNTIME: ErrorSeverity.MEDIUM,
        ErrorCategory.CONFIGURATION: ErrorSeverity.HIGH,
        ErrorCategory.RESOURCE: ErrorSeverity.CRITICAL,
        ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM,
    }
    
    @classmethod
    def classify_error(cls, error: Exception) -> tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an error into category and severity.
        
        Args:
            error: Exception to classify
            
        Returns:
            Tuple of (ErrorCategory, ErrorSeverity)
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Check error patterns
        for category, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in error_str or pattern.lower() in error_type.lower():
                    severity = cls.SEVERITY_MAPPING[category]
                    return category, severity
        
        # Default to unknown category with medium severity
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM


class ResilientErrorHandler:
    """
    Comprehensive error handler with retry logic, circuit breaker, and fallback mechanisms.
    
    This class provides centralized error handling for all FastText operations with:
    - Exponential backoff retry logic
    - Circuit breaker pattern for repeated failures
    - Graceful fallback to Unicode detection
    - Comprehensive error classification and logging
    - Performance metrics collection
    """
    
    def __init__(self, config: Optional[ErrorHandlingConfig] = None):
        """
        Initialize error handler with configuration.
        
        Args:
            config: Error handling configuration. Uses defaults if None.
        """
        self.config = config or ErrorHandlingConfig()
        self.circuit_breaker = CircuitBreaker(self.config.circuit_breaker_config)
        
        # Error metrics
        self._error_counts = {}
        self._error_history = []
        self._fallback_count = 0
        self._retry_count = 0
        self._total_operations = 0
        
        logger.info("ResilientErrorHandler initialized with circuit breaker and retry logic")
    
    def execute_with_resilience(self, 
                              operation: Callable,
                              fallback_operation: Optional[Callable] = None,
                              operation_name: str = "unknown",
                              *args, **kwargs) -> Any:
        """
        Execute an operation with full resilience patterns.
        
        Args:
            operation: Primary operation to execute
            fallback_operation: Fallback operation if primary fails
            operation_name: Name of operation for logging
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result from primary operation or fallback
            
        Raises:
            Exception: If both primary and fallback operations fail
        """
        self._total_operations += 1
        
        try:
            # Try primary operation through circuit breaker
            return self.circuit_breaker.call(
                self._execute_with_retry,
                operation, operation_name, *args, **kwargs
            )
            
        except CircuitBreakerOpenError:
            logger.warning(f"Circuit breaker open for {operation_name}, using fallback")
            return self._execute_fallback(fallback_operation, operation_name, *args, **kwargs)
            
        except Exception as e:
            logger.error(f"Primary operation {operation_name} failed: {e}")
            return self._execute_fallback(fallback_operation, operation_name, *args, **kwargs)
    
    def _execute_with_retry(self, operation: Callable, operation_name: str, *args, **kwargs) -> Any:
        """
        Execute operation with retry logic and exponential backoff.
        
        Args:
            operation: Operation to execute
            operation_name: Name for logging
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(1, self.config.retry_config.max_attempts + 1):
            try:
                logger.debug(f"Attempting {operation_name} (attempt {attempt}/{self.config.retry_config.max_attempts})")
                
                # Execute the operation
                result = operation(*args, **kwargs)
                
                if attempt > 1:
                    logger.info(f"Operation {operation_name} succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                category, severity = ErrorClassifier.classify_error(e)
                
                # Record error
                self._record_error(e, category, severity, operation_name)
                
                # Check if we should retry based on error severity
                if not self._should_retry(category, severity, attempt):
                    logger.warning(f"Not retrying {operation_name} due to {severity.value} severity error")
                    raise e
                
                if attempt < self.config.retry_config.max_attempts:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"Operation {operation_name} failed (attempt {attempt}): {e}. "
                                 f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    self._retry_count += 1
                else:
                    logger.error(f"Operation {operation_name} failed after {attempt} attempts: {e}")
        
        # All attempts failed
        raise last_exception
    
    def _execute_fallback(self, fallback_operation: Optional[Callable], 
                         operation_name: str, *args, **kwargs) -> Any:
        """
        Execute fallback operation if available.
        
        Args:
            fallback_operation: Fallback operation to execute
            operation_name: Name for logging
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Fallback operation result
            
        Raises:
            Exception: If fallback is not available or fails
        """
        if not self.config.enable_unicode_fallback or not fallback_operation:
            raise RuntimeError(f"No fallback available for {operation_name}")
        
        try:
            logger.info(f"Executing fallback for {operation_name}")
            self._fallback_count += 1
            
            result = fallback_operation(*args, **kwargs)
            logger.debug(f"Fallback for {operation_name} succeeded")
            
            return result
            
        except Exception as e:
            logger.error(f"Fallback for {operation_name} also failed: {e}")
            raise
    
    def _should_retry(self, category: ErrorCategory, severity: ErrorSeverity, attempt: int) -> bool:
        """
        Determine if an error should be retried based on category and severity.
        
        Args:
            category: Error category
            severity: Error severity
            attempt: Current attempt number
            
        Returns:
            True if retry should be attempted
        """
        # Never retry critical errors
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        # Don't retry high severity errors after first attempt
        if severity == ErrorSeverity.HIGH and attempt > 1:
            return False
        
        # Don't retry configuration errors
        if category == ErrorCategory.CONFIGURATION:
            return False
        
        # Retry network and temporary errors
        if category in (ErrorCategory.NETWORK, ErrorCategory.FASTTEXT_RUNTIME):
            return True
        
        # Retry model loading errors with limited attempts
        if category == ErrorCategory.MODEL_LOADING and attempt <= 2:
            return True
        
        # Default: retry medium and low severity errors
        return severity in (ErrorSeverity.LOW, ErrorSeverity.MEDIUM)
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff and jitter.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (multiplier ^ (attempt - 1))
        delay = self.config.retry_config.base_delay * (
            self.config.retry_config.backoff_multiplier ** (attempt - 1)
        )
        
        # Apply maximum delay limit
        delay = min(delay, self.config.retry_config.max_delay)
        
        # Add random jitter to prevent thundering herd
        if self.config.retry_config.jitter_factor > 0:
            jitter = delay * self.config.retry_config.jitter_factor * random.random()
            delay += jitter
        
        return delay
    
    def _record_error(self, error: Exception, category: ErrorCategory, 
                     severity: ErrorSeverity, operation_name: str) -> None:
        """
        Record error for metrics and analysis.
        
        Args:
            error: The exception that occurred
            category: Error category
            severity: Error severity
            operation_name: Name of the operation that failed
        """
        if not self.config.enable_metrics_collection:
            return
        
        error_key = f"{category.value}_{severity.value}"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        # Keep recent error history (last 100 errors)
        error_record = {
            "timestamp": time.time(),
            "operation": operation_name,
            "category": category.value,
            "severity": severity.value,
            "error_type": type(error).__name__,
            "error_message": str(error)[:200]  # Truncate long messages
        }
        
        self._error_history.append(error_record)
        if len(self._error_history) > 100:
            self._error_history.pop(0)
        
        if self.config.enable_detailed_logging:
            logger.warning(f"Error recorded: {operation_name} - {category.value}/{severity.value} - {error}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive error handling statistics.
        
        Returns:
            Dictionary with error metrics and circuit breaker status
        """
        # Calculate error rates
        error_rate = (sum(self._error_counts.values()) / self._total_operations * 100) if self._total_operations > 0 else 0.0
        fallback_rate = (self._fallback_count / self._total_operations * 100) if self._total_operations > 0 else 0.0
        
        # Recent error analysis (last 10 errors)
        recent_errors = self._error_history[-10:] if self._error_history else []
        
        return {
            "total_operations": self._total_operations,
            "error_counts": self._error_counts.copy(),
            "fallback_count": self._fallback_count,
            "retry_count": self._retry_count,
            "rates": {
                "error_rate_percent": round(error_rate, 2),
                "fallback_rate_percent": round(fallback_rate, 2),
            },
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "recent_errors": recent_errors,
            "configuration": {
                "max_retry_attempts": self.config.retry_config.max_attempts,
                "base_retry_delay": self.config.retry_config.base_delay,
                "max_retry_delay": self.config.retry_config.max_delay,
                "unicode_fallback_enabled": self.config.enable_unicode_fallback,
            }
        }
    
    def reset_stats(self) -> None:
        """Reset all error handling statistics."""
        logger.info("Resetting error handler statistics")
        
        self._error_counts.clear()
        self._error_history.clear()
        self._fallback_count = 0
        self._retry_count = 0
        self._total_operations = 0
        
        self.circuit_breaker.reset()
    
    def force_circuit_open(self, reason: str = "Manual override") -> None:
        """
        Manually open the circuit breaker.
        
        Args:
            reason: Reason for opening the circuit
        """
        self.circuit_breaker.force_open(reason)
    
    def force_circuit_close(self, reason: str = "Manual override") -> None:
        """
        Manually close the circuit breaker.
        
        Args:
            reason: Reason for closing the circuit
        """
        self.circuit_breaker.force_close(reason)
    
    def is_circuit_open(self) -> bool:
        """
        Check if circuit breaker is open.
        
        Returns:
            True if circuit is open (FastText disabled)
        """
        return self.circuit_breaker.is_open()
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status of the error handling system.
        
        Returns:
            Dictionary with health indicators
        """
        stats = self.get_error_stats()
        
        # Determine health status
        error_rate = stats["rates"]["error_rate_percent"]
        fallback_rate = stats["rates"]["fallback_rate_percent"]
        circuit_open = self.is_circuit_open()
        
        if circuit_open:
            health = "degraded"
            status = "Circuit breaker open - using fallback only"
        elif error_rate > 50:
            health = "unhealthy"
            status = f"High error rate: {error_rate}%"
        elif error_rate > 20 or fallback_rate > 30:
            health = "degraded"
            status = f"Elevated error/fallback rates: {error_rate}%/{fallback_rate}%"
        else:
            health = "healthy"
            status = "Normal operation"
        
        return {
            "health": health,
            "status": status,
            "circuit_breaker_open": circuit_open,
            "error_rate_percent": error_rate,
            "fallback_rate_percent": fallback_rate,
            "total_operations": stats["total_operations"],
            "recent_error_count": len(stats["recent_errors"]),
        }