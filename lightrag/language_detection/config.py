"""
Configuration management for FastText language detection.

This module provides configuration schema, validation, and environment variable
parsing for the FastText language detection system. It follows LightRAG's
configuration patterns and integrates with the existing logging infrastructure.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path

from .logger import get_fasttext_logger

# Use dedicated FastText logger
logger = get_fasttext_logger("config")


@dataclass
class FastTextConfig:
    """Configuration for FastText detection system."""
    
    # Core functionality settings
    enabled: bool = True
    model_path: Optional[str] = None
    cache_dir: str = "~/.lightrag/models/"
    cache_size: int = 10000
    cache_ttl: int = 3600  # seconds
    
    # Confidence thresholds for hybrid strategy
    vietnamese_threshold: float = 0.7
    english_threshold: float = 0.5
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.5
    
    # Performance settings
    max_text_length: int = 100000
    min_text_length_for_fasttext: int = 10
    batch_size: int = 32
    
    # Download settings
    model_url: str = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    model_checksum: str = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
    download_timeout: int = 300
    max_retries: int = 3
    
    # Logging and monitoring
    log_detection_method: bool = True
    log_performance_metrics: bool = False
    
    # Error handling and resilience settings
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_failure_window: int = 300  # seconds
    circuit_breaker_recovery_timeout: int = 60  # seconds
    enable_retry_logic: bool = True
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    enable_detailed_error_logging: bool = True
    
    @classmethod
    def from_env(cls) -> "FastTextConfig":
        """
        Create configuration from environment variables.
        
        Environment variables use the LIGHTRAG_FASTTEXT_ prefix following
        LightRAG's configuration patterns.
        
        Returns:
            FastTextConfig instance with values from environment variables
            or defaults if not set
        """
        config = cls()
        
        # Core functionality
        config.enabled = _get_env_bool("LIGHTRAG_FASTTEXT_ENABLED", config.enabled)
        config.model_path = _get_env_str("LIGHTRAG_FASTTEXT_MODEL_PATH", config.model_path)
        config.cache_dir = _get_env_str("LIGHTRAG_FASTTEXT_CACHE_DIR", config.cache_dir)
        config.cache_size = _get_env_int("LIGHTRAG_FASTTEXT_CACHE_SIZE", config.cache_size)
        config.cache_ttl = _get_env_int("LIGHTRAG_FASTTEXT_CACHE_TTL", config.cache_ttl)
        
        # Confidence thresholds
        config.vietnamese_threshold = _get_env_float("LIGHTRAG_FASTTEXT_VIETNAMESE_THRESHOLD", config.vietnamese_threshold)
        config.english_threshold = _get_env_float("LIGHTRAG_FASTTEXT_ENGLISH_THRESHOLD", config.english_threshold)
        config.high_confidence_threshold = _get_env_float("LIGHTRAG_FASTTEXT_HIGH_CONFIDENCE_THRESHOLD", config.high_confidence_threshold)
        config.low_confidence_threshold = _get_env_float("LIGHTRAG_FASTTEXT_LOW_CONFIDENCE_THRESHOLD", config.low_confidence_threshold)
        
        # Performance settings
        config.max_text_length = _get_env_int("LIGHTRAG_FASTTEXT_MAX_TEXT_LENGTH", config.max_text_length)
        config.min_text_length_for_fasttext = _get_env_int("LIGHTRAG_FASTTEXT_MIN_TEXT_LENGTH", config.min_text_length_for_fasttext)
        config.batch_size = _get_env_int("LIGHTRAG_FASTTEXT_BATCH_SIZE", config.batch_size)
        
        # Download settings
        config.model_url = _get_env_str("LIGHTRAG_FASTTEXT_MODEL_URL", config.model_url)
        config.model_checksum = _get_env_str("LIGHTRAG_FASTTEXT_MODEL_CHECKSUM", config.model_checksum)
        config.download_timeout = _get_env_int("LIGHTRAG_FASTTEXT_DOWNLOAD_TIMEOUT", config.download_timeout)
        config.max_retries = _get_env_int("LIGHTRAG_FASTTEXT_MAX_RETRIES", config.max_retries)
        
        # Logging and monitoring
        config.log_detection_method = _get_env_bool("LIGHTRAG_FASTTEXT_LOG_DETECTION_METHOD", config.log_detection_method)
        config.log_performance_metrics = _get_env_bool("LIGHTRAG_FASTTEXT_LOG_PERFORMANCE_METRICS", config.log_performance_metrics)
        
        # Error handling and resilience
        config.enable_circuit_breaker = _get_env_bool("LIGHTRAG_FASTTEXT_ENABLE_CIRCUIT_BREAKER", config.enable_circuit_breaker)
        config.circuit_breaker_failure_threshold = _get_env_int("LIGHTRAG_FASTTEXT_CB_FAILURE_THRESHOLD", config.circuit_breaker_failure_threshold)
        config.circuit_breaker_failure_window = _get_env_int("LIGHTRAG_FASTTEXT_CB_FAILURE_WINDOW", config.circuit_breaker_failure_window)
        config.circuit_breaker_recovery_timeout = _get_env_int("LIGHTRAG_FASTTEXT_CB_RECOVERY_TIMEOUT", config.circuit_breaker_recovery_timeout)
        config.enable_retry_logic = _get_env_bool("LIGHTRAG_FASTTEXT_ENABLE_RETRY_LOGIC", config.enable_retry_logic)
        config.retry_max_attempts = _get_env_int("LIGHTRAG_FASTTEXT_RETRY_MAX_ATTEMPTS", config.retry_max_attempts)
        config.retry_base_delay = _get_env_float("LIGHTRAG_FASTTEXT_RETRY_BASE_DELAY", config.retry_base_delay)
        config.retry_max_delay = _get_env_float("LIGHTRAG_FASTTEXT_RETRY_MAX_DELAY", config.retry_max_delay)
        config.enable_detailed_error_logging = _get_env_bool("LIGHTRAG_FASTTEXT_ENABLE_DETAILED_ERROR_LOGGING", config.enable_detailed_error_logging)
        
        return config
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        errors = []
        
        # Validate thresholds are in valid range [0.0, 1.0]
        thresholds = {
            "vietnamese_threshold": self.vietnamese_threshold,
            "english_threshold": self.english_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "low_confidence_threshold": self.low_confidence_threshold,
        }
        
        for name, value in thresholds.items():
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name} must be between 0.0 and 1.0, got {value}")
        
        # Validate threshold relationships
        if self.high_confidence_threshold <= self.low_confidence_threshold:
            errors.append(
                f"high_confidence_threshold ({self.high_confidence_threshold}) must be greater than "
                f"low_confidence_threshold ({self.low_confidence_threshold})"
            )
        
        # Validate positive integers
        positive_ints = {
            "cache_size": self.cache_size,
            "cache_ttl": self.cache_ttl,
            "max_text_length": self.max_text_length,
            "min_text_length_for_fasttext": self.min_text_length_for_fasttext,
            "batch_size": self.batch_size,
            "download_timeout": self.download_timeout,
            "max_retries": self.max_retries,
        }
        
        for name, value in positive_ints.items():
            if not isinstance(value, int) or value < 0:
                errors.append(f"{name} must be a non-negative integer, got {value}")
        
        # Validate cache directory is expandable
        try:
            expanded_cache_dir = Path(self.cache_dir).expanduser()
            if not expanded_cache_dir.parent.exists():
                logger.warning(f"Cache directory parent does not exist: {expanded_cache_dir.parent}")
        except Exception as e:
            errors.append(f"Invalid cache_dir path '{self.cache_dir}': {e}")
        
        # Validate model URL format
        if not self.model_url.startswith(("http://", "https://")):
            errors.append(f"model_url must be a valid HTTP/HTTPS URL, got '{self.model_url}'")
        
        # Validate model checksum format (should be hex string)
        if not all(c in "0123456789abcdefABCDEF" for c in self.model_checksum):
            errors.append(f"model_checksum must be a valid hexadecimal string, got '{self.model_checksum}'")
        
        if errors:
            error_msg = "FastText configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("FastText configuration validation passed")
    
    def get_expanded_cache_dir(self) -> Path:
        """
        Get the cache directory path with user home expansion.
        
        Returns:
            Path object with expanded cache directory
        """
        return Path(self.cache_dir).expanduser()
    
    def update_runtime_config(self, **kwargs) -> None:
        """
        Update configuration parameters at runtime where possible.
        
        Only certain parameters can be safely updated at runtime without
        requiring component restart. Parameters that affect model loading
        or core initialization cannot be updated.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Raises:
            ValueError: If attempting to update non-runtime parameters
            ValueError: If new values fail validation
        """
        # Parameters that can be safely updated at runtime
        runtime_updatable = {
            "vietnamese_threshold", "english_threshold", 
            "high_confidence_threshold", "low_confidence_threshold",
            "max_text_length", "min_text_length_for_fasttext",
            "batch_size", "log_detection_method", "log_performance_metrics"
        }
        
        # Parameters that require restart (cannot be updated at runtime)
        restart_required = {
            "enabled", "model_path", "cache_dir", "cache_size", "cache_ttl",
            "model_url", "model_checksum", "download_timeout", "max_retries"
        }
        
        # Check for non-runtime parameters
        non_runtime_params = set(kwargs.keys()) & restart_required
        if non_runtime_params:
            raise ValueError(
                f"Cannot update parameters at runtime (restart required): {non_runtime_params}. "
                f"Runtime updatable parameters: {runtime_updatable}"
            )
        
        # Check for unknown parameters
        unknown_params = set(kwargs.keys()) - runtime_updatable
        if unknown_params:
            raise ValueError(f"Unknown configuration parameters: {unknown_params}")
        
        # Store original values for rollback on validation failure
        original_values = {}
        for key in kwargs:
            if hasattr(self, key):
                original_values[key] = getattr(self, key)
        
        try:
            # Update values
            for key, value in kwargs.items():
                setattr(self, key, value)
            
            # Validate the updated configuration
            self.validate()
            
            logger.info(f"Runtime configuration updated successfully: {kwargs}")
            
        except Exception as e:
            # Rollback on validation failure
            for key, original_value in original_values.items():
                setattr(self, key, original_value)
            
            logger.error(f"Runtime configuration update failed, rolled back: {e}")
            raise ValueError(f"Configuration update failed: {e}")
    
    def get_runtime_updatable_params(self) -> Dict[str, any]:
        """
        Get current values of runtime-updatable parameters.
        
        Returns:
            Dictionary of parameters that can be updated at runtime
        """
        runtime_params = {
            "vietnamese_threshold": self.vietnamese_threshold,
            "english_threshold": self.english_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "low_confidence_threshold": self.low_confidence_threshold,
            "max_text_length": self.max_text_length,
            "min_text_length_for_fasttext": self.min_text_length_for_fasttext,
            "batch_size": self.batch_size,
            "log_detection_method": self.log_detection_method,
            "log_performance_metrics": self.log_performance_metrics,
        }
        return runtime_params
    
    def to_dict(self) -> Dict[str, any]:
        """
        Convert configuration to dictionary for logging/debugging.
        
        Returns:
            Dictionary representation of configuration (excluding sensitive data)
        """
        config_dict = {
            "enabled": self.enabled,
            "cache_dir": self.cache_dir,
            "cache_size": self.cache_size,
            "cache_ttl": self.cache_ttl,
            "vietnamese_threshold": self.vietnamese_threshold,
            "english_threshold": self.english_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "low_confidence_threshold": self.low_confidence_threshold,
            "max_text_length": self.max_text_length,
            "min_text_length_for_fasttext": self.min_text_length_for_fasttext,
            "batch_size": self.batch_size,
            "download_timeout": self.download_timeout,
            "max_retries": self.max_retries,
            "log_detection_method": self.log_detection_method,
            "log_performance_metrics": self.log_performance_metrics,
        }
        
        # Include model_path if set, but not the full URL or checksum for security
        if self.model_path:
            config_dict["model_path"] = self.model_path
        
        return config_dict


def _get_env_str(key: str, default: Optional[str]) -> Optional[str]:
    """Get string value from environment variable."""
    value = os.getenv(key)
    return value if value is not None else default


def _get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer value for {key}: '{value}', using default {default}")
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float value for {key}: '{value}', using default {default}")
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(key)
    if value is None:
        return default
    
    # Convert string to boolean
    value_lower = value.lower()
    if value_lower in ("true", "1", "yes", "on"):
        return True
    elif value_lower in ("false", "0", "no", "off"):
        return False
    else:
        logger.warning(f"Invalid boolean value for {key}: '{value}', using default {default}")
        return default