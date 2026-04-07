"""
FastText model lifecycle management: download, validation, loading.

This module implements comprehensive model management functionality including:
- Model download with retry logic and progress tracking
- Model validation using checksums and integrity checks
- Configurable cache directory management
- Offline operation support when model is already cached
"""

import hashlib
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

from .config import FastTextConfig
from .error_handler import ResilientErrorHandler, ErrorHandlingConfig
from .logger import get_fasttext_logger
from .security import SecureDownloader

# Use dedicated FastText logger
logger = get_fasttext_logger("model_manager")


class ModelDownloadError(Exception):
    """Raised when model download fails after all retries."""
    pass


class ModelValidationError(Exception):
    """Raised when model validation fails."""
    pass


class FastTextModelManager:
    """Manages FastText model lifecycle: download, validation, loading."""
    
    def __init__(self, config: Optional[FastTextConfig] = None, 
                 error_handler: Optional[ResilientErrorHandler] = None):
        """
        Initialize model manager with configuration and error handling.
        
        Args:
            config: FastText configuration. If None, loads from environment.
            error_handler: Error handler for resilient operations. If None, creates default.
        """
        self.config = config or FastTextConfig.from_env()
        self.error_handler = error_handler or ResilientErrorHandler(ErrorHandlingConfig())
        self.secure_downloader = SecureDownloader()
        self.cache_dir = self.config.get_expanded_cache_dir()
        self.model_filename = "lid.176.bin"
        self.model_path = self.cache_dir / self.model_filename
        
        logger.info(f"FastTextModelManager initialized with cache_dir: {self.cache_dir}, "
                   f"resilient error handling enabled, secure downloads enforced")
        
        # Ensure cache directory exists
        self._ensure_cache_directory()
    
    def _ensure_cache_directory(self) -> None:
        """Create cache directory if it doesn't exist."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache directory ensured: {self.cache_dir}")
        except OSError as e:
            logger.error(f"Failed to create cache directory {self.cache_dir}: {e}")
            raise ModelDownloadError(f"Cannot create cache directory: {e}")
    
    def ensure_model_available(self) -> str:
        """
        Download and validate model if not present, return path.
        
        Returns:
            str: Path to the validated model file
            
        Raises:
            ModelDownloadError: If model cannot be downloaded or validated
        """
        # Check if custom model path is specified
        if self.config.model_path:
            custom_path = Path(self.config.model_path).expanduser()
            if custom_path.exists():
                if self.validate_model(str(custom_path)):
                    logger.info(f"Using custom model path: {custom_path}")
                    return str(custom_path)
                else:
                    logger.warning(f"Custom model path validation failed: {custom_path}")
            else:
                logger.warning(f"Custom model path does not exist: {custom_path}")
        
        # Check if model exists in cache
        if self.model_path.exists():
            if self.validate_model(str(self.model_path)):
                logger.info(f"Using cached model: {self.model_path}")
                return str(self.model_path)
            else:
                logger.warning(f"Cached model validation failed, re-downloading: {self.model_path}")
                # Remove corrupted model
                try:
                    self.model_path.unlink()
                except OSError as e:
                    logger.warning(f"Failed to remove corrupted model: {e}")
        
        # Download model
        downloaded_path = self.download_model()
        
        # Validate downloaded model
        if not self.validate_model(downloaded_path):
            raise ModelDownloadError("Downloaded model failed validation")
        
        logger.info(f"Model successfully downloaded and validated: {downloaded_path}")
        return downloaded_path
    
    def validate_model(self, model_path: str) -> bool:
        """
        Verify model integrity using checksums.
        
        Args:
            model_path: Path to the model file to validate
            
        Returns:
            bool: True if model is valid, False otherwise
        """
        try:
            path = Path(model_path)
            if not path.exists():
                logger.debug(f"Model file does not exist: {model_path}")
                return False
            
            # Check file size (lid.176.bin should be around 126MB)
            file_size = path.stat().st_size
            min_size = 100 * 1024 * 1024  # 100MB minimum
            max_size = 200 * 1024 * 1024  # 200MB maximum
            
            if not (min_size <= file_size <= max_size):
                logger.warning(f"Model file size {file_size} bytes is outside expected range [{min_size}, {max_size}]")
                return False
            
            # Calculate SHA256 checksum
            logger.debug(f"Calculating checksum for model: {model_path}")
            sha256_hash = hashlib.sha256()
            
            with open(path, "rb") as f:
                # Read in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            
            calculated_checksum = sha256_hash.hexdigest()
            expected_checksum = self.config.model_checksum
            
            if calculated_checksum == expected_checksum:
                logger.debug(f"Model checksum validation passed: {calculated_checksum}")
                return True
            else:
                logger.warning(f"Model checksum mismatch. Expected: {expected_checksum}, Got: {calculated_checksum}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating model {model_path}: {e}")
            return False
    
    def download_model(self, force: bool = False) -> str:
        """
        Download model with resilient retry logic and progress tracking.
        
        Args:
            force: If True, download even if model already exists
            
        Returns:
            str: Path to the downloaded model file
            
        Raises:
            ModelDownloadError: If download fails after all retries
        """
        if not force and self.model_path.exists():
            logger.info(f"Model already exists: {self.model_path}")
            return str(self.model_path)
        
        # Validate URL security (Requirement 10.3)
        if not self.secure_downloader.validate_url(self.config.model_url):
            raise ModelDownloadError(f"Invalid or insecure model URL: {self.config.model_url}")
        
        logger.info(f"Downloading FastText model from: {self.config.model_url}")
        
        def _download_operation():
            """Internal download operation with progress tracking."""
            # Create temporary download path
            temp_path = self.model_path.with_suffix(".tmp")
            
            # Remove temporary file if it exists from previous attempt
            if temp_path.exists():
                temp_path.unlink()
            
            # Download with progress tracking
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if block_num % 1000 == 0:  # Log every 1000 blocks to avoid spam
                        logger.debug(f"Download progress: {percent}%")
            
            # Perform the download
            start_time = time.time()
            urlretrieve(
                self.config.model_url,
                str(temp_path),
                reporthook=progress_hook
            )
            download_time = time.time() - start_time
            
            # Check if download was successful
            if not temp_path.exists():
                raise ModelDownloadError("Download completed but file not found")
            
            file_size = temp_path.stat().st_size
            logger.info(f"Download completed in {download_time:.1f}s, size: {file_size} bytes")
            
            # Verify checksum before moving to final location (Requirement 10.4)
            logger.info("Verifying model integrity with checksum...")
            if not self.secure_downloader.verify_checksum(
                str(temp_path), 
                self.config.model_checksum,
                algorithm='sha256'
            ):
                # Remove invalid file
                temp_path.unlink()
                raise ModelDownloadError("Downloaded model failed checksum verification")
            
            logger.info("Model checksum verification passed")
            
            # Move temporary file to final location atomically
            temp_path.rename(self.model_path)
            
            return str(self.model_path)
        
        def _download_fallback():
            """Fallback when download fails completely."""
            logger.error("Model download failed after all retry attempts")
            raise ModelDownloadError("Failed to download model after all retries")
        
        try:
            return self.error_handler.execute_with_resilience(
                operation=_download_operation,
                fallback_operation=_download_fallback,
                operation_name="model_download"
            )
        except Exception as e:
            # Clean up any temporary files
            temp_path = self.model_path.with_suffix(".tmp")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise ModelDownloadError(f"Model download failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata and status information.
        
        Returns:
            Dict containing model information
        """
        info = {
            "model_url": self.config.model_url,
            "expected_checksum": self.config.model_checksum,
            "cache_dir": str(self.cache_dir),
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "model_valid": False,
            "file_size": None,
            "last_modified": None,
        }
        
        if self.model_path.exists():
            try:
                stat = self.model_path.stat()
                info["file_size"] = stat.st_size
                info["last_modified"] = stat.st_mtime
                info["model_valid"] = self.validate_model(str(self.model_path))
            except OSError as e:
                logger.warning(f"Error getting model file info: {e}")
        
        # Add custom model path info if specified
        if self.config.model_path:
            custom_path = Path(self.config.model_path).expanduser()
            info["custom_model_path"] = str(custom_path)
            info["custom_model_exists"] = custom_path.exists()
            if custom_path.exists():
                info["custom_model_valid"] = self.validate_model(str(custom_path))
        
        return info
    
    def cleanup_old_models(self, keep_current: bool = True) -> int:
        """
        Clean up old model files to prevent disk space accumulation.
        
        Args:
            keep_current: If True, keep the current model file
            
        Returns:
            int: Number of files cleaned up
        """
        cleaned_count = 0
        
        try:
            # Look for model files in cache directory
            for file_path in self.cache_dir.glob("*.bin"):
                if keep_current and file_path == self.model_path:
                    continue
                
                try:
                    file_path.unlink()
                    cleaned_count += 1
                    logger.info(f"Cleaned up old model file: {file_path}")
                except OSError as e:
                    logger.warning(f"Failed to clean up {file_path}: {e}")
            
            # Also clean up temporary files
            for temp_file in self.cache_dir.glob("*.tmp"):
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
                except OSError as e:
                    logger.warning(f"Failed to clean up {temp_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old model files")
        
        return cleaned_count