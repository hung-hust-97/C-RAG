"""
OCR engine selection logic for LightRAG.

This module provides intelligent OCR engine selection based on availability
and configuration, supporting DeepSeek OCR, MinerU OCR, and Docling.
"""

import logging
from typing import Dict

from lightrag.ocr.config import OCRConfig
from lightrag.ocr.deepseek import DeepSeekOCRExtractor
from lightrag.ocr.miner_u import MinerUOCRExtractor

logger = logging.getLogger(__name__)


class OCREngineSelector:
    """Intelligent OCR engine selector.
    
    This class manages OCR engine selection based on configuration and
    availability, providing automatic fallback and engine availability caching
    for optimal performance.
    
    Attributes:
        config: OCR configuration settings
        _availability_cache: Cache of engine availability status
    """
    
    def __init__(self, config: OCRConfig):
        """Initialize OCR engine selector.
        
        Args:
            config: OCR configuration containing engine preferences,
                   API endpoints, and processing parameters
                   
        Raises:
            ValueError: If config is None or invalid
        """
        if config is None:
            raise ValueError("config cannot be None")
        
        # Store configuration
        self.config = config
        
        # Initialize engine availability cache
        # Cache structure: {engine_name: (is_available, last_check_timestamp)}
        self._availability_cache: Dict[str, bool] = {}
        
        logger.info(
            f"Initialized OCR engine selector: engine={config.engine}, "
            f"fallback={config.enable_fallback}"
        )
    
    def select_engine(self) -> str:
        """Select appropriate OCR engine based on configuration and availability.
        
        Implements intelligent engine selection with the following logic:
        1. If engine is explicitly configured (not 'auto'), attempt to use that engine
        2. If explicit engine is unavailable and fallback is disabled, return 'none'
        3. If engine is 'auto', select with priority: mineru > deepseek > docling > none
        4. Log the selection decision for monitoring and debugging
        
        Returns:
            str: Selected engine name ('deepseek', 'mineru', 'docling', or 'none')
            
        Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 9.1, 10.6
        """
        # Step 1: Check explicit engine selection
        if self.config.engine != "auto":
            if self.config.engine == "deepseek":
                if self._is_deepseek_available():
                    logger.info(
                        "Selected DeepSeek OCR engine (explicit configuration)"
                    )
                    return "deepseek"
                else:
                    logger.warning(
                        f"Requested engine 'deepseek' not available "
                        f"(API: {self.config.deepseek_api_url})"
                    )
                    if not self.config.enable_fallback:
                        logger.warning("Fallback disabled, no OCR engine available")
                        return "none"

            elif self.config.engine == "mineru":
                if self._is_mineru_available():
                    logger.info("Selected MinerU OCR engine (explicit configuration)")
                    return "mineru"
                else:
                    logger.warning(
                        f"Requested engine 'mineru' not available "
                        f"(API: {self.config.mineru_api_url})"
                    )
                    if not self.config.enable_fallback:
                        logger.warning("Fallback disabled, no OCR engine available")
                        return "none"
            
            elif self.config.engine == "docling":
                if self._is_docling_available():
                    logger.info(
                        "Selected Docling extraction engine "
                        "(explicit configuration)"
                    )
                    return "docling"
                else:
                    logger.warning("Requested engine 'docling' not available")
                    if not self.config.enable_fallback:
                        logger.warning("Fallback disabled, no extraction engine available")
                        return "none"
            
            elif self.config.engine == "none":
                logger.info("OCR disabled by configuration")
                return "none"
        
        # Step 2: Auto-selection with priority: mineru > deepseek > docling > none
        for priority, engine in enumerate(self.get_auto_engine_candidates(), start=1):
            logger.info(
                "Selected %s OCR engine (auto-selection, priority %s)",
                engine,
                priority,
            )
            return engine
        
        logger.warning("No OCR engine available")
        return "none"

    def get_auto_engine_candidates(self) -> list[str]:
        """Return available auto-mode OCR engines in priority order."""
        candidates: list[str] = []

        if self._is_mineru_available():
            candidates.append("mineru")

        if self._is_deepseek_available():
            candidates.append("deepseek")

        if self._is_docling_available():
            candidates.append("docling")

        return candidates

    def _is_docling_available(self) -> bool:
        """Check if Docling is available."""
        if "docling" in self._availability_cache:
            return self._availability_cache["docling"]
        try:
            import docling  # noqa: F401
            self._availability_cache["docling"] = True
            return True
        except ImportError:
            self._availability_cache["docling"] = False
            return False

    def _is_deepseek_available(self) -> bool:
        """Check if DeepSeek OCR is available.
        
        Checks if DeepSeek OCR API is properly configured and reachable.
        Uses caching to avoid repeated availability checks.
        
        Returns:
            bool: True if DeepSeek OCR is available, False otherwise
        """
        # Check cache first
        if "deepseek" in self._availability_cache:
            return self._availability_cache["deepseek"]
        
        # Check if API URL is configured
        if not self.config.deepseek_api_url:
            logger.debug("DeepSeek OCR not available: API URL not configured")
            self._availability_cache["deepseek"] = False
            return False
        
        # For now, assume available if configured
        # The actual availability check will be done by DeepSeekOCRExtractor.is_available()
        # when the engine is actually used
        is_available = bool(self.config.deepseek_api_url)
        self._availability_cache["deepseek"] = is_available
        
        if is_available:
            logger.debug(f"DeepSeek OCR available at {self.config.deepseek_api_url}")
        
        return is_available

    def _is_mineru_available(self) -> bool:
        """Check if MinerU OCR is configured."""
        if "mineru" in self._availability_cache:
            return self._availability_cache["mineru"]

        if not self.config.mineru_api_url:
            logger.debug("MinerU OCR not available: API URL not configured")
            self._availability_cache["mineru"] = False
            return False

        is_available = bool(self.config.mineru_api_url)
        self._availability_cache["mineru"] = is_available

        if is_available:
            logger.debug(f"MinerU OCR available at {self.config.mineru_api_url}")

        return is_available
    
    def get_extractor(self, engine: str):
        """Get extraction function for specified engine.
        
        Returns the appropriate extraction function or class for the given
        OCR engine name. This provides a unified interface for obtaining
        extractors regardless of the engine type.
        
        Args:
            engine: Engine name ('deepseek', 'mineru', 'docling', or 'none')
            
        Returns:
            callable or class: Extraction function/class for the specified engine.
            - For 'deepseek': Returns DeepSeekOCRExtractor class
            - For 'mineru': Returns MinerUOCRExtractor class
            - For 'docling' or 'none': Returns None
            
        Raises:
            ValueError: If engine name is not recognized
            
        Validates: Requirements 2.1, 2.2, 2.3
        """
        if engine == "deepseek":
            logger.debug("Returning DeepSeek OCR extractor")
            return DeepSeekOCRExtractor

        elif engine == "mineru":
            logger.debug("Returning MinerU OCR extractor")
            return MinerUOCRExtractor
        
        elif engine in {"docling", "none"}:
            logger.debug("No remote OCR extractor selected")
            return None
        
        else:
            raise ValueError(
                f"Unknown OCR engine '{engine}'. "
                f"Must be one of: 'deepseek', 'mineru', 'docling', 'none'"
            )
