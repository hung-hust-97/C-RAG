"""
OCR engine selection logic for LightRAG.

This module provides intelligent OCR engine selection based on availability
and configuration, supporting DeepSeek OCR and Tesseract OCR with automatic
fallback capabilities.
"""

import logging
from typing import Dict

from lightrag.ocr.config import OCRConfig
from lightrag.ocr.deepseek import DeepSeekOCRExtractor

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
        3. If engine is 'auto', select with priority: deepseek > tesseract > none
        4. Log the selection decision for monitoring and debugging
        
        Returns:
            str: Selected engine name ('deepseek', 'tesseract', or 'none')
            
        Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 9.1, 10.6
        """
        # Step 1: Check explicit engine selection
        if self.config.engine != "auto":
            if self.config.engine == "deepseek":
                if self._is_deepseek_available():
                    logger.info("Selected DeepSeek OCR engine (explicit configuration)")
                    return "deepseek"
                else:
                    logger.warning(
                        f"Requested engine 'deepseek' not available "
                        f"(API: {self.config.deepseek_api_url})"
                    )
                    if not self.config.enable_fallback:
                        logger.warning("Fallback disabled, no OCR engine available")
                        return "none"
            
            elif self.config.engine == "tesseract":
                if self._is_tesseract_available():
                    logger.info("Selected Tesseract OCR engine (explicit configuration)")
                    return "tesseract"
                else:
                    logger.warning("Requested engine 'tesseract' not available")
                    if not self.config.enable_fallback:
                        logger.warning("Fallback disabled, no OCR engine available")
                        return "none"
            
            elif self.config.engine == "none":
                logger.info("OCR disabled by configuration")
                return "none"
        
        # Step 2: Auto-selection with priority: deepseek > tesseract > none
        if self._is_deepseek_available():
            logger.info("Selected DeepSeek OCR engine (auto-selection, priority 1)")
            return "deepseek"
        
        if self._is_tesseract_available():
            logger.info("Selected Tesseract OCR engine (auto-selection, fallback)")
            return "tesseract"
        
        logger.warning("No OCR engine available")
        return "none"
    
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
    
    def _is_tesseract_available(self) -> bool:
        """Check if Tesseract OCR is available.
        
        Checks if pytesseract and pdf2image libraries are installed.
        Uses caching to avoid repeated availability checks.
        
        Returns:
            bool: True if Tesseract OCR is available, False otherwise
        """
        # Check cache first
        if "tesseract" in self._availability_cache:
            return self._availability_cache["tesseract"]
        
        # Try to import pytesseract and pdf2image
        try:
            import pytesseract  # noqa: F401
            import pdf2image  # noqa: F401
            
            logger.debug("Tesseract OCR available (pytesseract and pdf2image installed)")
            self._availability_cache["tesseract"] = True
            return True
        
        except ImportError as e:
            logger.debug(f"Tesseract OCR not available: {e}")
            self._availability_cache["tesseract"] = False
            return False
    
    def get_extractor(self, engine: str):
        """Get extraction function for specified engine.
        
        Returns the appropriate extraction function or class for the given
        OCR engine name. This provides a unified interface for obtaining
        extractors regardless of the engine type.
        
        Args:
            engine: Engine name ('deepseek', 'tesseract', or 'none')
            
        Returns:
            callable or class: Extraction function/class for the specified engine.
            - For 'deepseek': Returns DeepSeekOCRExtractor class
            - For 'tesseract': Returns tesseract extraction function
            - For 'none': Returns None
            
        Raises:
            ValueError: If engine name is not recognized
            
        Validates: Requirements 2.1, 2.2, 2.3
        """
        if engine == "deepseek":
            logger.debug("Returning DeepSeek OCR extractor")
            return DeepSeekOCRExtractor
        
        elif engine == "tesseract":
            logger.debug("Returning Tesseract OCR extractor")
            return self._get_tesseract_extractor()
        
        elif engine == "none":
            logger.debug("No OCR engine selected")
            return None
        
        else:
            raise ValueError(
                f"Unknown OCR engine '{engine}'. "
                f"Must be one of: 'deepseek', 'tesseract', 'none'"
            )
    
    def _get_tesseract_extractor(self):
        """Get Tesseract OCR extraction function.
        
        Returns a callable that extracts text from PDF bytes using Tesseract OCR.
        
        Returns:
            callable: Function that accepts file_bytes and returns extracted text
        """
        def extract_with_tesseract(file_bytes: bytes) -> str:
            """Extract text from PDF using Tesseract OCR.
            
            Args:
                file_bytes: PDF file content as bytes
                
            Returns:
                str: Extracted text content
                
            Raises:
                ImportError: If pytesseract or pdf2image not installed
                Exception: If OCR processing fails
            """
            try:
                import pytesseract
                from pdf2image import convert_from_bytes
            except ImportError as e:
                logger.error(f"Tesseract OCR dependencies not available: {e}")
                raise ImportError(
                    "Tesseract OCR requires pytesseract and pdf2image. "
                    "Install with: pip install pytesseract pdf2image"
                ) from e
            
            logger.info("Processing PDF with Tesseract OCR")
            
            # Convert PDF to images
            images = convert_from_bytes(file_bytes)
            
            # Extract text from each page
            pages_text = []
            for i, image in enumerate(images):
                logger.debug(f"Processing page {i + 1}/{len(images)} with Tesseract")
                text = pytesseract.image_to_string(image, lang=self.config.tesseract_lang)
                pages_text.append(text)
            
            full_text = "\n".join(pages_text)
            logger.info(
                f"Tesseract OCR completed: {len(images)} pages, "
                f"{len(full_text)} characters"
            )
            
            return full_text
        
        return extract_with_tesseract
