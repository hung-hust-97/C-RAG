"""
OCR configuration dataclasses for LightRAG.

This module defines configuration structures for OCR processing,
including engine selection, API endpoints, and processing parameters.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """Configuration for OCR processing.
    
    Attributes:
        engine: OCR engine selection ('auto', 'deepseek', 'mineru', 'docling', 'none')
        deepseek_api_url: DeepSeek OCR API endpoint URL
        mineru_api_url: MinerU /file_parse API endpoint URL
        enable_fallback: Enable fallback to another configured extractor
        timeout: Request timeout in seconds
        preserve_markdown_structure: Keep markdown structure for entity extraction
    """
    
    engine: str = "auto"
    deepseek_api_url: str = "http://10.1.6.52:8006/ocr/pdf"
    deepseek_image_api_url: str = "http://10.1.6.52:8006/ocr/image"
    mineru_api_url: str = "http://10.1.6.52:8009/file_parse"
    mineru_backend: str = "vlm-auto-engine"
    mineru_lang_list: str = "en,vn"
    mineru_parse_method: str = "auto"
    mineru_return_middle_json: bool = False
    mineru_return_images: bool = False
    mineru_formula_enable: bool = True
    mineru_table_enable: bool = True
    enable_fallback: bool = True
    timeout: int = 1200
    preserve_markdown_structure: bool = True
    
    @classmethod
    def from_global_args(cls, global_args) -> "OCRConfig":
        """Load OCR configuration from global args.
        
        Args:
            global_args: Global configuration namespace from lightrag.api.config
            
        Returns:
            OCRConfig: Configured OCR settings
        """
        from lightrag.utils import get_env_value
        
        # Load OCR configuration from environment variables via get_env_value
        engine = get_env_value("DEEPSEEK_OCR_ENGINE", "auto")
        deepseek_api_url = get_env_value(
            "DEEPSEEK_API_URL", "http://10.1.6.52:8006/ocr/pdf"
        )
        deepseek_image_api_url = get_env_value(
            "DEEPSEEK_IMAGE_API_URL", "http://10.1.6.52:8006/ocr/image"
        )
        mineru_api_url = get_env_value(
            "MINERU_API_URL", "http://10.1.6.52:8009/file_parse"
        )
        mineru_backend = get_env_value("MINERU_BACKEND", "vlm-auto-engine")
        mineru_lang_list = get_env_value("MINERU_LANG_LIST", "en,vn")
        mineru_parse_method = get_env_value("MINERU_PARSE_METHOD", "auto")
        mineru_return_middle_json = get_env_value(
            "MINERU_RETURN_MIDDLE_JSON", False, bool
        )
        mineru_return_images = get_env_value("MINERU_RETURN_IMAGES", False, bool)
        mineru_formula_enable = get_env_value("MINERU_FORMULA_ENABLE", True, bool)
        mineru_table_enable = get_env_value("MINERU_TABLE_ENABLE", True, bool)
        enable_fallback = get_env_value("DEEPSEEK_OCR_FALLBACK", True, bool)
        timeout = get_env_value("DEEPSEEK_OCR_TIMEOUT", 1200, int)
        preserve_markdown_structure = get_env_value(
            "OCR_PRESERVE_MARKDOWN_STRUCTURE", True, bool
        )
        
        config = cls(
            engine=engine,
            deepseek_api_url=deepseek_api_url,
            deepseek_image_api_url=deepseek_image_api_url,
            mineru_api_url=mineru_api_url,
            mineru_backend=mineru_backend,
            mineru_lang_list=mineru_lang_list,
            mineru_parse_method=mineru_parse_method,
            mineru_return_middle_json=mineru_return_middle_json,
            mineru_return_images=mineru_return_images,
            mineru_formula_enable=mineru_formula_enable,
            mineru_table_enable=mineru_table_enable,
            enable_fallback=enable_fallback,
            timeout=timeout,
            preserve_markdown_structure=preserve_markdown_structure,
        )
        
        # Validate configuration
        config.validate()
        
        return config
    
    def validate(self) -> None:
        """Validate OCR configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate engine value
        valid_engines = [
            "auto",
            "deepseek",
            "mineru",
            "docling",
            "none",
        ]
        if self.engine not in valid_engines:
            raise ValueError(
                f"Invalid OCR engine '{self.engine}'. "
                f"Must be one of: {', '.join(valid_engines)}"
            )
        
        # Validate DeepSeek API URL when it is required or configured.
        if self.engine == "deepseek" and not self.deepseek_api_url:
            raise ValueError("deepseek_api_url is required when engine is 'deepseek'")

        if self.deepseek_api_url:
            if not (
                self.deepseek_api_url.startswith("http://")
                or self.deepseek_api_url.startswith("https://")
            ):
                raise ValueError(
                    f"Invalid deepseek_api_url '{self.deepseek_api_url}'. "
                    "Must start with http:// or https://"
                )

        if self.engine == "mineru" and not self.mineru_api_url:
            raise ValueError("mineru_api_url is required when engine is 'mineru'")

        if self.mineru_api_url:
            if not (
                self.mineru_api_url.startswith("http://")
                or self.mineru_api_url.startswith("https://")
            ):
                raise ValueError(
                    f"Invalid mineru_api_url '{self.mineru_api_url}'. "
                    "Must start with http:// or https://"
                )

        if self.engine == "auto" and not (
            self.mineru_api_url or self.deepseek_api_url
        ):
            logger.warning(
                "Auto OCR configured without MinerU or DeepSeek endpoint. "
                "Only local Docling can be selected if installed."
            )
        
        # Validate timeout
        if self.timeout <= 0:
            raise ValueError(
                f"Invalid timeout {self.timeout}. Must be a positive integer"
            )
        
        # Log configuration warnings
        if self.engine == "deepseek" and not self.enable_fallback:
            logger.warning(
                "DeepSeek OCR configured without fallback. "
                "Processing will fail if DeepSeek API is unavailable."
            )


@dataclass
class OCRResult:
    """Result from OCR processing.
    
    Attributes:
        text: Extracted text content
        engine_used: OCR engine that processed the document
        processing_time: Processing time in seconds
        page_count: Number of pages processed
        format: Output format ('plain' or 'markdown')
        confidence: OCR confidence score (if available)
        error: Error message (if processing failed)
    """
    
    text: str
    engine_used: str
    processing_time: float
    page_count: int
    format: str = "plain"
    confidence: Optional[float] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        """Validate OCRResult after initialization."""
        # Validate engine_used
        valid_engines = [
            "deepseek",
            "mineru",
            "none",
            "docling",
        ]
        if self.engine_used not in valid_engines:
            raise ValueError(
                f"Invalid engine_used '{self.engine_used}'. "
                f"Must be one of: {', '.join(valid_engines)}"
            )
        
        # Validate format
        valid_formats = ["plain", "markdown"]
        if self.format not in valid_formats:
            raise ValueError(
                f"Invalid format '{self.format}'. "
                f"Must be one of: {', '.join(valid_formats)}"
            )
        
        # Validate processing_time
        if self.processing_time < 0:
            raise ValueError(
                f"Invalid processing_time {self.processing_time}. Must be >= 0"
            )
        
        # Validate page_count
        if self.page_count < 0:
            raise ValueError(
                f"Invalid page_count {self.page_count}. Must be >= 0"
            )
        
        # Validate confidence if provided
        if self.confidence is not None:
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(
                    f"Invalid confidence {self.confidence}. Must be between 0.0 and 1.0"
                )
