"""
OCR configuration dataclasses for LightRAG.

This module defines configuration structures for OCR processing,
including engine selection, API endpoints, and processing parameters.
"""

from dataclasses import dataclass
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

VALID_MINERU_TIERS = {"", "flash", "basic", "standard", "advanced"}
VALID_MINERU_OCR_MODES = {"auto", "txt", "ocr"}


@dataclass
class OCRConfig:
    """Configuration for OCR processing.

    Attributes:
        engine: OCR engine selection ('auto', 'deepseek', 'mineru', 'docling', 'none')
        deepseek_api_url: DeepSeek OCR API endpoint URL
        mineru_api_url: MinerU v4 base URL (e.g. http://10.1.6.4:8000)
        mineru_api_key: Optional Bearer token when MinerU server enables --api-key
        mineru_tier: MinerU v4 quality tier ('flash', 'basic', 'standard', 'advanced', or '')
        mineru_ocr_mode: MinerU v4 OCR mode ('auto', 'txt', 'ocr')
        mineru_poll_interval: Poll interval in seconds for GET /v1/tasks/{task_id}
        enable_fallback: Enable fallback to another configured extractor
        timeout: Request timeout in seconds
        preserve_markdown_structure: Keep markdown structure for entity extraction
    """

    engine: str = "auto"
    deepseek_api_url: str = "http://10.1.6.52:8006/ocr/pdf"
    deepseek_image_api_url: str = "http://10.1.6.52:8006/ocr/image"
    mineru_api_url: str = "http://10.1.6.4:8000"
    mineru_api_key: Optional[str] = None
    mineru_tier: str = "standard"
    mineru_ocr_mode: str = "auto"
    mineru_poll_interval: float = 2.0
    # Legacy v3.x attributes retained for backward compatibility
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
        """Load OCR configuration from environment / global args."""
        from lightrag.utils import get_env_value

        engine = get_env_value(
            "DEEPSEEK_OCR_ENGINE",
            getattr(global_args, "deepseek_ocr_engine", "auto"),
        )
        deepseek_api_url = get_env_value(
            "DEEPSEEK_API_URL",
            getattr(global_args, "deepseek_api_url", "http://10.1.6.52:8006/ocr/pdf"),
        )
        deepseek_image_api_url = get_env_value(
            "DEEPSEEK_IMAGE_API_URL",
            "http://10.1.6.52:8006/ocr/image",
        )
        mineru_api_url = get_env_value(
            "MINERU_API_URL",
            getattr(global_args, "mineru_api_url", "http://10.1.6.4:8000"),
        )
        raw_api_key = get_env_value(
            "MINERU_API_KEY",
            getattr(global_args, "mineru_api_key", None),
        )
        mineru_api_key = (
            raw_api_key.strip()
            if isinstance(raw_api_key, str) and raw_api_key.strip()
            else None
        )
        mineru_tier = get_env_value(
            "MINERU_TIER",
            getattr(global_args, "mineru_tier", "standard"),
        )
        mineru_parse_method = get_env_value(
            "MINERU_PARSE_METHOD",
            getattr(global_args, "mineru_parse_method", "auto"),
        )
        if os.getenv("MINERU_OCR_MODE") is not None:
            mineru_ocr_mode = get_env_value("MINERU_OCR_MODE", "auto")
        elif os.getenv("MINERU_PARSE_METHOD") is not None and mineru_parse_method in VALID_MINERU_OCR_MODES:
            mineru_ocr_mode = mineru_parse_method
        else:
            mineru_ocr_mode = getattr(global_args, "mineru_ocr_mode", "auto")

        mineru_poll_interval = get_env_value(
            "MINERU_POLL_INTERVAL",
            getattr(global_args, "mineru_poll_interval", 2.0),
            float,
        )
        mineru_backend = get_env_value("MINERU_BACKEND", "vlm-auto-engine")
        mineru_lang_list = get_env_value("MINERU_LANG_LIST", "en,vn")
        mineru_return_middle_json = get_env_value(
            "MINERU_RETURN_MIDDLE_JSON", False, bool
        )
        mineru_return_images = get_env_value("MINERU_RETURN_IMAGES", False, bool)
        mineru_formula_enable = get_env_value("MINERU_FORMULA_ENABLE", True, bool)
        mineru_table_enable = get_env_value("MINERU_TABLE_ENABLE", True, bool)
        enable_fallback = get_env_value(
            "DEEPSEEK_OCR_FALLBACK",
            getattr(global_args, "deepseek_ocr_fallback", True),
            bool,
        )
        timeout = get_env_value(
            "DEEPSEEK_OCR_TIMEOUT",
            getattr(global_args, "deepseek_ocr_timeout", 1200),
            int,
        )
        preserve_markdown_structure = get_env_value(
            "OCR_PRESERVE_MARKDOWN_STRUCTURE",
            getattr(global_args, "ocr_preserve_markdown_structure", True),
            bool,
        )

        config = cls(
            engine=engine,
            deepseek_api_url=deepseek_api_url,
            deepseek_image_api_url=deepseek_image_api_url,
            mineru_api_url=mineru_api_url,
            mineru_api_key=mineru_api_key,
            mineru_tier=mineru_tier,
            mineru_ocr_mode=mineru_ocr_mode,
            mineru_poll_interval=mineru_poll_interval,
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

        config.validate()
        return config

    def validate(self) -> None:
        """Validate OCR configuration."""
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

        normalized_tier = (self.mineru_tier or "").strip().lower()
        if normalized_tier not in VALID_MINERU_TIERS:
            raise ValueError(
                f"Invalid mineru_tier '{self.mineru_tier}'. "
                "Must be one of: flash, basic, standard, advanced, or empty"
            )
        self.mineru_tier = normalized_tier

        normalized_ocr_mode = (self.mineru_ocr_mode or "auto").strip().lower()
        if normalized_ocr_mode not in VALID_MINERU_OCR_MODES:
            raise ValueError(
                f"Invalid mineru_ocr_mode '{self.mineru_ocr_mode}'. "
                "Must be one of: auto, txt, ocr"
            )
        self.mineru_ocr_mode = normalized_ocr_mode

        if self.mineru_poll_interval <= 0:
            raise ValueError(
                f"Invalid mineru_poll_interval {self.mineru_poll_interval}. Must be > 0"
            )

        if self.engine == "auto" and not self.mineru_api_url:
            logger.warning(
                "Auto OCR configured without MinerU endpoint. "
                "Only local Docling can be selected if installed."
            )

        if self.timeout <= 0:
            raise ValueError(
                f"Invalid timeout {self.timeout}. Must be a positive integer"
            )

        if self.engine == "deepseek" and not self.enable_fallback:
            logger.warning(
                "DeepSeek OCR configured without fallback. "
                "Processing will fail if DeepSeek API is unavailable."
            )


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    engine_used: str
    processing_time: float
    page_count: int
    format: str = "plain"
    confidence: Optional[float] = None
    error: Optional[str] = None

    def __post_init__(self):
        """Validate OCRResult after initialization."""
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

        valid_formats = ["plain", "markdown"]
        if self.format not in valid_formats:
            raise ValueError(
                f"Invalid format '{self.format}'. "
                f"Must be one of: {', '.join(valid_formats)}"
            )

        if self.processing_time < 0:
            raise ValueError(
                f"Invalid processing_time {self.processing_time}. Must be >= 0"
            )

        if self.page_count < 0:
            raise ValueError(
                f"Invalid page_count {self.page_count}. Must be >= 0"
            )

        if self.confidence is not None:
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(
                    f"Invalid confidence {self.confidence}. Must be between 0.0 and 1.0"
                )
