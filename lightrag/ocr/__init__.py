"""
OCR module for LightRAG - Intelligent OCR engine selection and processing.

This module provides OCR capabilities for processing scanned PDF documents,
with support for DeepSeek OCR, MinerU OCR, and Docling.
It includes:

- OCRConfig: Configuration dataclass for OCR settings
- DeepSeekOCRExtractor: Client for DeepSeek OCR API
- MinerUOCRExtractor: Client for MinerU /file_parse API
- OCREngineSelector: Intelligent OCR engine selection
- MarkdownPreprocessor: Markdown preprocessing for entity extraction
- Custom exceptions for OCR error handling
- process_document_with_ocr: Main document processing function with intelligent OCR routing
"""

import time
import os
import tempfile
import logging
import asyncio
from typing import Optional

from lightrag.ocr.config import OCRConfig, OCRResult
from lightrag.ocr.deepseek import DeepSeekOCRExtractor
from lightrag.ocr.miner_u import MinerUOCRExtractor
from lightrag.ocr.selector import OCREngineSelector
from lightrag.ocr.markdown import MarkdownPreprocessor
from lightrag.ocr.exceptions import (
    DeepSeekOCRError,
    OCRTimeoutError,
    PDFProcessingError,
    DeepSeekAPIConnectionError,
    MinerUAPIConnectionError,
    MinerUOCRError,
    MinerUOCRTimeoutError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OCRConfig",
    "OCRResult",
    "DeepSeekOCRExtractor",
    "MinerUOCRExtractor",
    "OCREngineSelector",
    "MarkdownPreprocessor",
    "DeepSeekOCRError",
    "OCRTimeoutError",
    "PDFProcessingError",
    "DeepSeekAPIConnectionError",
    "MinerUAPIConnectionError",
    "MinerUOCRError",
    "MinerUOCRTimeoutError",
    "process_document_with_ocr",
]


async def process_document_with_ocr(
    file_bytes: bytes,
    password: Optional[str] = None,
    ocr_config: Optional[OCRConfig] = None,
    file_extension: str = ".pdf",
) -> OCRResult:
    """Process document with prioritized OCR routing.
    
    Routing:
    1. Select the configured OCR engine.
    2. Auto mode tries MinerU, then DeepSeek, then Docling when compatible.
    3. Return an OCRResult with error details if all selected engines fail.
    
    Args:
        file_bytes: File content as bytes
        password: Optional password for encrypted PDFs
        ocr_config: OCR configuration
        file_extension: extension of the file
        
    Returns:
        OCRResult: Result object with text or error
    """
    if not file_bytes:
        raise ValueError("file_bytes cannot be empty")
    
    start_time = time.time()
    if ocr_config is None:
        ocr_config = OCRConfig()
    
    logger.info(f"Processing {file_extension} with prioritized OCR routing")
    
    selector = OCREngineSelector(ocr_config)

    normalized_extension = file_extension.lower()
    if normalized_extension and not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"

    if ocr_config.engine == "auto":
        engines = selector.get_auto_engine_candidates()
    else:
        engines = [selector.select_engine()]

    last_error = None
    for engine in engines:
        if engine == "none":
            break

        if engine == "docling" and normalized_extension != ".pdf":
            logger.debug(
                "Skipping Docling OCR candidate for unsupported extension: %s",
                normalized_extension,
            )
            continue

        try:
            if engine == "mineru":
                logger.info(f"Attempting MinerU OCR for {file_extension}")
                result = await _extract_with_mineru(
                    file_bytes,
                    ocr_config,
                    start_time,
                    file_extension,
                )
            elif engine == "deepseek":
                logger.info(f"Attempting direct DeepSeek OCR for {file_extension}")
                result = await _extract_with_deepseek(
                    file_bytes,
                    ocr_config,
                    start_time,
                    file_extension,
                )
            elif engine == "docling":
                logger.info(f"Attempting Docling extraction for {file_extension}")
                result = await _extract_with_docling(file_bytes, ocr_config, start_time)
            else:
                continue

            if result.text.strip():
                return result

            last_error = f"{engine} returned no content"
            logger.warning("%s for %s.", last_error, file_extension)
            if ocr_config.engine != "auto" or not ocr_config.enable_fallback:
                return result
        except Exception as e:
            last_error = f"{engine} failed: {str(e)}"
            logger.error("%s for %s", last_error, file_extension)
            if ocr_config.engine != "auto" or not ocr_config.enable_fallback:
                return OCRResult(
                    text="",
                    engine_used=engine,
                    processing_time=time.time() - start_time,
                    page_count=0,
                    error=last_error,
                )

    return OCRResult(
        text="",
        engine_used="none",
        processing_time=time.time() - start_time,
        page_count=0,
        error=last_error
        or f"No suitable extraction method for {file_extension} or primary engine failed",
    )


async def _extract_with_deepseek(
    file_bytes: bytes,
    config: OCRConfig,
    start_time: float,
    file_extension: str = ".pdf",
) -> OCRResult:
    """Extract text using DeepSeek OCR API.
    
    Args:
        file_bytes: file content as bytes
        config: OCR configuration
        start_time: Processing start timestamp
        file_extension: string denoting the extension of the file
        
    Returns:
        OCRResult: Result with extracted text and metadata
        
    Raises:
        DeepSeekOCRError: If OCR processing fails
        OCRTimeoutError: If request exceeds timeout
        DeepSeekAPIConnectionError: If API is unreachable
    """
    # Initialize DeepSeek OCR client
    extractor = DeepSeekOCRExtractor(
        api_url=config.deepseek_api_url,
        timeout=config.timeout,
    )
    
    try:
        # Check if the file is an image
        if file_extension.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".tiff",
            ".bmp",
        ]:
            logger.info("Extracting text from image using DeepSeek OCR.")
            result = await extractor.extract_text_from_image(file_bytes)
            # Image extraction might not provide page_count, default to 1
            page_count = result.get("page_count", 1)
        else:
            # Assume PDF or document
            result = await extractor.extract_text_from_pdf(file_bytes)
            page_count = result["page_count"]
            
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=result["text"],
            engine_used="deepseek",
            processing_time=processing_time,
            page_count=page_count,
            format="markdown",  # DeepSeek OCR returns markdown
        )
        
    finally:
        # Clean up HTTP client
        await extractor.close()


async def _extract_with_mineru(
    file_bytes: bytes,
    config: OCRConfig,
    start_time: float,
    file_extension: str = ".pdf",
) -> OCRResult:
    """Extract text using MinerU OCR API."""
    extractor = MinerUOCRExtractor(
        api_url=config.mineru_api_url,
        timeout=config.timeout,
        backend=config.mineru_backend,
        lang_list=config.mineru_lang_list,
        parse_method=config.mineru_parse_method,
        return_middle_json=config.mineru_return_middle_json,
        return_images=config.mineru_return_images,
        formula_enable=config.mineru_formula_enable,
        table_enable=config.mineru_table_enable,
    )

    try:
        normalized_extension = file_extension.lower()
        if normalized_extension and not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"
        filename = f"document{normalized_extension or '.pdf'}"

        result = await extractor.extract_text_from_document(file_bytes, filename)
        processing_time = time.time() - start_time

        return OCRResult(
            text=result["text"],
            engine_used="mineru",
            processing_time=processing_time,
            page_count=result.get("page_count", 0),
            format="markdown",
        )

    finally:
        await extractor.close()


async def _extract_with_docling(
    file_bytes: bytes,
    config: OCRConfig,
    start_time: float,
) -> OCRResult:
    """Extract text using Docling.
    
    Args:
        file_bytes: PDF file content as bytes
        config: OCR configuration
        start_time: Processing start timestamp
        
    Returns:
        OCRResult: Result with extracted text and metadata
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
    except ImportError as e:
        logger.error(f"Docling dependencies not available: {e}")
        raise ImportError(
            "Docling mode requires docling. "
            "Install with: pip install docling"
        ) from e

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        converter = DocumentConverter(allowed_formats=[InputFormat.PDF])
        
        def convert_and_export():
            res = converter.convert(tmp_path)
            md = res.document.export_to_markdown()
            pages = res.document.num_pages if hasattr(res.document, "num_pages") else 0
            return md, pages

        markdown_content, page_count = await asyncio.to_thread(convert_and_export)


        processing_time = time.time() - start_time

        return OCRResult(
            text=markdown_content,
            engine_used="docling",
            processing_time=processing_time,
            page_count=page_count,
            format="markdown",
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
