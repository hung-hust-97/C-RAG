"""
OCR module for LightRAG - Intelligent OCR engine selection and processing.

This module provides OCR capabilities for processing scanned PDF documents,
with support for DeepSeek OCR and Tesseract OCR engines. It includes:

- OCRConfig: Configuration dataclass for OCR settings
- DeepSeekOCRExtractor: Client for DeepSeek OCR API
- OCREngineSelector: Intelligent OCR engine selection
- MarkdownPreprocessor: Markdown preprocessing for entity extraction
- Custom exceptions for OCR error handling
- process_pdf_with_ocr: Main PDF processing function with intelligent OCR routing
"""

import logging
import time
from io import BytesIO
from typing import Optional

from lightrag.ocr.config import OCRConfig, OCRResult
from lightrag.ocr.deepseek import DeepSeekOCRExtractor
from lightrag.ocr.selector import OCREngineSelector
from lightrag.ocr.markdown import MarkdownPreprocessor
from lightrag.ocr.exceptions import (
    DeepSeekOCRError,
    OCRTimeoutError,
    PDFProcessingError,
    DeepSeekAPIConnectionError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OCRConfig",
    "OCRResult",
    "DeepSeekOCRExtractor",
    "OCREngineSelector",
    "MarkdownPreprocessor",
    "DeepSeekOCRError",
    "OCRTimeoutError",
    "PDFProcessingError",
    "DeepSeekAPIConnectionError",
    "process_pdf_with_ocr",
]


async def process_pdf_with_ocr(
    file_bytes: bytes,
    password: Optional[str] = None,
    ocr_config: Optional[OCRConfig] = None,
) -> OCRResult:
    """Process PDF with intelligent OCR routing.
    
    This function implements the main PDF processing workflow:
    1. Try standard text extraction using pypdf first
    2. If text found, return immediately without OCR
    3. If no text (scanned PDF), select OCR engine using OCREngineSelector
    4. Process with selected engine (deepseek or tesseract)
    5. Implement fallback chain: deepseek -> tesseract -> error
    6. Log all extraction attempts and results
    7. Return OCRResult with text and metadata
    
    Args:
        file_bytes: PDF file content as bytes
        password: Optional password for encrypted PDFs
        ocr_config: OCR configuration (if None, uses default config)
        
    Returns:
        OCRResult: Result object containing extracted text and metadata
        
    Raises:
        PDFProcessingError: If all extraction methods fail
        ValueError: If file_bytes is empty or invalid
        
    Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 6.1, 6.2, 6.3, 6.6, 9.2, 9.3
    """
    if not file_bytes:
        raise ValueError("file_bytes cannot be empty")
    
    start_time = time.time()
    
    # Use default config if not provided
    if ocr_config is None:
        ocr_config = OCRConfig()
    
    logger.info("Starting PDF processing with intelligent OCR routing")
    
    # Step 1: Try standard text extraction using pypdf first
    logger.info("Attempting standard text extraction with pypdf")
    try:
        from pypdf import PdfReader
        
        pdf_file = BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        
        # Handle encrypted PDFs
        if reader.is_encrypted:
            if not password:
                logger.warning("PDF is encrypted but no password provided")
            else:
                decrypt_result = reader.decrypt(password)
                if decrypt_result == 0:
                    logger.error("Incorrect PDF password")
                    raise PDFProcessingError("Incorrect PDF password")
        
        # Extract text from all pages
        text_content = ""
        page_count = len(reader.pages)
        for page in reader.pages:
            text_content += page.extract_text() + "\n"
        
        # Step 2: Check if text extraction succeeded
        if text_content and text_content.strip():
            processing_time = time.time() - start_time
            logger.info(
                f"Standard text extraction successful: {page_count} pages, "
                f"{len(text_content)} characters, {processing_time:.2f}s"
            )
            return OCRResult(
                text=text_content,
                engine_used="pypdf",
                processing_time=processing_time,
                page_count=page_count,
                format="plain",
            )
        
        # Step 3: No text found - this is likely a scanned PDF
        logger.info(
            f"No text content found in PDF ({page_count} pages), "
            "routing to OCR processing"
        )
        
    except Exception as e:
        logger.warning(f"pypdf extraction failed: {e}, routing to OCR")
        page_count = 0  # Unknown page count if pypdf fails
    
    # Step 4: Select OCR engine based on config and availability
    selector = OCREngineSelector(ocr_config)
    engine = selector.select_engine()
    
    logger.info(f"Selected OCR engine: {engine}")
    
    # Step 5: Process with selected engine
    if engine == "deepseek":
        try:
            logger.info("Processing with DeepSeek OCR")
            result = await _extract_with_deepseek(file_bytes, ocr_config, start_time)
            logger.info(
                f"DeepSeek OCR successful: {result.page_count} pages, "
                f"{len(result.text)} characters, {result.processing_time:.2f}s"
            )
            return result
            
        except (DeepSeekOCRError, OCRTimeoutError, DeepSeekAPIConnectionError) as e:
            logger.warning(f"DeepSeek OCR failed: {e}")
            
            # Implement fallback chain: deepseek -> tesseract
            if ocr_config.enable_fallback:
                logger.info("Fallback enabled, attempting Tesseract OCR")
                engine = "tesseract"
            else:
                logger.error("Fallback disabled, OCR processing failed")
                processing_time = time.time() - start_time
                return OCRResult(
                    text="",
                    engine_used="none",
                    processing_time=processing_time,
                    page_count=page_count,
                    format="plain",
                    error=f"DeepSeek OCR failed: {str(e)}",
                )
    
    if engine == "tesseract":
        try:
            logger.info("Processing with Tesseract OCR")
            result = await _extract_with_tesseract(file_bytes, ocr_config, start_time)
            logger.info(
                f"Tesseract OCR successful: {result.page_count} pages, "
                f"{len(result.text)} characters, {result.processing_time:.2f}s"
            )
            return result
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            processing_time = time.time() - start_time
            return OCRResult(
                text="",
                engine_used="none",
                processing_time=processing_time,
                page_count=page_count,
                format="plain",
                error=f"Tesseract OCR failed: {str(e)}",
            )
    
    # No OCR engine available or all methods failed
    logger.warning("No OCR engine available for scanned PDF")
    processing_time = time.time() - start_time
    return OCRResult(
        text="",
        engine_used="none",
        processing_time=processing_time,
        page_count=page_count,
        format="plain",
        error="No OCR engine available",
    )


async def _extract_with_deepseek(
    file_bytes: bytes,
    config: OCRConfig,
    start_time: float,
) -> OCRResult:
    """Extract text using DeepSeek OCR API.
    
    Args:
        file_bytes: PDF file content as bytes
        config: OCR configuration
        start_time: Processing start timestamp
        
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
        # Check availability first
        is_available = await extractor.is_available()
        if not is_available:
            raise DeepSeekAPIConnectionError(
                f"DeepSeek OCR API is not available: {config.deepseek_api_url}"
            )
        
        # Extract text from PDF
        result = await extractor.extract_text_from_pdf(file_bytes)
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=result["text"],
            engine_used="deepseek",
            processing_time=processing_time,
            page_count=result["page_count"],
            format="markdown",  # DeepSeek OCR returns markdown
        )
        
    finally:
        # Clean up HTTP client
        await extractor.close()


async def _extract_with_tesseract(
    file_bytes: bytes,
    config: OCRConfig,
    start_time: float,
) -> OCRResult:
    """Extract text using Tesseract OCR.
    
    Args:
        file_bytes: PDF file content as bytes
        config: OCR configuration
        start_time: Processing start timestamp
        
    Returns:
        OCRResult: Result with extracted text and metadata
        
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
    
    # Convert PDF to images
    images = convert_from_bytes(file_bytes, dpi=300)
    page_count = len(images)
    
    # Extract text from each page
    pages_text = []
    for i, image in enumerate(images):
        logger.debug(f"Processing page {i + 1}/{page_count} with Tesseract")
        text = pytesseract.image_to_string(image, lang=config.tesseract_lang)
        pages_text.append(text)
    
    full_text = "\n".join(pages_text)
    processing_time = time.time() - start_time
    
    return OCRResult(
        text=full_text,
        engine_used="tesseract",
        processing_time=processing_time,
        page_count=page_count,
        format="plain",
    )
