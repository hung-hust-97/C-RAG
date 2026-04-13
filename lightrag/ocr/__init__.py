"""
OCR module for LightRAG - Intelligent OCR engine selection and processing.

This module provides OCR capabilities for processing scanned PDF documents,
with support for DeepSeek OCR and Tesseract OCR engines. It includes:

- OCRConfig: Configuration dataclass for OCR settings
- DeepSeekOCRExtractor: Client for DeepSeek OCR API
- OCREngineSelector: Intelligent OCR engine selection
- MarkdownPreprocessor: Markdown preprocessing for entity extraction
- Custom exceptions for OCR error handling
- process_document_with_ocr: Main document processing function with intelligent OCR routing
"""

import time
import os
import tempfile
import logging
from io import BytesIO
from typing import Optional, List, Dict, Any

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
    "process_document_with_ocr",
]


async def process_document_with_ocr(
    file_bytes: bytes,
    password: Optional[str] = None,
    ocr_config: Optional[OCRConfig] = None,
    file_extension: str = ".pdf",
) -> OCRResult:
    """Process document with intelligent OCR routing.
    
    This function implements the main document processing workflow:
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
    
    # Step 4: Select OCR engine based on config and availability
    selector = OCREngineSelector(ocr_config)
    engine = selector.select_engine()
    
    logger.info(f"Selected OCR engine: {engine}")
    
    # Hybrid Extraction Logic (Docling + DeepSeek Refinement)
    if ocr_config.use_hybrid_mode and (engine in ["auto", "docling", "hybrid"]):
        try:
            logger.info("Attempting Hybrid extraction (Docling + DeepSeek Refinement)")
            result = await _hybrid_extract(file_bytes, ocr_config, start_time)
            if result.text.strip():
                return result
            logger.warning("Hybrid extraction returned no content.")
        except Exception as e:
            logger.warning(f"Hybrid extraction failed: {e}")

    # Hierarchical Extraction Logic (Fallback if Hybrid disabled or failed)
    # 1. Try DeepSeek (High Quality OCR/Markdown)
    if engine == "deepseek" or ocr_config.enable_fallback:
        try:
            logger.info("Attempting extraction with DeepSeek OCR")
            result = await _extract_with_deepseek(file_bytes, ocr_config, start_time, file_extension)
            if result.text.strip():
                return result
            logger.warning("DeepSeek OCR returned empty result.")
        except Exception as e:
            logger.warning(f"DeepSeek OCR failed: {e}")

    # 2. Try Docling (Structural Extraction)
    if engine == "docling" or ocr_config.enable_fallback:
        try:
            logger.info("Attempting extraction with Docling")
            result = await _extract_with_docling(file_bytes, ocr_config, start_time)
            if result.text.strip():
                return result
            logger.warning("Docling extraction returned empty result.")
        except Exception as e:
            logger.warning(f"Docling extraction failed: {e}")

    # 3. Try Tesseract (Local OCR Fallback)
    if engine == "tesseract" or ocr_config.enable_fallback:
        try:
            logger.info("Attempting extraction with Tesseract OCR")
            result = await _extract_with_tesseract(file_bytes, ocr_config, start_time)
            if result.text.strip():
                return result
            logger.warning("Tesseract OCR returned empty result.")
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")

    # 4. Final Fallback: pypdf (Basic Text Extraction)
    # Only attempted if all above failed or were skipped.
    try:
        logger.info("All advanced engines failed/skipped. Attempting basic pypdf extraction as last resort.")
        import pypdf
        reader = pypdf.PdfReader(BytesIO(file_bytes))
        page_count = len(reader.pages)
        extracted_text = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        
        full_text = "\n".join(extracted_text)
        if full_text.strip():
            logger.info(f"pypdf last-resort extraction successful: {len(full_text)} chars")
            return OCRResult(
                text=full_text,
                engine_used="pypdf",
                processing_time=time.time() - start_time,
                page_count=page_count,
                format="plain",
            )
    except Exception as e:
        logger.error(f"pypdf last-resort extraction also failed: {e}")

    # If we reach here, everything failed
    processing_time = time.time() - start_time
    return OCRResult(
        text="",
        engine_used="none",
        processing_time=processing_time,
        page_count=0,
        format="plain",
        error="All extraction methods (DeepSeek, Docling, Tesseract, pypdf) failed or returned no content",
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
        if file_extension.lower() in [".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"]:
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
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF]
        )
        conversion_result = converter.convert(tmp_path)
        markdown_content = conversion_result.document.export_to_markdown()
        page_count = conversion_result.document.num_pages if hasattr(conversion_result.document, "num_pages") else 0
        
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


async def _hybrid_extract(
    file_bytes: bytes,
    config: OCRConfig,
    start_time: float,
) -> OCRResult:
    """Hybrid extraction: Docling for structure + DeepSeek for complex regions.
    
    1. Process with Docling to get structure and elements.
    2. Identify 'unreadable' or 'complex' elements (tables, figures).
    3. Crop those regions using PyMuPDF.
    4. Send crops to DeepSeek Image OCR API.
    5. Reconstruct improved markdown.
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError as e:
        logger.error(f"Hybrid dependencies not available: {e}")
        raise ImportError(
            "Hybrid mode requires docling, pymupdf, and pillow. "
            "Install with: pip install docling pymupdf pillow"
        ) from e

    # 1. Run Docling
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
        
    try:
        # Configure Docling to preserve coordinates
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Let DeepSeek do the heavy lifting for images
        pipeline_options.do_table_structure = True
        
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF]
        )
        
        conversion_result = converter.convert(tmp_path)
        doc = conversion_result.document
        
        # 2. Identify and Refine Complex Elements
        # We focus on tables that might be "broken" or poorly extracted
        refinement_tasks = []
        
        for element, _level in doc.iterate_items():
            # Check for Table elements
            # Element types vary by docling version, we use string check or class check
            element_type = element.__class__.__name__
            
            if "TableItem" in element_type:
                # Check if table content looks "poor" or if it marks as an image
                table_text = element.export_to_markdown()
                if len(table_text.strip()) < config.min_table_chars or "|" not in table_text:
                    if hasattr(element, "prov") and element.prov:
                        refinement_tasks.append(element)
            
            # Optionally check for PictureItem if we want to OCR images found in PDF
            elif "PictureItem" in element_type:
                if hasattr(element, "prov") and element.prov:
                    refinement_tasks.append(element)

        if refinement_tasks:
            logger.info(f"Hybrid Refinement identifying {len(refinement_tasks)} complex regions")
            
            # Load PDF with PyMuPDF for cropping
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            deepseek_extractor = DeepSeekOCRExtractor(
                api_url=config.deepseek_api_url,
                image_api_url=config.deepseek_image_api_url,
                timeout=config.timeout
            )
            
            try:
                for element in refinement_tasks:
                    prov = element.prov[0]
                    page_no = prov.page_no - 1  # Docling uses 1-based, fitz uses 0-based
                    
                    if 0 <= page_no < len(pdf_doc):
                        page = pdf_doc[page_no]
                        # docling bbox: [l, t, r, b] - need to verify coordinate system
                        # docling usually uses points? fitz uses points.
                        bbox = prov.bbox
                        # Some versions might have [x1, y1, x2, y2]
                        # Fitz Rect: x0, y0, x1, y1
                        rect = fitz.Rect(bbox.l, bbox.t, bbox.r, bbox.b)
                        
                        # Add a small margin
                        rect.x0 = max(0, rect.x0 - 5)
                        rect.y0 = max(0, rect.y0 - 5)
                        rect.x1 = min(page.rect.width, rect.x1 + 5)
                        rect.y1 = min(page.rect.height, rect.y1 + 5)
                        
                        # Create image crop
                        pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2)) # 2x zoom for better OCR
                        img_bytes = pix.tobytes("png")
                        
                        # Send to DeepSeek
                        try:
                            refined_result = await deepseek_extractor.extract_text_from_image(img_bytes)
                            refined_text = refined_result.get("text", "")
                            
                            if refined_text:
                                if not hasattr(doc, "_refined_content"):
                                    doc._refined_content = {}
                                doc._refined_content[id(element)] = refined_text
                        except Exception as e:
                            logger.warning(f"Regional refinement failed for element {id(element)}: {e}")
            finally:
                await deepseek_extractor.close()
                pdf_doc.close()

        # 3. Apply Refined Content via Monkeypatching
        # We replace the export_to_markdown method of the specific elements
        # so that when doc.export_to_markdown() is called, it uses our refined text.
        if hasattr(doc, "_refined_content"):
            for element, _level in doc.iterate_items():
                elem_id = id(element)
                if elem_id in doc._refined_content:
                    refined_text = doc._refined_content[elem_id]
                    logger.info(f"Replacing element {elem_id} with refined text (length: {len(refined_text)})")
                    
                    # Store original method just in case, though doc is short-lived
                    # We use a closure to ensure we capture the correct refined_text
                    def make_refined_exporter(text):
                        def refined_exporter(*args, **kwargs):
                            return text
                        return refined_exporter
                    
                    element.export_to_markdown = make_refined_exporter(refined_text)

        markdown_content = doc.export_to_markdown()
        
        page_count = doc.num_pages if hasattr(doc, "num_pages") else 0
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=markdown_content,
            engine_used="hybrid",
            processing_time=processing_time,
            page_count=page_count,
            format="markdown",
        )
    except Exception as e:
        logger.error(f"Hybrid extraction failed within processing: {e}")
        raise
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
