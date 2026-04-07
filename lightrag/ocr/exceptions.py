"""OCR-specific exception classes for the DeepSeek OCR integration.

This module defines custom exceptions for OCR processing errors, providing
granular error handling for different failure scenarios in the OCR pipeline.
"""


class DeepSeekOCRError(Exception):
    """Base exception for DeepSeek OCR processing failures.
    
    Raised when DeepSeek OCR processing fails for any reason, including:
    - API errors (non-2xx status codes)
    - Invalid response format
    - OCR processing failures
    - Configuration errors specific to DeepSeek OCR
    
    This exception is raised when fallback is disabled or as a base class
    for more specific DeepSeek OCR errors.
    """
    pass


class OCRTimeoutError(DeepSeekOCRError):
    """Exception raised when OCR processing exceeds the configured timeout.
    
    Raised when:
    - DeepSeek OCR API request exceeds DEEPSEEK_OCR_TIMEOUT
    - Processing time approaches timeout limit and operation is cancelled
    
    The system may attempt fallback to Tesseract OCR if enabled.
    """
    pass


class PDFProcessingError(Exception):
    """Exception raised when all PDF extraction methods fail.
    
    Raised when:
    - All methods in the fallback chain fail (pypdf -> docling -> deepseek -> tesseract)
    - PDF file is corrupted or invalid
    - No extraction method is available or configured
    
    This is a terminal error indicating the PDF cannot be processed.
    """
    pass


class DeepSeekAPIConnectionError(DeepSeekOCRError):
    """Exception raised when DeepSeek OCR API endpoint is unreachable.
    
    Raised when:
    - Network connection to API endpoint fails
    - API endpoint is unreachable or not responding
    - DNS resolution fails for API URL
    - Connection timeout occurs before request is sent
    
    The system may attempt fallback to Tesseract OCR if enabled.
    """
    pass
