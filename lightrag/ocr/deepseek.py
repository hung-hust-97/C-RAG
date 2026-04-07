"""
DeepSeek OCR API client for LightRAG.

This module provides an HTTP client for the DeepSeek OCR API, enabling
extraction of text from PDF documents using the DeepSeek OCR service.
"""

import logging
import httpx
from typing import Dict

from lightrag.ocr.exceptions import (
    DeepSeekOCRError,
    OCRTimeoutError,
    DeepSeekAPIConnectionError,
)

logger = logging.getLogger(__name__)


class DeepSeekOCRExtractor:
    """Client for DeepSeek OCR API.
    
    This class manages HTTP communication with the DeepSeek OCR API endpoint,
    providing text extraction from PDF documents with automatic retry logic
    and connection pooling for optimal performance.
    
    Attributes:
        api_url: DeepSeek OCR API endpoint URL
        timeout: Request timeout in seconds
        client: Async HTTP client with connection pooling
    """
    
    def __init__(self, api_url: str, timeout: int = 300):
        """Initialize DeepSeek OCR API client.
        
        Args:
            api_url: DeepSeek OCR API endpoint (loaded from DEEPSEEK_API_URL env var)
            timeout: Request timeout in seconds (loaded from DEEPSEEK_OCR_TIMEOUT env var)
            
        Raises:
            ValueError: If api_url is empty or timeout is not positive
        """
        if not api_url:
            raise ValueError("api_url cannot be empty")
        if timeout <= 0:
            raise ValueError("timeout must be a positive integer")
        
        self.api_url = api_url
        self.timeout = timeout
        
        # Configure retry logic with exponential backoff (max 3 retries)
        # Retry on network errors and 5xx status codes
        transport = httpx.AsyncHTTPTransport(
            retries=3,  # Maximum 3 retry attempts
        )
        
        # Create httpx.AsyncClient with connection pooling and keep-alive
        self.client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
        )
        
        logger.info(
            f"Initialized DeepSeek OCR client: api_url={api_url}, timeout={timeout}s"
        )
    
    async def extract_text_from_pdf(self, file_bytes: bytes) -> Dict[str, any]:
        """Extract text from PDF using DeepSeek OCR API.
        
        Sends a PDF file to the DeepSeek OCR API endpoint via multipart/form-data
        POST request and returns the extracted text and page count.
        
        Args:
            file_bytes: PDF file content as bytes
            
        Returns:
            dict: Response containing 'text' (str) and 'page_count' (int)
            
        Raises:
            DeepSeekOCRError: If OCR processing fails
            OCRTimeoutError: If request exceeds timeout
            DeepSeekAPIConnectionError: If API endpoint is unreachable
        """
        if not file_bytes:
            raise ValueError("file_bytes cannot be empty")
        
        logger.info(f"Sending PDF to DeepSeek OCR API: {self.api_url}")
        
        try:
            # Send multipart/form-data POST request to /ocr/pdf endpoint
            files = {"file": ("document.pdf", file_bytes, "application/pdf")}
            
            response = await self.client.post(
                self.api_url,
                files=files,
            )
            
            # Check for error status codes
            if response.status_code >= 500:
                error_msg = f"DeepSeek OCR API server error: {response.status_code}"
                logger.error(error_msg)
                raise DeepSeekOCRError(error_msg)
            
            if response.status_code >= 400:
                error_msg = f"DeepSeek OCR API client error: {response.status_code}"
                logger.error(error_msg)
                raise DeepSeekOCRError(error_msg)
            
            # Parse response to extract text and page_count
            response_data = response.json()
            
            text = response_data.get("text", "")
            page_count = response_data.get("page_count", 0)
            
            logger.info(
                f"DeepSeek OCR extraction successful: {page_count} pages, "
                f"{len(text)} characters"
            )
            
            return {
                "text": text,
                "page_count": page_count,
            }
            
        except httpx.TimeoutException as e:
            error_msg = f"DeepSeek OCR API request timeout after {self.timeout}s"
            logger.error(error_msg)
            raise OCRTimeoutError(error_msg) from e
        
        except (httpx.ConnectError, httpx.NetworkError) as e:
            error_msg = f"Failed to connect to DeepSeek OCR API: {self.api_url}"
            logger.error(error_msg)
            raise DeepSeekAPIConnectionError(error_msg) from e
        
        except Exception as e:
            error_msg = f"DeepSeek OCR API call failed: {str(e)}"
            logger.error(error_msg)
            raise DeepSeekOCRError(error_msg) from e
    
    async def is_available(self) -> bool:
        """Check if DeepSeek OCR API is available.
        
        Attempts to connect to the API endpoint to verify availability.
        Uses a short timeout to avoid blocking.
        
        Returns:
            bool: True if API endpoint is reachable, False otherwise
        """
        try:
            # Use a short timeout for availability check (5 seconds)
            response = await self.client.get(
                self.api_url,
                timeout=5.0,
            )
            
            # Consider 2xx and 4xx as "available" (endpoint exists)
            # Only 5xx or connection errors indicate unavailability
            is_available = response.status_code < 500
            
            if is_available:
                logger.debug(f"DeepSeek OCR API is available: {self.api_url}")
            else:
                logger.warning(
                    f"DeepSeek OCR API returned server error: {response.status_code}"
                )
            
            return is_available
            
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            logger.warning(f"DeepSeek OCR API is not available: {str(e)}")
            return False
        
        except Exception as e:
            logger.warning(f"Error checking DeepSeek OCR API availability: {str(e)}")
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close the HTTP client."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client and release resources."""
        await self.client.aclose()
        logger.debug("DeepSeek OCR client closed")
