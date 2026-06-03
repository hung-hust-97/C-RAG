import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Optional

import httpx

from lightrag.ocr.exceptions import (
    MinerUAPIConnectionError,
    MinerUOCRError,
    MinerUOCRTimeoutError,
)

logger = logging.getLogger(__name__)


def _bool_form_value(value: bool) -> str:
    return "true" if value else "false"


def _sanitize_response(data: dict[str, Any]) -> dict[str, Any]:
    """Sanitize response dictionary to avoid huge payloads in logs."""
    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            sanitized[key] = (
                f"<{len(value)} chars string...>" if len(value) > 200 else value
            )
        elif isinstance(value, list):
            sanitized[key] = f"<list of {len(value)} items>"
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_response(value)
        else:
            sanitized[key] = value
    return sanitized


def _parse_json_if_needed(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return value

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _count_pages_from_value(value: Any) -> int:
    """Infer page count from common MinerU/DeepSeek-style response fields."""
    value = _parse_json_if_needed(value)

    if isinstance(value, int):
        return max(value, 0)

    if isinstance(value, list):
        return len(value)

    if not isinstance(value, dict):
        return 0

    page_count = value.get("page_count")
    if isinstance(page_count, int) and page_count >= 0:
        return page_count

    pages = value.get("pages")
    if isinstance(pages, list):
        return len(pages)
    if isinstance(pages, int) and pages >= 0:
        return pages

    pdf_info = value.get("pdf_info")
    if isinstance(pdf_info, list):
        return len(pdf_info)

    middle_json = value.get("middle_json")
    if middle_json is not None:
        return _count_pages_from_value(middle_json)

    return 0


def _extract_text_from_result(result: dict[str, Any]) -> str:
    """Extract Markdown/text from a single MinerU result object."""
    for field in ("md_content", "markdown", "text"):
        value = result.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def normalize_mineru_response(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize MinerU JSON into LightRAG OCR extractor output.

    MinerU's FastAPI response usually stores Markdown at
    results[filename].md_content. This function also supports root-level
    fallback fields so tests and older/newer API variants remain compatible.
    """
    texts: list[str] = []
    root_page_count = _count_pages_from_value(data)
    page_count = 0

    results = data.get("results")
    if isinstance(results, dict):
        for result in results.values():
            if not isinstance(result, dict):
                continue
            text = _extract_text_from_result(result)
            if text.strip():
                texts.append(text)
            page_count += _count_pages_from_value(result)
    elif isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            text = _extract_text_from_result(result)
            if text.strip():
                texts.append(text)
            page_count += _count_pages_from_value(result)

    if not texts:
        root_text = _extract_text_from_result(data)
        if root_text.strip():
            texts.append(root_text)

    if page_count == 0:
        page_count = root_page_count

    return {
        "text": "\n\n".join(texts),
        "page_count": page_count,
    }


class MinerUOCRExtractor:
    """Client for MinerU /file_parse API."""

    def __init__(
        self,
        api_url: str,
        timeout: int = 1200,
        backend: str = "vlm-auto-engine",
        lang_list: str = "en,vn",
        parse_method: str = "auto",
        return_middle_json: bool = False,
        return_images: bool = False,
        formula_enable: bool = True,
        table_enable: bool = True,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        if not api_url:
            raise ValueError("api_url cannot be empty")
        if timeout <= 0:
            raise ValueError("timeout must be a positive integer")

        self.api_url = api_url
        self.timeout = timeout
        self.backend = backend
        self.lang_list = lang_list
        self.parse_method = parse_method
        self.return_middle_json = return_middle_json
        self.return_images = return_images
        self.formula_enable = formula_enable
        self.table_enable = table_enable

        if transport is None:
            transport = httpx.AsyncHTTPTransport(retries=3)

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
            "Initialized MinerU OCR client: api_url=%s, backend=%s, timeout=%ss",
            api_url,
            backend,
            timeout,
        )

    def _form_data(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "lang_list": self.lang_list,
            "parse_method": self.parse_method,
            "return_md": "true",
            "return_middle_json": _bool_form_value(self.return_middle_json),
            "return_images": _bool_form_value(self.return_images),
            "formula_enable": _bool_form_value(self.formula_enable),
            "table_enable": _bool_form_value(self.table_enable),
            "response_format_zip": "false",
        }

    async def extract_text_from_document(
        self,
        file_bytes: bytes,
        filename: str = "document.pdf",
    ) -> dict[str, Any]:
        """Extract text from a document using MinerU /file_parse."""
        if not file_bytes:
            raise ValueError("file_bytes cannot be empty")

        logger.info("Sending document to MinerU OCR API: %s", self.api_url)

        content_type = (
            mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        safe_filename = Path(filename).name or "document.pdf"
        files = {
            "files": (safe_filename, file_bytes, content_type),
        }

        try:
            response = await self.client.post(
                self.api_url,
                data=self._form_data(),
                files=files,
            )

            if response.status_code >= 400:
                try:
                    response_text = response.text
                except Exception:
                    response_text = "Could not read response text"

                error_type = "server" if response.status_code >= 500 else "client"
                error_msg = (
                    f"MinerU OCR API {error_type} error: "
                    f"{response.status_code}, Response: {response_text}"
                )
                logger.error(error_msg)
                raise MinerUOCRError(error_msg)

            response_data = response.json()
            result = normalize_mineru_response(response_data)

            text = result["text"]
            page_count = result["page_count"]
            if not text.strip():
                logger.warning(
                    "MinerU OCR returned empty result despite HTTP 200. "
                    "Response data: %s",
                    _sanitize_response(response_data),
                )
            else:
                logger.info(
                    "MinerU OCR extraction successful: %s pages, %s characters",
                    page_count,
                    len(text),
                )

            return result

        except httpx.TimeoutException as e:
            error_msg = f"MinerU OCR API request timeout after {self.timeout}s"
            logger.error(error_msg)
            raise MinerUOCRTimeoutError(error_msg) from e

        except (httpx.ConnectError, httpx.NetworkError) as e:
            error_msg = f"Failed to connect to MinerU OCR API: {self.api_url}"
            logger.error(error_msg)
            raise MinerUAPIConnectionError(error_msg) from e

        except MinerUOCRError:
            raise

        except Exception as e:
            error_msg = f"MinerU OCR API call failed: {str(e)}"
            logger.error(error_msg)
            raise MinerUOCRError(error_msg) from e

    async def extract_text_from_pdf(self, file_bytes: bytes) -> dict[str, Any]:
        return await self.extract_text_from_document(file_bytes, "document.pdf")

    async def extract_text_from_image(self, image_bytes: bytes) -> dict[str, Any]:
        return await self.extract_text_from_document(image_bytes, "image.png")

    async def is_available(self) -> bool:
        """Check whether the MinerU endpoint is reachable."""
        try:
            response = await self.client.get(self.api_url, timeout=5.0)
            is_available = response.status_code < 500
            if is_available:
                logger.debug("MinerU OCR API is available: %s", self.api_url)
            else:
                logger.warning(
                    "MinerU OCR API returned server error: %s",
                    response.status_code,
                )
            return is_available
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            logger.warning("MinerU OCR API is not available: %s", str(e))
            return False
        except Exception as e:
            logger.warning("Error checking MinerU OCR API availability: %s", str(e))
            return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self.client.aclose()
        logger.debug("MinerU OCR client closed")
