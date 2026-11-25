import asyncio
import hashlib
import io
import json
import logging
import re
import time
import zipfile
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Optional
from urllib.parse import quote, urlparse, urlunparse

import httpx

from lightrag.ocr.exceptions import (
    MinerUAPIConnectionError,
    MinerUOCRError,
    MinerUOCRTimeoutError,
)

logger = logging.getLogger(__name__)

VALID_MINERU_TIERS = {"", "flash", "basic", "standard", "advanced"}
VALID_MINERU_OCR_MODES = {"auto", "txt", "ocr"}
_MAX_ZIP_ENTRY_BYTES = 128 * 1024 * 1024  # 128 MiB guard against zip bombs
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_LEGACY_URL_SUFFIXES = (
    "/v1/file_parse",
    "/file_parse",
    "/v1/tasks",
    "/v1/health",
    "/v1",
)


def normalize_mineru_base_url(api_url: str) -> str:
    """Normalize MinerU endpoint URL to its base origin/prefix without /v1 or /file_parse."""
    if not api_url or not api_url.strip():
        raise ValueError("api_url cannot be empty")
    cleaned = api_url.strip().rstrip("/")
    for suffix in _LEGACY_URL_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].rstrip("/")
            break
    return cleaned


def sanitize_mineru_filename(filename: str) -> str:
    """Sanitize file_name for MinerU v4.2.3 (/v1/tasks forbids path separators, ':', control chars)."""
    raw = (filename or "").strip()
    name = PurePosixPath(PureWindowsPath(raw).name).name
    name = re.sub(r"[/\\:\x00-\x1f\x7f]+", "_", name).strip()
    if not name or name in {".", ".."}:
        return "document.pdf"
    if len(name) > 255:
        dot_idx = name.rfind(".")
        if 0 < dot_idx < len(name) - 1 and len(name) - dot_idx <= 16:
            ext = name[dot_idx:]
            name = name[: 255 - len(ext)] + ext
        else:
            name = name[:255]
    return name


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
        if value and all(isinstance(item, dict) for item in value):
            page_indices = [
                item.get("page_idx")
                for item in value
                if isinstance(item.get("page_idx"), int) and item.get("page_idx") >= 0
            ]
            if page_indices:
                return max(page_indices) + 1
            page_numbers = [
                item.get("page_number") or item.get("page_no")
                for item in value
                if isinstance(item.get("page_number") or item.get("page_no"), int)
            ]
            if page_numbers:
                return max(page_numbers)
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
    """Normalize legacy MinerU JSON into LightRAG OCR extractor output."""
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


def _is_safe_zip_member(member_name: str) -> bool:
    """Validate ZIP member paths against directory traversal."""
    if not member_name or member_name.startswith(("/", "\\")):
        return False
    win_parts = PureWindowsPath(member_name).parts
    posix_parts = PurePosixPath(member_name).parts
    if ".." in win_parts or ".." in posix_parts:
        return False
    if win_parts and ":" in win_parts[0]:
        return False
    return True


def _read_bounded_zip_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """Read a ZIP entry strictly in memory with path and size validation."""
    if not _is_safe_zip_member(info.filename):
        raise MinerUOCRError(f"Unsafe entry path in MinerU result ZIP: {info.filename!r}")
    if info.file_size > _MAX_ZIP_ENTRY_BYTES:
        raise MinerUOCRError(
            f"MinerU ZIP entry {info.filename!r} exceeds max allowed size ({info.file_size} bytes)"
        )
    chunks: list[bytes] = []
    total = 0
    with zf.open(info, "r") as fp:
        while True:
            chunk = fp.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_ZIP_ENTRY_BYTES:
                raise MinerUOCRError(
                    f"MinerU ZIP entry {info.filename!r} decompressed size exceeds limit"
                )
            chunks.append(chunk)
    return b"".join(chunks)


def extract_mineru_zip_bytes(zip_bytes: bytes) -> dict[str, Any]:
    """Extract markdown.md and page count in-memory from a MinerU v4.2.3 result ZIP."""
    if not zip_bytes:
        raise MinerUOCRError("MinerU result archive is empty")

    try:
        zf_ctx = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    except zipfile.BadZipFile as exc:
        raise MinerUOCRError(f"Invalid ZIP archive from MinerU result_url: {exc}") from exc

    with zf_ctx as zf:
        infos = zf.infolist()
        for info in infos:
            if not _is_safe_zip_member(info.filename):
                raise MinerUOCRError(
                    f"Unsafe entry path in MinerU result ZIP: {info.filename!r}"
                )

        file_infos = [i for i in infos if not i.is_dir()]
        by_exact = {i.filename: i for i in file_infos}
        by_basename: dict[str, zipfile.ZipInfo] = {}
        for i in file_infos:
            base = PurePosixPath(i.filename).name
            by_basename.setdefault(base, i)

        md_info = by_exact.get("markdown.md") or by_basename.get("markdown.md")
        if md_info is None:
            for i in file_infos:
                if i.filename.lower().endswith(".md"):
                    md_info = i
                    break

        markdown_text = ""
        if md_info is not None:
            markdown_text = _read_bounded_zip_entry(zf, md_info).decode(
                "utf-8", errors="replace"
            )

        page_count = 0
        for meta_name in ("middle_json.json", "structured_content.json"):
            meta_info = by_exact.get(meta_name) or by_basename.get(meta_name)
            if meta_info is None:
                continue
            try:
                meta_raw = _read_bounded_zip_entry(zf, meta_info).decode(
                    "utf-8", errors="replace"
                )
                meta_json = json.loads(meta_raw)
                page_count = _count_pages_from_value(meta_json)
                if page_count > 0:
                    break
            except Exception as exc:
                logger.debug("Could not infer page_count from %s: %s", meta_name, exc)

        if page_count == 0 and markdown_text.strip():
            page_count = 1

        return {
            "text": markdown_text,
            "page_count": page_count,
            "files_in_zip": [i.filename for i in file_infos],
        }


def _extract_error_details(payload: Any) -> tuple[Optional[str], str, Optional[str]]:
    """Extract (code, message, param) from MinerU v4.2.3 error payloads."""
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            code = err.get("code")
            msg = err.get("message") or json.dumps(err, ensure_ascii=False)
            param = err.get("param")
            return (
                str(code) if code is not None else None,
                str(msg),
                str(param) if param is not None else None,
            )
        if "code" in payload or "message" in payload:
            code = payload.get("code")
            msg = payload.get("message") or json.dumps(payload, ensure_ascii=False)
            param = payload.get("param")
            return (
                str(code) if code is not None else None,
                str(msg),
                str(param) if param is not None else None,
            )
        return None, json.dumps(payload, ensure_ascii=False), None
    return None, str(payload), None


def _extract_http_error(response: httpx.Response) -> tuple[Optional[str], str, Optional[str]]:
    try:
        data = response.json()
        return _extract_error_details(data)
    except Exception:
        try:
            return None, response.text, None
        except Exception:
            return None, f"HTTP {response.status_code}", None


def _is_tier_engine_unavailable(code: Optional[str], message: str) -> bool:
    """Detect server-side VLM/tier runtime unavailability eligible for basic-tier fallback."""
    if code in {"quality_tier_unavailable", "engine_unavailable"}:
        return True
    lowered = (message or "").lower()
    return "cannot start vllm" in lowered or ("vllm" in lowered and "incompatible" in lowered)


class MinerUOCRExtractor:
    """Async client for MinerU v4.2.3 /v1/tasks OCR API."""

    def __init__(
        self,
        api_url: str,
        timeout: int = 1200,
        tier: Optional[str] = "standard",
        ocr_mode: Optional[str] = None,
        api_key: Optional[str] = None,
        poll_interval: float = 2.0,
        page_range: str = "",
        output_formats: Optional[list[str]] = None,
        # Deprecated v3.x kwargs accepted for caller compatibility (never sent on wire):
        backend: Optional[str] = None,
        lang_list: Optional[str] = None,
        parse_method: Optional[str] = None,
        return_middle_json: bool = False,
        return_images: bool = False,
        formula_enable: bool = True,
        table_enable: bool = True,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.base_url = normalize_mineru_base_url(api_url)
        self.api_url = self.base_url
        if timeout <= 0:
            raise ValueError("timeout must be a positive integer")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        resolved_tier = (tier or "").strip().lower()
        if resolved_tier not in VALID_MINERU_TIERS:
            raise ValueError(
                f"Invalid MinerU tier {tier!r}. Must be one of: flash, basic, standard, advanced, or empty"
            )

        resolved_ocr_mode = (ocr_mode or parse_method or "auto").strip().lower()
        if resolved_ocr_mode not in VALID_MINERU_OCR_MODES:
            raise ValueError(
                f"Invalid MinerU ocr_mode {resolved_ocr_mode!r}. Must be one of: auto, txt, ocr"
            )

        self.timeout = timeout
        self.tier = resolved_tier
        self.ocr_mode = resolved_ocr_mode
        self.api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.poll_interval = float(poll_interval)
        self.page_range = (page_range or "").strip()
        self.output_formats = list(output_formats) if output_formats else ["default"]

        # Keep deprecated attributes readable if inspected by legacy code
        self.backend = backend or "vlm-auto-engine"
        self.lang_list = lang_list or "en,vn"
        self.parse_method = self.ocr_mode
        self.return_middle_json = return_middle_json
        self.return_images = return_images
        self.formula_enable = formula_enable
        self.table_enable = table_enable

        # Do not attach automatic transport retries to avoid duplicate POST /v1/tasks jobs on timeout
        self.client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(min(float(timeout), 300.0)),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
        )

        logger.info(
            "Initialized MinerU v4 async OCR client: base_url=%s, tier=%s, ocr_mode=%s, timeout=%ss",
            self.base_url,
            self.tier or "auto",
            self.ocr_mode,
            self.timeout,
        )

    def _resolve_server_url(self, target_url: str) -> str:
        """Rewrite loopback-advertised /v1/... URLs when base_url targets a remote MinerU server."""
        if not target_url:
            return target_url
        base_parsed = urlparse(self.base_url)
        target_parsed = urlparse(target_url)
        base_host = (base_parsed.hostname or "").lower()
        target_host = (target_parsed.hostname or "").lower()
        if (
            base_host
            and base_host not in _LOOPBACK_HOSTS
            and target_host in _LOOPBACK_HOSTS
            and target_parsed.path.startswith("/v1/")
        ):
            base_prefix = base_parsed.path.rstrip("/")
            rewritten_path = f"{base_prefix}{target_parsed.path}" if base_prefix else target_parsed.path
            return urlunparse(
                (
                    base_parsed.scheme,
                    base_parsed.netloc,
                    rewritten_path,
                    target_parsed.params,
                    target_parsed.query,
                    target_parsed.fragment,
                )
            )
        return target_url

    def _is_same_origin(self, target_url: str) -> bool:
        base_parsed = urlparse(self.base_url)
        target_parsed = urlparse(target_url)
        if not target_parsed.scheme and not target_parsed.netloc:
            return True
        return (
            target_parsed.scheme.lower() == base_parsed.scheme.lower()
            and target_parsed.netloc.lower() == base_parsed.netloc.lower()
        )

    def _headers_for_url(
        self,
        target_url: str,
        extra: Optional[dict[str, str]] = None,
    ) -> dict[str, str]:
        """Build headers for target_url, attaching Bearer token strictly for same-origin URLs."""
        headers: dict[str, str] = dict(extra or {})
        if self.api_key and self._is_same_origin(target_url):
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_task_payload(
        self,
        safe_filename: str,
        file_size: int,
        sha256sum: str,
        tier_override: Optional[str] = None,
    ) -> dict[str, Any]:
        file_manifest: dict[str, Any] = {
            "file_name": safe_filename,
            "bytes": file_size,
            "sha256sum": sha256sum,
        }
        if self.page_range:
            file_manifest["page_range"] = self.page_range

        effective_tier = self.tier if tier_override is None else tier_override
        payload: dict[str, Any] = {
            "files": [file_manifest],
            "ocr_mode": self.ocr_mode,
            "output_formats": self.output_formats,
        }
        if effective_tier:
            payload["tier"] = effective_tier
        if self.page_range:
            payload["page_range"] = self.page_range
        return payload

    async def _cancel_task_safely(self, task_id: str) -> None:
        """Best-effort cancellation for timed-out or cancelled tasks."""
        if not task_id:
            return
        cancel_url = f"{self.base_url}/v1/tasks/{quote(task_id, safe='')}/cancel"
        try:
            resp = await self.client.post(
                cancel_url,
                headers=self._headers_for_url(cancel_url),
                timeout=5.0,
            )
            if resp.status_code not in (200, 404, 409):
                logger.debug(
                    "MinerU cancel task %s returned status %s",
                    task_id,
                    resp.status_code,
                )
        except Exception as exc:
            logger.debug("Failed to cancel MinerU task %s: %s", task_id, exc)

    async def _create_task_with_503_backoff(
        self,
        payload: dict[str, Any],
        deadline: float,
    ) -> dict[str, Any]:
        tasks_url = f"{self.base_url}/v1/tasks"
        headers = self._headers_for_url(tasks_url, {"Content-Type": "application/json"})

        for retry_idx in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MinerUOCRTimeoutError(
                    f"MinerU OCR timed out before creating task ({self.timeout}s)"
                )

            response = await self.client.post(
                tasks_url,
                headers=headers,
                json=payload,
                timeout=min(60.0, max(5.0, remaining)),
            )

            if response.status_code == 503 and retry_idx < 2:
                code, msg, _ = _extract_http_error(response)
                backoff = min(2.0 * (retry_idx + 1), max(0.1, deadline - time.monotonic()))
                logger.warning(
                    "MinerU POST /v1/tasks returned 503 (code=%s, msg=%s); retrying in %.1fs",
                    code,
                    msg,
                    backoff,
                )
                await asyncio.sleep(backoff)
                continue

            if response.status_code >= 400:
                code, msg, param = _extract_http_error(response)
                param_suffix = f" (param={param})" if param else ""
                error_msg = (
                    f"MinerU POST /v1/tasks error {response.status_code} "
                    f"[{code or 'http_error'}]{param_suffix}: {msg}"
                )
                logger.error(error_msg)
                raise MinerUOCRError(error_msg)

            data = response.json()
            if not isinstance(data, dict) or not data.get("task_id") or not data.get("files"):
                raise MinerUOCRError(
                    f"Unexpected MinerU create-task response: {_sanitize_response(data if isinstance(data, dict) else {})}"
                )
            return data

        raise MinerUOCRError("MinerU POST /v1/tasks failed after 503 retries")

    async def extract_text_from_document(
        self,
        file_bytes: bytes,
        filename: str = "document.pdf",
    ) -> dict[str, Any]:
        """Extract Markdown text from a document using MinerU v4.2.3 async /v1/tasks."""
        if not file_bytes:
            raise ValueError("file_bytes cannot be empty")

        safe_filename = sanitize_mineru_filename(filename)
        file_size = len(file_bytes)
        sha256sum = hashlib.sha256(file_bytes).hexdigest()
        effective_tier = self.tier
        deadline = time.monotonic() + self.timeout

        logger.info(
            "Submitting document to MinerU v4 /v1/tasks: base_url=%s, file=%s, bytes=%s, tier=%s, ocr_mode=%s",
            self.base_url,
            safe_filename,
            file_size,
            effective_tier or "auto",
            self.ocr_mode,
        )

        active_task_id: Optional[str] = None
        task_terminal = False

        try:
            for attempt in range(3):
                active_task_id = None
                task_terminal = False
                payload = self._build_task_payload(
                    safe_filename, file_size, sha256sum, tier_override=effective_tier
                )

                # Step 1: Create async task manifest
                try:
                    created = await self._create_task_with_503_backoff(payload, deadline)
                except MinerUOCRError as create_err:
                    if effective_tier in ("standard", "advanced") and _is_tier_engine_unavailable(
                        None, str(create_err)
                    ):
                        logger.warning(
                            "MinerU tier '%s' unavailable during create-task (%s); falling back to tier='basic'",
                            effective_tier,
                            create_err,
                        )
                        effective_tier = "basic"
                        continue
                    raise
                task_id = str(created["task_id"])
                active_task_id = task_id
                file_info = created["files"][0]
                file_id = str(file_info["file_id"])

                # Step 2: Upload raw binary if server did not hit SHA-256 cache
                upload_info = file_info.get("upload")
                if isinstance(upload_info, dict) and upload_info.get("url"):
                    upload_url = self._resolve_server_url(str(upload_info["url"]))
                    raw_upload_headers = upload_info.get("headers")
                    base_upload_headers = (
                        {str(k): str(v) for k, v in raw_upload_headers.items()}
                        if isinstance(raw_upload_headers, dict) and raw_upload_headers
                        else {"Content-Type": "application/octet-stream"}
                    )
                    upload_headers = self._headers_for_url(upload_url, base_upload_headers)

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        await self._cancel_task_safely(task_id)
                        task_terminal = True
                        raise MinerUOCRTimeoutError(
                            f"MinerU OCR task {task_id} timed out before upload ({self.timeout}s)"
                        )

                    put_resp = await self.client.put(
                        upload_url,
                        headers=upload_headers,
                        content=file_bytes,
                        timeout=min(300.0, max(5.0, remaining)),
                    )

                    if put_resp.status_code in (404, 410) and attempt == 0:
                        code, msg, _ = _extract_http_error(put_resp)
                        logger.warning(
                            "MinerU upload URL expired/missing for task %s (%s: %s); recreating task once",
                            task_id,
                            code,
                            msg,
                        )
                        continue

                    if put_resp.status_code >= 400:
                        code, msg, param = _extract_http_error(put_resp)
                        param_suffix = f" (param={param})" if param else ""
                        error_msg = (
                            f"MinerU PUT upload error {put_resp.status_code} "
                            f"[{code or 'upload_failed'}]{param_suffix}: {msg}"
                        )
                        logger.error(error_msg)
                        await self._cancel_task_safely(task_id)
                        task_terminal = True
                        raise MinerUOCRError(error_msg)
                else:
                    logger.info(
                        "MinerU SHA-256 cache hit for %s (task_id=%s, file_id=%s); skipping PUT upload",
                        safe_filename,
                        task_id,
                        file_id,
                    )

                # Step 3: Poll task status until file reaches terminal state
                status_url = f"{self.base_url}/v1/tasks/{quote(task_id, safe='')}"
                status_headers = self._headers_for_url(status_url)
                result_url: Optional[str] = None
                recreate_needed = False

                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        await self._cancel_task_safely(task_id)
                        task_terminal = True
                        error_msg = (
                            f"MinerU OCR task {task_id} timed out after {self.timeout}s"
                        )
                        logger.error(error_msg)
                        raise MinerUOCRTimeoutError(error_msg)

                    poll_resp = await self.client.get(
                        status_url,
                        headers=status_headers,
                        timeout=min(30.0, max(5.0, remaining)),
                    )

                    if poll_resp.status_code == 404 and attempt == 0:
                        logger.warning(
                            "MinerU task %s returned 404 task_not_found (in-memory reset suspected); recreating once",
                            task_id,
                        )
                        recreate_needed = True
                        break

                    if poll_resp.status_code == 503:
                        sleep_for = min(self.poll_interval, max(0.1, deadline - time.monotonic()))
                        await asyncio.sleep(sleep_for)
                        continue

                    if poll_resp.status_code >= 400:
                        code, msg, param = _extract_http_error(poll_resp)
                        param_suffix = f" (param={param})" if param else ""
                        error_msg = (
                            f"MinerU GET /v1/tasks/{task_id} error {poll_resp.status_code} "
                            f"[{code or 'poll_failed'}]{param_suffix}: {msg}"
                        )
                        logger.error(error_msg)
                        raise MinerUOCRError(error_msg)

                    detail = poll_resp.json()
                    files_detail = detail.get("files") if isinstance(detail, dict) else None
                    file_entry: dict[str, Any] = {}
                    if isinstance(files_detail, list) and files_detail:
                        file_entry = next(
                            (
                                f
                                for f in files_detail
                                if isinstance(f, dict) and f.get("file_id") == file_id
                            ),
                            files_detail[0] if isinstance(files_detail[0], dict) else {},
                        )

                    file_status = str(file_entry.get("status") or "")
                    task_status = str(detail.get("status") or "") if isinstance(detail, dict) else ""

                    if file_status == "completed":
                        task_terminal = True
                        raw_result_url = file_entry.get("result_url") or (
                            f"{self.base_url}/v1/files/{quote(file_id, safe='')}/content"
                        )
                        result_url = self._resolve_server_url(str(raw_result_url))
                        break

                    if (
                        file_status in ("failed", "canceled")
                        or task_status in ("failed", "canceled")
                        or (task_status == "partial" and file_status not in ("waiting-file", "pending", "running", "completed"))
                    ):
                        task_terminal = True
                        err_obj = file_entry.get("error") or (
                            detail.get("error") if isinstance(detail, dict) else None
                        )
                        code, msg, param = _extract_error_details(
                            {"error": err_obj} if isinstance(err_obj, dict) else err_obj
                        )
                        if effective_tier in ("standard", "advanced") and _is_tier_engine_unavailable(
                            code, msg
                        ):
                            logger.warning(
                                "MinerU task %s failed on tier='%s' (%s); falling back to tier='basic'",
                                task_id,
                                effective_tier,
                                msg,
                            )
                            effective_tier = "basic"
                            recreate_needed = True
                            break
                        param_suffix = f" (param={param})" if param else ""
                        error_msg = (
                            f"MinerU OCR task {task_id} {file_status or task_status} "
                            f"[{code or 'parse_failed'}]{param_suffix}: {msg}"
                        )
                        logger.error(error_msg)
                        if code == "task_timeout":
                            raise MinerUOCRTimeoutError(error_msg)
                        raise MinerUOCRError(error_msg)

                    sleep_for = min(self.poll_interval, max(0.05, deadline - time.monotonic()))
                    await asyncio.sleep(sleep_for)

                if recreate_needed:
                    continue

                if not result_url:
                    raise MinerUOCRError(f"MinerU task {task_id} completed without result_url")

                # Step 4: Download result archive & extract markdown.md + page_count
                remaining = deadline - time.monotonic()
                dl_resp = await self.client.get(
                    result_url,
                    headers=self._headers_for_url(result_url),
                    timeout=min(300.0, max(10.0, remaining)),
                )

                if dl_resp.status_code == 404 and attempt == 0:
                    logger.warning(
                        "MinerU result_url returned 404 for task %s; recreating task once",
                        task_id,
                    )
                    continue

                if dl_resp.status_code >= 400:
                    code, msg, param = _extract_http_error(dl_resp)
                    param_suffix = f" (param={param})" if param else ""
                    error_msg = (
                        f"MinerU download result error {dl_resp.status_code} "
                        f"[{code or 'download_failed'}]{param_suffix}: {msg}"
                    )
                    logger.error(error_msg)
                    raise MinerUOCRError(error_msg)

                content_type = (dl_resp.headers.get("content-type") or "").lower()
                if "text/html" in content_type and not zipfile.is_zipfile(io.BytesIO(dl_resp.content)):
                    html_text = dl_resp.content.decode("utf-8", errors="replace")
                    result = {"text": html_text, "page_count": 1 if html_text.strip() else 0}
                else:
                    zip_result = extract_mineru_zip_bytes(dl_resp.content)
                    result = {
                        "text": zip_result["text"],
                        "page_count": zip_result["page_count"],
                    }

                text = result["text"]
                page_count = result["page_count"]
                if not text.strip():
                    logger.warning(
                        "MinerU OCR task %s completed with empty markdown content for %s",
                        task_id,
                        safe_filename,
                    )
                else:
                    logger.info(
                        "MinerU OCR extraction successful: task_id=%s, file=%s, pages=%s, chars=%s",
                        task_id,
                        safe_filename,
                        page_count,
                        len(text),
                    )
                return result

            raise MinerUOCRError("MinerU OCR failed after recreating lost task")

        except asyncio.CancelledError:
            if active_task_id and not task_terminal:
                with io.BytesIO():
                    try:
                        await asyncio.shield(self._cancel_task_safely(active_task_id))
                    except Exception:
                        pass
            raise

        except MinerUOCRError:
            raise

        except httpx.TimeoutException as exc:
            if active_task_id and not task_terminal:
                await self._cancel_task_safely(active_task_id)
            error_msg = f"MinerU OCR API request timeout (limit={self.timeout}s): {exc}"
            logger.error(error_msg)
            raise MinerUOCRTimeoutError(error_msg) from exc

        except (httpx.ConnectError, httpx.NetworkError) as exc:
            error_msg = f"Failed to connect to MinerU OCR API ({self.base_url}): {exc}"
            logger.error(error_msg)
            raise MinerUAPIConnectionError(error_msg) from exc

        except Exception as exc:
            if active_task_id and not task_terminal:
                await self._cancel_task_safely(active_task_id)
            error_msg = f"MinerU OCR API call failed: {exc}"
            logger.error(error_msg)
            raise MinerUOCRError(error_msg) from exc

    async def extract_text_from_pdf(self, file_bytes: bytes) -> dict[str, Any]:
        return await self.extract_text_from_document(file_bytes, "document.pdf")

    async def extract_text_from_image(self, image_bytes: bytes) -> dict[str, Any]:
        return await self.extract_text_from_document(image_bytes, "image.png")

    async def is_available(self) -> bool:
        """Check whether the MinerU v4 /v1/health endpoint reports status ok."""
        health_url = f"{self.base_url}/v1/health"
        try:
            response = await self.client.get(health_url, timeout=5.0)
            if response.status_code != 200:
                logger.warning(
                    "MinerU /v1/health returned HTTP %s at %s",
                    response.status_code,
                    health_url,
                )
                return False
            try:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("status") != "ok":
                    logger.warning(
                        "MinerU /v1/health reported non-ok status: %s",
                        _sanitize_response(payload),
                    )
                    return False
            except Exception:
                pass
            logger.debug("MinerU OCR API is healthy: %s", health_url)
            return True
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            logger.warning("MinerU OCR API is not available (%s): %s", health_url, exc)
            return False
        except Exception as exc:
            logger.warning("Error checking MinerU OCR availability (%s): %s", health_url, exc)
            return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self.client.aclose()
        logger.debug("MinerU OCR client closed")
