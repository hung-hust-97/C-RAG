import os
import copy
import tempfile
import asyncio
from pathlib import Path
from io import BytesIO
from functools import lru_cache

from lightrag.ocr import OCRConfig, process_document_with_ocr
from lightrag.ocr.markdown import MarkdownPreprocessor
from lightrag.utils import logger
from lightrag.api.config import global_args




@lru_cache(maxsize=1)
def _is_docling_available() -> bool:
    """Kiểm tra xem thư viện docling có khả dụng hay không (có cache kết quả)."""
    try:
        import docling  # noqa: F401  # type: ignore[import-not-found]

        return True
    except ImportError:
        return False


def _convert_with_docling(file_path: Path) -> str:
    """Chuyển đổi tài liệu bằng docling"""
    from docling.document_converter import DocumentConverter  # type: ignore

    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


def _run_docling_from_bytes_sync(file_bytes: bytes, ext: str) -> str:
    """Xử lý docling từ bytes"""
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        return _convert_with_docling(tmp_path)
    finally:
        if tmp_path.exists():
            os.unlink(tmp_path)


async def _run_ocr_pipeline(file_bytes: bytes, ext: str, force_engine: str = None):
    """Xử lý OCR từ bytes"""
    ocr_config = OCRConfig.from_global_args(global_args)
    if force_engine:
        ocr_config = copy.copy(ocr_config)
        ocr_config.engine = force_engine

    password = getattr(global_args, "pdf_decrypt_password", None)
    result = await process_document_with_ocr(
        file_bytes=file_bytes,
        password=password,
        ocr_config=ocr_config,
        file_extension=ext,
    )
    return result, ocr_config


def _extract_docx(file_bytes: bytes) -> str:
    """Trích xuất nội dung DOCX bao gồm cả bảng theo đúng thứ tự tài liệu"""
    from docx import Document  # type: ignore
    from docx.table import Table  # type: ignore
    from docx.text.paragraph import Paragraph  # type: ignore

    doc = Document(BytesIO(file_bytes))

    def escape_cell(cell_value: str | None) -> str:
        if cell_value is None:
            return ""
        return (
            str(cell_value)
            .replace("\\", "\\\\")
            .replace("\t", "&emsp;&emsp;")
            .replace("\r\n", "<br>")
            .replace("\r", "<br>")
            .replace("\n", "<br>")
        )

    content_parts = []
    in_table = False

    for element in doc.element.body:
        if element.tag.endswith("p"):
            if in_table:
                content_parts.append("")
                in_table = False
            paragraph = Paragraph(element, doc)
            content_parts.append(paragraph.text)

        elif element.tag.endswith("tbl"):
            if content_parts and not in_table:
                content_parts.append("")
            in_table = True
            table = Table(element, doc)
            for row in table.rows:
                row_text = [escape_cell(cell.text) for cell in row.cells]
                if any(row_text):
                    content_parts.append("\t".join(row_text))

    return "\n".join(content_parts)


def _extract_pptx(file_bytes: bytes) -> str:
    """Trích xuất nội dung PPTX"""
    from pptx import Presentation

    prs = Presentation(BytesIO(file_bytes))
    return "\n".join(
        shape.text
        for slide in prs.slides
        for shape in slide.shapes
        if hasattr(shape, "text")
    )


def _extract_xlsx(file_bytes: bytes) -> str:
    """Trích xuất nội dung XLSX"""
    from openpyxl import load_workbook

    wb = load_workbook(BytesIO(file_bytes))

    def escape_cell(cell_value: str | int | float | None) -> str:
        if cell_value is None:
            return ""
        return (
            str(cell_value)
            .replace("\\", "\\\\")
            .replace("\t", "\\t")
            .replace("\r\n", "\\n")
            .replace("\r", "\\n")
            .replace("\n", "\\n")
        )

    content_parts = []
    sheet_separator = "=" * 20

    for idx, sheet in enumerate(wb):
        if idx > 0:
            content_parts.append("")

        safe_title = (
            str(sheet.title).replace("\n", " ").replace("\t", " ").replace("\r", " ")
        )
        content_parts.append(f"{sheet_separator} Sheet: {safe_title} {sheet_separator}")

        max_cols = sheet.max_column or 0
        for row in sheet.iter_rows(values_only=True):
            row_parts = [
                escape_cell(row[i]) if i < len(row) else "" for i in range(max_cols)
            ]
            content_parts.append("\t".join(row_parts) if any(row_parts) else "")

    content_parts.append(sheet_separator)
    return "\n".join(content_parts)


async def extract_content(file_bytes: bytes, file_name: str) -> tuple[str, str]:
    ext = Path(file_name).suffix.lower()

    # 1. Plain Text Formats
    if ext in TEXT_EXTENSIONS:
        for encoding in ["utf-8", "latin-1", "cp1252", "iso-8859-1"]:
            try:
                content = file_bytes.decode(encoding)
                if not content.strip():
                    raise ValueError(f"[Kafka Extract] Empty text file: {file_name}")
                logger.info(
                    f"[Kafka Extract] Text file read with {encoding}: {file_name} ({len(content)} chars)"
                )
                return content, "plain"
            except UnicodeDecodeError:
                continue

        raise ValueError(f"[Kafka Extract] Text file encoding error for: {file_name}")

    # 2. Office Formats
    if ext in OFFICE_EXTENSIONS:
        try:
            if _is_docling_available():
                logger.info(f"[Kafka Extract] Attempting Docling for {file_name}")
                content = await asyncio.to_thread(
                    _run_docling_from_bytes_sync, file_bytes, ext
                )
                if content.strip():
                    return content, "markdown"

            logger.info(
                f"[Kafka Extract] Docling failed/unavailable, using native extractor for {file_name}"
            )
            if ext == ".docx":
                content = await asyncio.to_thread(_extract_docx, file_bytes)
            elif ext == ".pptx":
                content = await asyncio.to_thread(_extract_pptx, file_bytes)
            elif ext == ".xlsx":
                content = await asyncio.to_thread(_extract_xlsx, file_bytes)

            if content.strip():
                return content, "plain"
        except Exception as e:
            logger.warning(
                f"[Kafka Extract] Primary extraction failed for {file_name}: {e}. Trying OCR fallback."
            )

        try:
            ocr_res, _ = await _run_ocr_pipeline(file_bytes, ext)
            if ocr_res.text.strip():
                return ocr_res.text, (ocr_res.format or "plain")
        except Exception as e:
            logger.error(f"[Kafka Extract] OCR fallback failed for {file_name}: {e}")

        raise ValueError(
            f"[Kafka Extract] All extraction methods failed for office document {file_name}"
        )

    # 3. Binary / OCR Pipeline
    if ext in BINARY_EXTENSIONS:
        try:
            is_image = ext in {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}
            ocr_res, ocr_config = await _run_ocr_pipeline(file_bytes, ext)

            logger.info(
                f"[Kafka Extract] Document processed with {ocr_res.engine_used} engine: "
                f"{file_name} ({ocr_res.page_count} pages, {ocr_res.processing_time:.2f}s)"
            )

            content = ocr_res.text
            output_format = ocr_res.format or "plain"

            if output_format == "markdown":
                preprocessor = MarkdownPreprocessor(
                    preserve_structure=ocr_config.preserve_markdown_structure
                )
                content = preprocessor.preprocess(content)

            if content.strip():
                return content, output_format

            is_binary_strict = is_image or ext == ".pdf"
            if is_binary_strict:
                logger.warning(
                    f"[Kafka Extract] OCR returned no content for {file_name}. Skipping fallback."
                )
            elif _is_docling_available():
                logger.info(f"[Kafka Extract] Fallback to Docling for {file_name}")
                content = await asyncio.to_thread(
                    _run_docling_from_bytes_sync, file_bytes, ext
                )
                if content.strip():
                    return content, "markdown"

            raise ValueError(
                f"Document extraction returned no content: {getattr(ocr_res, 'error', 'Unknown Error')}"
            )

        except Exception as e:
            raise ValueError(
                f"Document processing error: Failed to extract text from Document {file_name}: {e}"
            )

    raise ValueError(
        f"Unsupported file type: {ext} - File extension {ext} is not supported"
    )

TEXT_EXTENSIONS = {
    ".txt", ".md", ".mdx", ".html", ".htm", ".tex", ".json", ".xml", ".yaml", ".yml",
    ".log", ".conf", ".ini", ".properties", ".sql", ".bat", ".sh", ".c", ".h", ".cpp",
    ".hpp", ".py", ".java", ".js", ".ts", ".swift", ".go", ".rs", ".rb", ".php", ".css",
    ".scss", ".less",
}
OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx"}
BINARY_EXTENSIONS = {
    ".rtf", ".odt", ".epub", ".csv", ".pdf", ".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp",
}
