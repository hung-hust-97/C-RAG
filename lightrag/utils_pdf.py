from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterator

import fitz  # PyMuPDF
from PIL import Image, ImageOps

from lightrag.utils import logger


@dataclass(frozen=True)
class PdfBatch:
    index: int
    start_page: int
    end_page: int
    filename: str
    content: bytes


def _save_pdf(doc: fitz.Document) -> bytes:
    return doc.tobytes(
        garbage=4,
        clean=True,
        deflate=True,
        deflate_images=True,
        deflate_fonts=True,
        use_objstms=1,
    )


def _resize_for_ocr(
    image: Image.Image,
    *,
    max_page_width_px: int,
) -> Image.Image:
    image = ImageOps.exif_transpose(image)

    if image.mode != "L":
        image = image.convert("L")

    if image.width <= max_page_width_px:
        return image

    ratio = max_page_width_px / image.width
    new_height = max(1, int(image.height * ratio))

    return image.resize(
        (max_page_width_px, new_height),
        Image.Resampling.LANCZOS,
    )


def _render_page_to_grayscale_jpeg(
    page: fitz.Page,
    *,
    dpi: int,
    jpeg_quality: int,
    max_page_width_px: int,
) -> bytes:
    """
    Render 1 trang PDF thành ảnh JPEG grayscale.

    - bỏ vector phức tạp;
    - bỏ màu;
    - chuẩn hóa trang thành ảnh;
    - giới hạn kích thước pixel.
    """
    pix = page.get_pixmap(
        dpi=dpi,
        colorspace=fitz.csGRAY,
        alpha=False,
    )

    image = Image.frombytes(
        "L",
        (pix.width, pix.height),
        pix.samples,
    )

    image = _resize_for_ocr(
        image,
        max_page_width_px=max_page_width_px,
    )

    output = io.BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
        progressive=True,
    )

    return output.getvalue()


def iter_pdf_grayscale_batches(
    file_bytes: bytes,
    *,
    source_filename: str = "document.pdf",
    pages_per_batch: int = 5,
    dpi: int = 130,
    jpeg_quality: int = 30,
    max_page_width_px: int = 1400,
) -> Iterator[PdfBatch]:
    """
    Chuyển toàn bộ PDF sang grayscale và chia thành nhiều PDF nhỏ.

    Mỗi batch là một PDF mới, bên trong mỗi trang là ảnh JPEG grayscale.
    """
    if not file_bytes:
        return

    if pages_per_batch <= 0:
        raise ValueError("pages_per_batch must be greater than 0")

    base_name = source_filename.rsplit(".", 1)[0]

    with fitz.open("pdf", file_bytes) as src_doc:
        total_pages = src_doc.page_count

        logger.info(
            "[PDF Batch] Start grayscale batching: "
            f"file={source_filename}, pages={total_pages}, "
            f"pages_per_batch={pages_per_batch}, dpi={dpi}, "
            f"jpeg_quality={jpeg_quality}, max_page_width_px={max_page_width_px}"
        )

        for start_page in range(0, total_pages, pages_per_batch):
            end_page = min(start_page + pages_per_batch, total_pages)

            batch_doc = fitz.open()

            try:
                for page_index in range(start_page, end_page):
                    page = src_doc.load_page(page_index)

                    jpeg_bytes = _render_page_to_grayscale_jpeg(
                        page,
                        dpi=dpi,
                        jpeg_quality=jpeg_quality,
                        max_page_width_px=max_page_width_px,
                    )

                    new_page = batch_doc.new_page(
                        width=page.rect.width,
                        height=page.rect.height,
                    )
                    new_page.insert_image(
                        new_page.rect,
                        stream=jpeg_bytes,
                    )

                batch_bytes = _save_pdf(batch_doc)
                batch_index = start_page // pages_per_batch + 1
                batch_filename = f"{base_name}_batch_{batch_index:04d}.pdf"

                logger.info(
                    "[PDF Batch] Created grayscale batch: "
                    f"file={batch_filename}, pages={start_page + 1}-{end_page}, "
                    f"size={len(batch_bytes)} bytes"
                )

                yield PdfBatch(
                    index=batch_index,
                    start_page=start_page + 1,
                    end_page=end_page,
                    filename=batch_filename,
                    content=batch_bytes,
                )

            finally:
                batch_doc.close()
