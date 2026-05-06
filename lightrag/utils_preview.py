"""
Utilities for creating compressed previews of documents.

Uses existing project dependencies (PyMuPDF/fitz, Pillow) to generate
lightweight previews for client display without requiring additional packages.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def compress_pdf_for_preview(
    pdf_path: str | Path,
    max_pages: int = 10,
    target_width: int = 800,
    jpeg_quality: int = 75,
) -> Tuple[bytes, dict]:
    """
    Compress PDF for preview by converting pages to low-res images.
    
    Args:
        pdf_path: Path to source PDF file
        max_pages: Maximum number of pages to include (default: 10)
        target_width: Target width in pixels for each page (default: 800)
        jpeg_quality: JPEG compression quality 0-100 (default: 75)
    
    Returns:
        Tuple of (compressed_pdf_bytes, metadata_dict)
        metadata includes: original_size, compressed_size, page_count, compression_ratio
    
    Example:
        >>> pdf_bytes, meta = compress_pdf_for_preview("document.pdf", max_pages=5)
        >>> print(f"Reduced from {meta['original_size']/1024:.1f}KB to {meta['compressed_size']/1024:.1f}KB")
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError as e:
        logger.error(f"Required dependencies not available: {e}")
        raise
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    original_size = pdf_path.stat().st_size
    
    # Open source PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_to_process = min(max_pages, total_pages)
    
    # Create new PDF for compressed output
    output_doc = fitz.open()
    
    for page_num in range(pages_to_process):
        page = doc[page_num]
        
        # Calculate zoom to achieve target width
        page_width = page.rect.width
        zoom = target_width / page_width if page_width > target_width else 1.0
        
        # Render page to pixmap (image) at lower resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image for JPEG compression
        img_data = pix.tobytes("jpeg", jpeg_quality)
        img = Image.open(io.BytesIO(img_data))
        
        # Save compressed image to buffer
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=jpeg_quality, optimize=True)
        img_bytes = img_buffer.getvalue()
        
        # Create new page in output PDF with compressed image
        img_doc = fitz.open("jpeg", img_bytes)
        output_doc.insert_pdf(img_doc)
        img_doc.close()
    
    # Save compressed PDF to bytes
    output_buffer = io.BytesIO()
    output_doc.save(output_buffer, garbage=4, deflate=True, clean=True)
    compressed_bytes = output_buffer.getvalue()
    
    doc.close()
    output_doc.close()
    
    # Calculate metadata
    compressed_size = len(compressed_bytes)
    compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    
    metadata = {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "page_count": pages_to_process,
        "total_pages": total_pages,
        "compression_ratio": round(compression_ratio, 2),
        "truncated": pages_to_process < total_pages,
    }
    
    logger.info(
        f"PDF preview: {original_size/1024:.1f}KB → {compressed_size/1024:.1f}KB "
        f"({compression_ratio:.1f}% reduction, {pages_to_process}/{total_pages} pages)"
    )
    
    return compressed_bytes, metadata


def create_pdf_thumbnail(
    pdf_path: str | Path,
    target_size: Tuple[int, int] = (400, 600),
    jpeg_quality: int = 70,
) -> Tuple[bytes, dict]:
    """
    Create a thumbnail image of the first page of a PDF.
    
    Args:
        pdf_path: Path to source PDF file
        target_size: Target (width, height) in pixels (default: 400x600)
        jpeg_quality: JPEG compression quality 0-100 (default: 70)
    
    Returns:
        Tuple of (jpeg_bytes, metadata_dict)
        metadata includes: width, height, file_size, page_count
    
    Example:
        >>> thumb_bytes, meta = create_pdf_thumbnail("document.pdf")
        >>> b64_thumb = base64.b64encode(thumb_bytes).decode('utf-8')
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image
    except ImportError as e:
        logger.error(f"Required dependencies not available: {e}")
        raise
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Open PDF and get first page
    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        raise ValueError(f"PDF has no pages: {pdf_path}")
    
    page = doc[0]
    
    # Calculate zoom to fit target size while maintaining aspect ratio
    page_rect = page.rect
    width_ratio = target_size[0] / page_rect.width
    height_ratio = target_size[1] / page_rect.height
    zoom = min(width_ratio, height_ratio)
    
    # Render page to pixmap
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    
    # Convert to PIL Image
    img_data = pix.tobytes("jpeg", jpeg_quality)
    img = Image.open(io.BytesIO(img_data))
    
    # Compress to JPEG
    output_buffer = io.BytesIO()
    img.save(output_buffer, format='JPEG', quality=jpeg_quality, optimize=True)
    jpeg_bytes = output_buffer.getvalue()
    
    doc.close()
    
    metadata = {
        "width": img.width,
        "height": img.height,
        "file_size": len(jpeg_bytes),
        "page_count": len(doc),
    }
    
    logger.debug(
        f"PDF thumbnail: {img.width}x{img.height}, {len(jpeg_bytes)/1024:.1f}KB"
    )
    
    return jpeg_bytes, metadata


def pdf_to_base64_preview(
    pdf_path: str | Path,
    max_pages: int = 10,
    target_width: int = 800,
    jpeg_quality: int = 75,
) -> Tuple[str, dict]:
    """
    Create a base64-encoded compressed PDF preview.
    
    Convenience wrapper around compress_pdf_for_preview() that returns
    base64 string ready for embedding in JSON responses.
    
    Args:
        pdf_path: Path to source PDF file
        max_pages: Maximum number of pages to include
        target_width: Target width in pixels for each page
        jpeg_quality: JPEG compression quality 0-100
    
    Returns:
        Tuple of (base64_string, metadata_dict)
    
    Example:
        >>> b64_pdf, meta = pdf_to_base64_preview("document.pdf", max_pages=5)
        >>> response = {"preview_b64": b64_pdf, "metadata": meta}
    """
    pdf_bytes, metadata = compress_pdf_for_preview(
        pdf_path, max_pages, target_width, jpeg_quality
    )
    b64_string = base64.b64encode(pdf_bytes).decode('utf-8')
    return b64_string, metadata


def pdf_thumbnail_to_base64(
    pdf_path: str | Path,
    target_size: Tuple[int, int] = (400, 600),
    jpeg_quality: int = 70,
) -> Tuple[str, dict]:
    """
    Create a base64-encoded thumbnail of the first PDF page.
    
    Convenience wrapper around create_pdf_thumbnail() that returns
    base64 string ready for embedding in JSON responses.
    
    Args:
        pdf_path: Path to source PDF file
        target_size: Target (width, height) in pixels
        jpeg_quality: JPEG compression quality 0-100
    
    Returns:
        Tuple of (base64_string, metadata_dict)
    
    Example:
        >>> b64_thumb, meta = pdf_thumbnail_to_base64("document.pdf")
        >>> response = {"thumbnail_b64": b64_thumb, "width": meta["width"]}
    """
    jpeg_bytes, metadata = create_pdf_thumbnail(pdf_path, target_size, jpeg_quality)
    b64_string = base64.b64encode(jpeg_bytes).decode('utf-8')
    return b64_string, metadata


def compress_image_for_preview(
    image_path: str | Path,
    max_width: int = 800,
    jpeg_quality: int = 75,
) -> Tuple[bytes, dict]:
    """
    Compress an image file for preview display.
    
    Args:
        image_path: Path to source image file
        max_width: Maximum width in pixels (maintains aspect ratio)
        jpeg_quality: JPEG compression quality 0-100
    
    Returns:
        Tuple of (jpeg_bytes, metadata_dict)
    
    Example:
        >>> img_bytes, meta = compress_image_for_preview("photo.png")
        >>> b64_img = base64.b64encode(img_bytes).decode('utf-8')
    """
    try:
        from PIL import Image
    except ImportError as e:
        logger.error(f"PIL not available: {e}")
        raise
    
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original_size = image_path.stat().st_size
    
    # Open and resize image
    img = Image.open(image_path)
    original_width, original_height = img.size
    
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to RGB if necessary (for PNG with alpha, etc.)
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Compress to JPEG
    output_buffer = io.BytesIO()
    img.save(output_buffer, format='JPEG', quality=jpeg_quality, optimize=True)
    jpeg_bytes = output_buffer.getvalue()
    
    compressed_size = len(jpeg_bytes)
    compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    
    metadata = {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "original_dimensions": (original_width, original_height),
        "compressed_dimensions": (img.width, img.height),
        "compression_ratio": round(compression_ratio, 2),
    }
    
    logger.info(
        f"Image preview: {original_size/1024:.1f}KB → {compressed_size/1024:.1f}KB "
        f"({compression_ratio:.1f}% reduction)"
    )
    
    return jpeg_bytes, metadata


def image_to_base64_preview(
    image_path: str | Path,
    max_width: int = 800,
    jpeg_quality: int = 75,
) -> Tuple[str, dict]:
    """
    Create a base64-encoded compressed image preview.
    
    Args:
        image_path: Path to source image file
        max_width: Maximum width in pixels
        jpeg_quality: JPEG compression quality 0-100
    
    Returns:
        Tuple of (base64_string, metadata_dict)
    """
    jpeg_bytes, metadata = compress_image_for_preview(image_path, max_width, jpeg_quality)
    b64_string = base64.b64encode(jpeg_bytes).decode('utf-8')
    return b64_string, metadata
