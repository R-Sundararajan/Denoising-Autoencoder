# inference/pdf_utils.py
# PDF processing utilities: extract pages as images, rebuild PDFs from images.
# Uses PyMuPDF (fitz) for reliable PDF handling on Windows.

import io
from PIL import Image

# PyMuPDF is imported as 'fitz'
import fitz  # pip install PyMuPDF


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> list:
    """
    Convert a PDF file (as bytes) into a list of PIL images, one per page.

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        dpi: Resolution for rendering. 200 is good for documents.

    Returns:
        List of PIL Images (one per page), in RGB mode.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Render page to pixmap at specified DPI
        # Default PDF resolution is 72 DPI, so zoom = dpi / 72
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)

        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)

    doc.close()
    return images


def images_to_pdf(images: list) -> bytes:
    """
    Combine a list of PIL images into a single PDF file.

    Args:
        images: List of PIL Images (any mode — will be converted to RGB).

    Returns:
        PDF file as bytes, ready to download or save.
    """
    if not images:
        raise ValueError("No images provided to create PDF.")

    # Convert all images to RGB (PDF requires it)
    rgb_images = []
    for img in images:
        if img.mode != "RGB":
            rgb_images.append(img.convert("RGB"))
        else:
            rgb_images.append(img)

    # Save as PDF using PIL's built-in PDF support
    pdf_buffer = io.BytesIO()
    if len(rgb_images) == 1:
        rgb_images[0].save(pdf_buffer, format="PDF")
    else:
        # First image + append remaining pages
        rgb_images[0].save(
            pdf_buffer,
            format="PDF",
            save_all=True,
            append_images=rgb_images[1:],
        )

    pdf_buffer.seek(0)
    return pdf_buffer.read()
