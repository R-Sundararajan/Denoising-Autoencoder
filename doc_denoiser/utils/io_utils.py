# utils/io_utils.py
# File I/O utilities for saving and loading images and PDFs.

import os
from PIL import Image


def save_image(pil_image: Image.Image, output_path: str):
    """
    Save a PIL image to disk. Creates parent directories if needed.

    Args:
        pil_image: PIL Image to save.
        output_path: Full path including filename and extension (e.g., "outputs/images/result.png").
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pil_image.save(output_path)
    print(f"[IO] Saved image to: {output_path}")


def save_pdf(pdf_bytes: bytes, output_path: str):
    """
    Save PDF bytes to disk. Creates parent directories if needed.

    Args:
        pdf_bytes: Raw PDF bytes.
        output_path: Full path (e.g., "outputs/pdfs/result.pdf").
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"[IO] Saved PDF to: {output_path}")


def ensure_directories():
    """
    Create all required project directories if they don't exist.
    Call this at app startup.
    """
    dirs = [
        "data/raw",
        "data/processed",
        "models/weights",
        "outputs/images",
        "outputs/pdfs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
