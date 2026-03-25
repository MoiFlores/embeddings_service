"""
Extrae texto de PDFs e imágenes.
- PDF con texto digital: PyMuPDF (rápido, sin OCR)
- PDF escaneado / imagen: pytesseract
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from typing import Generator

CHUNK_SIZE = 500        # caracteres por chunk
CHUNK_OVERLAP = 50      # solapamiento entre chunks


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extrae texto de un PDF. Si una página no tiene texto, aplica OCR."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = []

    for page in doc:
        text = page.get_text().strip()
        if text:
            full_text.append(text)
        else:
            # Página escaneada: renderizar y aplicar OCR
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_text = pytesseract.image_to_string(img, lang="spa+eng")
            full_text.append(ocr_text.strip())

    doc.close()
    return "\n".join(full_text)


def extract_text_from_image(file_bytes: bytes) -> str:
    """OCR directo sobre una imagen."""
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img, lang="spa+eng")


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Divide el texto en chunks con solapamiento."""
    if not text.strip():
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap

    return [c for c in chunks if c]  # Filtrar chunks vacíos
