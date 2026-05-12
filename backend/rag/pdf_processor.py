import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import get_settings
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file given its bytes."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []
    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        if page_text.strip():
            text_parts.append(page_text)
    page_count = doc.page_count
    doc.close()

    full_text = "\n\n".join(text_parts)
    logger.info(f"Extracted {len(full_text)} characters from {page_count} pages")
    return full_text


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    logger.info(f"Split text into {len(chunks)} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap})")
    return chunks


def process_pdf(file_bytes: bytes) -> list[str]:
    """Full pipeline: extract text from PDF bytes and chunk it."""
    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        raise ValueError("PDF contains no extractable text")
    chunks = chunk_text(text)
    return chunks
