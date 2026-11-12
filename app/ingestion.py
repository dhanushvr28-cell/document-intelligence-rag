"""Text extraction (PDF, DOCX, TXT, MD) and overlap-aware chunking."""
import io
import re
from dataclasses import dataclass

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedFile(ValueError):
    pass


@dataclass
class ChunkData:
    position: int
    page: int
    text: str


def extract_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    """Return (page_number, text) pairs. Non-paged formats are page 1."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED:
        raise UnsupportedFile(f"Unsupported file type '{ext}'. Allowed: {sorted(SUPPORTED)}")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    elif ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        pages = [(1, "\n\n".join(p.text for p in doc.paragraphs))]
    else:
        pages = [(1, data.decode("utf-8", errors="ignore"))]
    return [(n, t) for n, t in pages if t.strip()]


def _split_units(text: str) -> list[str]:
    """Split into paragraphs, then sentences, so chunks end on natural boundaries."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            units.extend(s for s in re.split(r"(?<=[.!?])\s+", para) if s)
    return units


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Greedy sentence packing up to `size` characters, carrying `overlap` characters forward."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[str] = []
    current = ""
    for unit in _split_units(text):
        while len(unit) > size:  # a single sentence longer than a chunk: hard split
            head, unit = unit[:size], unit[size - overlap:]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        if current and len(current) + 1 + len(unit) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail  # avoid starting mid-word
            current = tail if len(tail) + 1 + len(unit) <= size else ""
        current = f"{current} {unit}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages: list[tuple[int, str]], size: int, overlap: int) -> list[ChunkData]:
    out: list[ChunkData] = []
    for page, text in pages:
        for piece in chunk_text(text, size, overlap):
            out.append(ChunkData(position=len(out), page=page, text=piece))
    return out
