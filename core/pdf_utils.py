from __future__ import annotations
from pypdf import PdfReader
from pathlib import Path
from typing import List, Dict, Any

def extract_pdf_pages(pdf_path: Path, max_pages: int = 30) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages_text = []
    n = min(len(reader.pages), max_pages)
    for i in range(n):
        try:
            t = reader.pages[i].extract_text() or ""
        except Exception:
            t = ""
        pages_text.append(t)
    return pages_text

def chunk_text_with_page(pages_text: List[str], chunk_chars: int, overlap: int) -> List[Dict[str, Any]]:
    # 把每页拼接成带分隔符的长串，同时记录每个字符属于哪一页。
    full = []
    char2page = []
    for pi, txt in enumerate(pages_text, start=1):
        txt = txt.replace("\x00", " ")
        full.append(txt + "\n")
        char2page.extend([pi] * (len(txt) + 1))
    full_text = "".join(full)
    if not full_text.strip():
        return []

    chunks = []
    step = max(1, chunk_chars - overlap)
    n = len(full_text)

    for start in range(0, n, step):
        end = min(n, start + chunk_chars)
        chunk = full_text[start:end].strip()
        effective_len = len(chunk.replace("\n", "").replace(" ", ""))
        if effective_len < 30:
            continue
        ps = char2page[start] if start < len(char2page) else 1
        pe = char2page[end - 1] if end - 1 < len(char2page) else ps
        chunks.append({"text": chunk, "page_start": ps, "page_end": pe})
        if end >= n:
            break
    return chunks
