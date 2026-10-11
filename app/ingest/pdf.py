"""PDF loading. PyMuPDF is the fastest thing that handles two-column academic
layouts without exploding, so that's what we use.
"""

from pathlib import Path

import pymupdf


def load(path: str | Path) -> dict:
    """Return {"title", "page_count", "pages": [{page, text}]} for a PDF.

    Blank pages are dropped. Page numbers stay 1-based and match what a PDF
    reader shows, because that is what a citation has to point at.
    """
    pages: list[dict] = []
    with pymupdf.open(path) as doc:
        title = (doc.metadata or {}).get("title") or ""
        page_count = doc.page_count
        for number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text and text.strip():
                pages.append({"page": number, "text": text})

    title = title.strip() or Path(path).stem
    return {"title": title, "page_count": page_count, "pages": pages}


def page_count(path: str | Path) -> int:
    with pymupdf.open(path) as doc:
        return doc.page_count
