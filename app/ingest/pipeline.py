"""Upload -> text -> chunks -> embeddings -> Chroma.

One function, `ingest_pdf`, because that is the only way a document ever
enters the system. Duplicate handling is by content hash: uploading the same
PDF twice returns the first one instead of double-indexing it.
"""

import hashlib
from pathlib import Path

from app import registry
from app.config import settings
from app.ingest import chunking, pdf
from app.models import Document
from app.retrieval import store


def sha256_of(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def ingest_pdf(
    path: str | Path,
    filename: str | None = None,
    title: str | None = None,
) -> tuple[Document, bool]:
    """Returns (document, was_duplicate)."""
    path = Path(path)
    registry.init()

    digest = sha256_of(path)
    existing = registry.find_by_hash(digest)
    if existing:
        return existing, True

    loaded = pdf.load(path)
    pages = loaded["pages"]
    resolved_title = (title or loaded["title"] or path.stem).strip()

    doc_id = digest[:12]
    raw_chunks = chunking.chunk_pages(
        pages,
        size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    chunks = [
        {
            "chunk_id": f"{doc_id}-{c['index']:04d}",
            "doc_id": doc_id,
            "doc_title": resolved_title,
            "page": c["page"],
            "index": c["index"],
            "text": c["text"],
        }
        for c in raw_chunks
    ]

    store.add_chunks(chunks)

    doc = Document(
        doc_id=doc_id,
        title=resolved_title,
        filename=filename or path.name,
        pages=loaded["page_count"],
        chunks=len(chunks),
        size_bytes=path.stat().st_size,
        sha256=digest,
        uploaded_at=registry.now(),
    )
    registry.add(doc)
    return doc, False


def remove_document(doc_id: str) -> bool:
    """Drops chunks from the vector store and the registry entry.

    The original file is left on disk on purpose - re-ingesting is cheap and
    the uploaded PDF is the only copy of the source.
    """
    store.delete_document(doc_id)
    return registry.remove(doc_id)


def reindex(doc_id: str) -> Document | None:
    """Re-chunk an already-uploaded document (after changing chunk settings)."""
    doc = registry.get(doc_id)
    if not doc:
        return None
    path = settings.upload_dir / doc.filename
    if not path.exists():
        return None

    store.delete_document(doc_id)
    loaded = pdf.load(path)
    raw_chunks = chunking.chunk_pages(
        loaded["pages"], size=settings.chunk_size, overlap=settings.chunk_overlap
    )
    chunks = [
        {
            "chunk_id": f"{doc_id}-{c['index']:04d}",
            "doc_id": doc_id,
            "doc_title": doc.title,
            "page": c["page"],
            "index": c["index"],
            "text": c["text"],
        }
        for c in raw_chunks
    ]
    store.add_chunks(chunks)

    doc.chunks = len(chunks)
    doc.pages = loaded["page_count"]
    registry.add(doc)
    return doc
