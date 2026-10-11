"""Upload -> text -> chunks -> embeddings -> Chroma.

One function, `ingest_pdf`, because that is the only way a document ever
enters the system. Duplicate handling is by content hash: uploading the same
PDF twice returns the first one instead of double-indexing it.

A copy of every PDF is kept in the upload dir, named after its content hash.
That is what `reindex` reads from, so it needs to hold for documents that
arrived through the CLI too and not just through the HTTP route.
"""

import hashlib
import shutil
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


def _ensure_stored(path: Path, doc_id: str) -> Path:
    """Copy the PDF into the upload dir, named after its content hash.

    Every ingest path has to end up here, not just the HTTP upload. The
    registry points at a file by name, so a document ingested from the CLI
    used to leave the registry referencing a path that did not exist and
    `reindex` would 404 on it. Keying the stored name on the content hash also
    means re-uploading the same paper overwrites its own file rather than
    leaving an orphan behind.
    """
    settings.ensure_dirs()
    suffix = path.suffix or ".pdf"
    target = settings.upload_dir / f"{doc_id}{suffix}"
    if path.resolve() != target.resolve():
        shutil.copyfile(path, target)
    return target


def ingest_pdf(
    path: str | Path,
    title: str | None = None,
) -> tuple[Document, bool]:
    """Returns (document, was_duplicate)."""
    path = Path(path)
    registry.init()

    digest = sha256_of(path)
    doc_id = digest[:12]
    stored = _ensure_stored(path, doc_id)

    existing = registry.find_by_hash(digest)
    if existing:
        # heal documents indexed before the copy existed
        if existing.filename != stored.name:
            existing.filename = stored.name
            registry.add(existing)
        return existing, True

    loaded = pdf.load(path)
    pages = loaded["pages"]
    resolved_title = (title or loaded["title"] or path.stem).strip()

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
        filename=stored.name,
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

    The stored PDF is left on disk on purpose. Re-ingesting it is cheap, it is
    the only copy the system has, and deleting user files on a DELETE that was
    only meant to clear an index would be a nasty surprise.
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
