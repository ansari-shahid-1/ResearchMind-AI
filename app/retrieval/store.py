"""Chroma-backed chunk store.

We keep the raw chunk text and page number in Chroma metadata as well as in
the registry, so a retrieval result is self-contained and can be rendered as
a citation without a second lookup.

`_REVISION` is bumped on every write and is what the BM25 cache keys off, so
ingesting a document invalidates the sparse index without us having to thread
a cache object through the app.
"""

from typing import Any

import chromadb

from app.config import settings
from app.retrieval.embeddings import embed, embed_one

_REVISION = 0
_client = None


def revision() -> int:
    return _REVISION


def _bump() -> None:
    global _REVISION
    _REVISION += 1


def client():
    global _client
    if _client is None:
        settings.ensure_dirs()
        _client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return _client


def collection():
    return client().get_or_create_collection(
        name=settings.collection,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[dict], embeddings: list[list[float]] | None = None) -> int:
    """chunks: [{chunk_id, doc_id, doc_title, page, index, text}]"""
    if not chunks:
        return 0
    vectors = embeddings if embeddings is not None else embed([c["text"] for c in chunks])
    collection().add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "doc_id": c["doc_id"],
                "doc_title": c["doc_title"],
                "page": int(c["page"]),
                "index": int(c["index"]),
            }
            for c in chunks
        ],
    )
    _bump()
    return len(chunks)


def query_dense(
    query: str,
    k: int = 20,
    doc_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    result = collection().query(
        query_embeddings=[embed_one(query)],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for i, chunk_id in enumerate(ids):
        meta = metas[i] or {}
        hits.append(
            {
                "chunk_id": chunk_id,
                "doc_id": meta.get("doc_id", ""),
                "doc_title": meta.get("doc_title", ""),
                "page": int(meta.get("page", 0)),
                "text": docs[i],
                "distance": float(dists[i]) if i < len(dists) else 0.0,
            }
        )
    return hits


def all_chunks() -> list[dict[str, Any]]:
    data = collection().get(include=["documents", "metadatas"])
    out = []
    for i, chunk_id in enumerate(data.get("ids") or []):
        meta = (data.get("metadatas") or [])[i] or {}
        out.append(
            {
                "chunk_id": chunk_id,
                "doc_id": meta.get("doc_id", ""),
                "doc_title": meta.get("doc_title", ""),
                "page": int(meta.get("page", 0)),
                "index": int(meta.get("index", 0)),
                "text": (data.get("documents") or [])[i],
            }
        )
    return out


def get_page(doc_id: str, page: int) -> list[dict[str, Any]]:
    """Every chunk on one page of one document, in order.

    This is the extractor tool's lookup: it works on a whole page rather than
    a similarity match, because "what was the reported F1 on page 6" should
    not depend on whether the question embeds near that table.
    """
    data = collection().get(
        where={"$and": [{"doc_id": doc_id}, {"page": int(page)}]},
        include=["documents", "metadatas"],
    )
    rows = []
    for i, chunk_id in enumerate(data.get("ids") or []):
        meta = (data.get("metadatas") or [])[i] or {}
        rows.append(
            {
                "chunk_id": chunk_id,
                "doc_id": meta.get("doc_id", ""),
                "doc_title": meta.get("doc_title", ""),
                "page": int(meta.get("page", 0)),
                "index": int(meta.get("index", 0)),
                "text": (data.get("documents") or [])[i],
            }
        )
    return sorted(rows, key=lambda r: r["index"])


def delete_document(doc_id: str) -> None:
    collection().delete(where={"doc_id": doc_id})
    _bump()


def count() -> int:
    return collection().count()


def reset() -> None:
    """Only used by tests and the eval runner."""
    # Chroma raises rather than no-oping when the collection is absent, which
    # is the common case on a fresh checkout
    try:
        client().delete_collection(settings.collection)
    except Exception:
        pass
    _bump()
