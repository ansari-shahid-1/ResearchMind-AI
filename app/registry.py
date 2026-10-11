"""SQLite document registry.

Chroma knows about chunks, not about files. This keeps the file-level
bookkeeping: what was uploaded, what it hashed to, how many chunks came out.
Kept as plain sqlite3 rather than an ORM because there is exactly one table
and an ORM would be more code than the queries.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from app.config import settings
from app.models import Document

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    filename    TEXT NOT NULL,
    pages       INTEGER NOT NULL,
    chunks      INTEGER NOT NULL,
    size_bytes  INTEGER NOT NULL,
    sha256      TEXT NOT NULL,
    uploaded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    settings.ensure_dirs()
    conn = sqlite3.connect(settings.registry_db)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    with connect() as conn:
        conn.executescript(_SCHEMA)


def add(doc: Document) -> None:
    with connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, title, filename, pages, chunks, size_bytes, sha256, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc.doc_id,
                doc.title,
                doc.filename,
                doc.pages,
                doc.chunks,
                doc.size_bytes,
                doc.sha256,
                doc.uploaded_at,
            ),
        )


def get(doc_id: str) -> Document | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return Document(**dict(row)) if row else None


def find_by_hash(sha256: str) -> Document | None:
    """Uploads of the same PDF twice should not double-index it."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE sha256 = ? LIMIT 1", (sha256,)
        ).fetchone()
    return Document(**dict(row)) if row else None


def list_all() -> list[Document]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
    return [Document(**dict(r)) for r in rows]


def remove(doc_id: str) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    return cur.rowcount > 0


def clear() -> None:
    with connect() as conn:
        conn.execute("DELETE FROM documents")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def to_json(docs: list[Document]) -> str:
    return json.dumps([d.model_dump() for d in docs], indent=2)
