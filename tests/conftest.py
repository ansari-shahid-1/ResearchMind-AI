"""Shared fixtures.

The settings object is a module-level singleton, so tests point it at a tmp
directory instead of letting anything touch the real ./data. The store keeps a
cached Chroma client, so that has to be dropped too or a test would keep
writing into the previous tmp path.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import registry  # noqa: E402
from app.config import settings  # noqa: E402
from app.retrieval import search, store  # noqa: E402


@pytest.fixture
def temp_env(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "chroma_dir", tmp_path / "chroma")
    monkeypatch.setattr(settings, "registry_db", tmp_path / "registry.db")
    monkeypatch.setattr(store, "_client", None)
    # the BM25 cache is keyed on the store revision, which restarts at 0
    monkeypatch.setattr(search, "_bm25_cache", {"revision": None, "index": None, "chunks": []})

    settings.ensure_dirs()
    registry.init()
    yield tmp_path

    store._client = None
