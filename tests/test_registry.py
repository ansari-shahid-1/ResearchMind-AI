from app import registry
from app.models import Document


def make(doc_id="abc123", sha="deadbeef", title="A Paper"):
    return Document(
        doc_id=doc_id,
        title=title,
        filename=f"{doc_id}.pdf",
        pages=5,
        chunks=10,
        size_bytes=1234,
        sha256=sha,
        uploaded_at=registry.now(),
    )


def test_add_and_get(temp_env):
    registry.add(make())
    doc = registry.get("abc123")
    assert doc is not None
    assert doc.title == "A Paper"
    assert doc.chunks == 10


def test_get_missing_returns_none(temp_env):
    assert registry.get("nope") is None


def test_find_by_hash(temp_env):
    registry.add(make())
    assert registry.find_by_hash("deadbeef").doc_id == "abc123"
    assert registry.find_by_hash("other") is None


def test_list_is_newest_first(temp_env):
    first = make("a1", "s1", "First")
    registry.add(first)
    second = make("a2", "s2", "Second")
    second.uploaded_at = "2999-01-01T00:00:00+00:00"
    registry.add(second)
    titles = [d.title for d in registry.list_all()]
    assert titles[0] == "Second"


def test_remove(temp_env):
    registry.add(make())
    assert registry.remove("abc123") is True
    assert registry.get("abc123") is None
    assert registry.remove("abc123") is False


def test_duplicate_doc_id_replaces(temp_env):
    registry.add(make())
    updated = make()
    updated.chunks = 99
    registry.add(updated)
    assert len(registry.list_all()) == 1
    assert registry.get("abc123").chunks == 99


def test_clear(temp_env):
    registry.add(make("a1", "s1"))
    registry.add(make("a2", "s2"))
    registry.clear()
    assert registry.list_all() == []


def test_init_is_idempotent(temp_env):
    registry.init()
    registry.init()
    assert registry.list_all() == []
