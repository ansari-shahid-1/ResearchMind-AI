"""Pipeline tests.

The embedding step is monkeypatched out. These are about file and registry
bookkeeping, and loading a sentence-transformer to prove that a copy landed on
disk would make the suite a minute slower for no extra confidence.

The bug these cover: a document ingested from the CLI never got copied into
the upload dir, so the registry pointed at a file that did not exist and
`reindex` returned 404 for anything not uploaded over HTTP.
"""

import shutil

import pymupdf
import pytest

from app import registry
from app.config import settings
from app.ingest.pipeline import ingest_pdf, reindex, remove_document
from app.retrieval import store


def make_pdf(path, pages: int = 2, title: str = "Test Paper"):
    doc = pymupdf.open()
    for number in range(pages):
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 500, 700),
            f"Page {number + 1} of the test paper. " * 12,
            fontsize=10,
        )
    doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def no_embeddings(monkeypatch):
    """Record chunks instead of embedding them."""
    recorded: list[list[dict]] = []

    def fake_add(chunks, embeddings=None):
        recorded.append(chunks)
        return len(chunks)

    monkeypatch.setattr(store, "add_chunks", fake_add)
    monkeypatch.setattr(store, "delete_document", lambda doc_id: None)
    return recorded


def test_ingest_copies_the_pdf_into_the_upload_dir(temp_env, no_embeddings):
    source = make_pdf(temp_env / "somewhere-else.pdf")
    doc, duplicate = ingest_pdf(source)

    assert duplicate is False
    stored = settings.upload_dir / doc.filename
    assert stored.exists(), "the PDF was not copied into the upload dir"
    assert stored.name.startswith(doc.doc_id)
    # and the copy is not the original file
    assert stored.resolve() != source.resolve()


def test_reindex_works_for_a_document_ingested_from_a_path_outside_uploads(
    temp_env, no_embeddings
):
    source = make_pdf(temp_env / "outside.pdf", pages=3)
    doc, _ = ingest_pdf(source)
    assert doc.chunks >= 3

    again = reindex(doc.doc_id)
    assert again is not None, "reindex could not find the stored PDF"
    assert again.chunks == doc.chunks
    assert again.pages == 3


def test_reingesting_the_same_bytes_is_a_duplicate(temp_env, no_embeddings):
    source = make_pdf(temp_env / "paper.pdf")
    first, _ = ingest_pdf(source)

    # the same bytes arriving under a different name, as when someone
    # downloads the same paper twice
    second_path = temp_env / "renamed.pdf"
    shutil.copyfile(source, second_path)
    second, duplicate = ingest_pdf(second_path)

    assert duplicate is True
    assert second.doc_id == first.doc_id
    assert len(registry.list_all()) == 1
    assert len(no_embeddings) == 1, "the duplicate was indexed twice"


def test_two_different_pdfs_are_not_duplicates(temp_env, no_embeddings):
    """Dedupe is by content hash, so two separately generated files differ."""
    first, _ = ingest_pdf(make_pdf(temp_env / "a.pdf", title="Paper A"))
    second, duplicate = ingest_pdf(make_pdf(temp_env / "b.pdf", title="Paper B"))

    assert duplicate is False
    assert first.doc_id != second.doc_id
    assert len(registry.list_all()) == 2


def test_duplicate_ingest_repairs_a_stale_filename(temp_env, no_embeddings):
    source = make_pdf(temp_env / "paper.pdf")
    doc, _ = ingest_pdf(source)

    # simulate a row written before the copy existed
    doc.filename = "not-a-real-file.pdf"
    registry.add(doc)

    healed, duplicate = ingest_pdf(source)
    assert duplicate is True
    assert healed.filename == f"{doc.doc_id}.pdf"
    assert (settings.upload_dir / healed.filename).exists()


def test_title_falls_back_to_the_document_then_the_filename(temp_env, no_embeddings):
    with_title = make_pdf(temp_env / "has-title.pdf", title="From Metadata")
    assert ingest_pdf(with_title)[0].title == "From Metadata"

    no_title = make_pdf(temp_env / "no-title.pdf", title="")
    assert ingest_pdf(no_title)[0].title == "no-title"


def test_explicit_title_wins(temp_env, no_embeddings):
    source = make_pdf(temp_env / "paper.pdf", title="From Metadata")
    doc, _ = ingest_pdf(source, title="Override")
    assert doc.title == "Override"


def test_remove_document_clears_the_registry_entry(temp_env, no_embeddings):
    doc, _ = ingest_pdf(make_pdf(temp_env / "paper.pdf"))
    assert remove_document(doc.doc_id) is True
    assert registry.get(doc.doc_id) is None


def test_reindex_of_unknown_document_returns_none(temp_env, no_embeddings):
    assert reindex("nope") is None


def test_reindex_returns_none_when_the_stored_file_is_gone(temp_env, no_embeddings):
    doc, _ = ingest_pdf(make_pdf(temp_env / "paper.pdf"))
    (settings.upload_dir / doc.filename).unlink()
    assert reindex(doc.doc_id) is None
