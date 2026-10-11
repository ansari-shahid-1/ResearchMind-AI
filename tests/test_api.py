"""API tests.

These deliberately stay on the paths that do not embed anything, so the suite
runs in a second rather than a minute. Ingest and query are covered by the
eval harness and the ingest CLI instead.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(temp_env):
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 0
    assert "ollama" in body


def test_root(client):
    assert client.get("/").json()["name"] == "ResearchMind AI"


def test_documents_starts_empty(client):
    assert client.get("/documents").json() == []


def test_query_without_documents_is_conflict(client):
    response = client.post("/query", json={"question": "anything?", "mode": "rag"})
    assert response.status_code == 409
    assert "No documents" in response.json()["detail"]


def test_query_rejects_blank_question(client):
    response = client.post("/query", json={"question": "   ", "mode": "rag"})
    assert response.status_code == 400


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400


def test_delete_unknown_document(client):
    assert client.delete("/documents/does-not-exist").status_code == 404


def test_reindex_unknown_document(client):
    assert client.post("/documents/does-not-exist/reindex").status_code == 404
