"""HTTP surface. Deliberately thin - every route is a couple of lines over a
service function, so the same code paths are reachable from the CLI and the
eval harness without going through HTTP.
"""

import hashlib
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app import llm, registry
from app.agent import graph as agent_graph
from app.config import settings
from app.ingest.pipeline import ingest_pdf, reindex, remove_document
from app.models import Answer, Document, IngestResponse, QueryRequest
from app.qa import answer as rag_answer

router = APIRouter()

MAX_UPLOAD_BYTES = 40 * 1024 * 1024


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents": len(registry.list_all()),
        "chunks": _chunk_count(),
        "chat_model": settings.chat_model,
        "embed_model": settings.embed_model,
        "ollama": llm.health(),
    }


def _chunk_count() -> int:
    try:
        from app.retrieval import store

        return store.count()
    except Exception:
        return 0


@router.post("/documents", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
) -> IngestResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 40 MB.")
    if not raw.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Only PDFs are supported.")

    settings.ensure_dirs()
    # the content hash is the doc id, so re-uploading the same paper writes
    # to the same path instead of leaving an orphan copy behind
    doc_id = hashlib.sha256(raw).hexdigest()[:12]
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    target = settings.upload_dir / f"{doc_id}{suffix}"
    target.write_bytes(raw)

    try:
        doc, duplicate = ingest_pdf(target, title=title)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}") from exc

    if doc.chunks == 0:
        raise HTTPException(
            status_code=422,
            detail="No extractable text - this looks like a scanned PDF. OCR is not implemented.",
        )

    return IngestResponse(
        doc_id=doc.doc_id,
        title=doc.title,
        pages=doc.pages,
        chunks=doc.chunks,
        duplicates=duplicate,
    )


@router.get("/documents", response_model=list[Document])
def list_documents() -> list[Document]:
    return registry.list_all()


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    if not registry.get(doc_id):
        raise HTTPException(status_code=404, detail="Unknown document.")
    return {"deleted": remove_document(doc_id), "doc_id": doc_id}


@router.post("/documents/{doc_id}/reindex", response_model=Document)
def reindex_document(doc_id: str) -> Document:
    doc = reindex(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown document or the file is missing.")
    return doc


@router.post("/query", response_model=Answer)
def query(request: QueryRequest) -> Answer:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    if not registry.list_all():
        raise HTTPException(status_code=409, detail="No documents uploaded yet.")

    if request.mode == "agent":
        return agent_graph.run(question, doc_ids=request.doc_ids)
    return rag_answer(question, doc_ids=request.doc_ids, top_k=request.top_k)
