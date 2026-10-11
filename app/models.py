"""Shared pydantic schemas for the API and internal services."""

from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    title: str
    filename: str
    pages: int
    chunks: int
    size_bytes: int
    uploaded_at: str
    sha256: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    index: int
    text: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    text: str
    score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def label(self) -> str:
        return f"{self.doc_title} p.{self.page}"


class Citation(BaseModel):
    marker: int  # the [1], [2] ... the model wrote
    doc_id: str
    doc_title: str
    page: int
    snippet: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sources: list[RetrievedChunk] = Field(default_factory=list)
    mode: Literal["rag", "agent"] = "rag"
    steps: list[dict] = Field(default_factory=list)
    latency_ms: float = 0.0
    usage: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    mode: Literal["rag", "agent"] = "agent"
    top_k: int | None = None


class IngestResponse(BaseModel):
    doc_id: str
    title: str
    pages: int
    chunks: int
    duplicates: bool = False
