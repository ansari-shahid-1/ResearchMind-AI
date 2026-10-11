"""Hybrid retrieval.

Dense search is good at paraphrases, BM25 is good at exact terms - method
names, dataset names, numbers. Research-paper questions are full of the
latter ("what F1 did the FiD model get on NQ?"), so we run both and fuse
with reciprocal rank fusion instead of picking one.

RRF is used rather than score blending because cosine distances and BM25
scores are not on a comparable scale, and normalising them per-query turned
out to be more fragile than just using the ranks.
"""

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from app.config import settings
from app.models import RetrievedChunk
from app.retrieval import store

_TOKEN = re.compile(r"[a-z0-9]+")

# Short list, just enough to stop "the paper" dominating the sparse side.
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "was", "were", "which", "with",
}

_bm25_cache: dict = {"revision": None, "index": None, "chunks": []}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _bm25():
    rev = store.revision()
    if _bm25_cache["revision"] != rev or _bm25_cache["index"] is None:
        chunks = store.all_chunks()
        corpus = [tokenize(c["text"]) for c in chunks]
        _bm25_cache.update(
            revision=rev,
            chunks=chunks,
            index=BM25Okapi(corpus) if corpus else None,
        )
    return _bm25_cache["index"], _bm25_cache["chunks"]


def query_sparse(query: str, k: int = 20, doc_ids: list[str] | None = None) -> list[dict]:
    index, chunks = _bm25()
    if index is None:
        return []
    scores = index.get_scores(tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in ranked:
        if doc_ids and chunks[i]["doc_id"] not in doc_ids:
            continue
        if scores[i] <= 0:
            break
        out.append({**chunks[i], "bm25": float(scores[i])})
        if len(out) >= k:
            break
    return out


def _rrf(dense: list[dict], sparse: list[dict], k: int) -> list[RetrievedChunk]:
    scores: dict[str, float] = defaultdict(float)
    pool: dict[str, dict] = {}

    for rank, hit in enumerate(dense, start=1):
        cid = hit["chunk_id"]
        scores[cid] += 1.0 / (k + rank)
        pool[cid] = dict(hit, _dense_rank=rank)

    for rank, hit in enumerate(sparse, start=1):
        cid = hit["chunk_id"]
        scores[cid] += 1.0 / (k + rank)
        if cid in pool:
            pool[cid]["_sparse_rank"] = rank
        else:
            pool[cid] = dict(hit, _sparse_rank=rank)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for cid, score in ranked:
        hit = pool[cid]
        out.append(
            RetrievedChunk(
                chunk_id=cid,
                doc_id=hit.get("doc_id", ""),
                doc_title=hit.get("doc_title", ""),
                page=int(hit.get("page", 0)),
                text=hit.get("text", ""),
                score=round(score, 6),
                dense_rank=hit.get("_dense_rank"),
                sparse_rank=hit.get("_sparse_rank"),
            )
        )
    return out


def search(
    query: str,
    k: int | None = None,
    doc_ids: list[str] | None = None,
    use_sparse: bool = True,
) -> list[RetrievedChunk]:
    """Main entry point used by the agent tool and the plain RAG path."""
    k = k or settings.top_k
    dense = store.query_dense(query, k=settings.dense_candidates, doc_ids=doc_ids)

    if not use_sparse:
        return [
            RetrievedChunk(
                chunk_id=h["chunk_id"],
                doc_id=h["doc_id"],
                doc_title=h["doc_title"],
                page=h["page"],
                text=h["text"],
                score=round(1.0 - h["distance"], 6),
                dense_rank=i,
            )
            for i, h in enumerate(dense, start=1)
        ]

    sparse = query_sparse(query, k=settings.sparse_candidates, doc_ids=doc_ids)
    return _rrf(dense, sparse, settings.rrf_k)[:k]
