"""Metric functions.

All pure functions over plain data so they can be unit tested without a
vector store or a model. The runner does the IO and calls these.

Ground truth is (document, page) pairs rather than chunk ids. Chunk ids change
every time the chunk size changes, which would make the numbers incomparable
across runs, and page level is what the citations promise anyway.
"""

import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence

_WORD = re.compile(r"[a-z0-9]+")

_ARTICLES = {"a", "an", "the"}


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall((text or "").lower()) if t not in _ARTICLES]


def _key(doc: str, page: int) -> tuple[str, int]:
    return (doc.strip().lower(), int(page))


def gold_keys(gold_docs: Sequence[str], gold_pages: dict[str, Sequence[int]]) -> set:
    """Flatten {doc_title: [pages]} into a set of (doc, page) keys."""
    keys = set()
    for doc in gold_docs:
        for page in gold_pages.get(doc, []):
            keys.add(_key(doc, page))
    return keys


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------

def retrieval_metrics(
    retrieved: Sequence,
    gold: set,
    k: int,
) -> dict:
    """Recall@k, precision@k and MRR over (doc, page) keys.

    A retrieved chunk counts as correct if it is on a gold page of a gold
    document. Several chunks from the same gold page collapse to one hit, so
    precision is not inflated by a long chunked page.
    """
    top = list(retrieved)[:k]
    if not gold:
        return {"recall": None, "precision": None, "mrr": None, "hit": None}

    keys = [_key(c.doc_title, c.page) for c in top]
    hits = [key for key in keys if key in gold]
    unique_hits = set(hits)

    recall = len(unique_hits) / len(gold)
    precision = len(unique_hits) / len(top) if top else 0.0

    mrr = 0.0
    for rank, key in enumerate(keys, start=1):
        if key in gold:
            mrr = 1.0 / rank
            break

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "mrr": round(mrr, 4),
        "hit": bool(unique_hits),
    }


# --------------------------------------------------------------------------
# answer correctness
# --------------------------------------------------------------------------

def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token overlap F1. Cheap, deterministic, no model needed."""
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0

    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return round(2 * precision * recall / (precision + recall), 4)


# A number, but not the digit inside an identifier. "F1", "k1" and "p.4" are
# names, not quantities; "Recall@20" and "9,412" are quantities. The
# lookarounds are what separate those, and getting this wrong made the metric
# punish correct answers for mentioning a metric name.
NUM = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])")


def _numbers(text: str) -> list[str]:
    return [m.replace(",", "") for m in NUM.findall(text or "")]


def numeric_match(prediction: str, reference: str) -> float | None:
    """What fraction of the reference's numbers appear in the prediction.

    Included because a lot of these questions have a numeric answer, and
    token F1 punishes a correct answer for phrasing differently ("52.7" vs
    "52.7 points"). This checks the number itself, which is what matters.
    """
    gold_numbers = _numbers(reference)
    if not gold_numbers:
        return None
    pred_numbers = set(_numbers(prediction))
    hits = sum(1 for n in gold_numbers if n in pred_numbers)
    return round(hits / len(gold_numbers), 4)


# --------------------------------------------------------------------------
# citations
# --------------------------------------------------------------------------

def citation_metrics(citations: Sequence, gold: set) -> dict:
    """Is every citation a real gold source, and is every gold source cited?

    Precision is the honest part: a citation pointing at a page that does not
    contain the answer is a fabricated source even if the answer is right.
    """
    if not citations:
        return {"citation_precision": 0.0, "citation_recall": 0.0, "cited": 0}
    if not gold:
        return {"citation_precision": None, "citation_recall": None, "cited": len(citations)}

    keys = {_key(c.doc_title, c.page) for c in citations}
    correct = keys & gold
    return {
        "citation_precision": round(len(correct) / len(keys), 4),
        "citation_recall": round(len(correct) / len(gold), 4),
        "cited": len(citations),
    }


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------

def mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.fmean(clean), 4) if clean else None


def pct(values: Sequence[float], percentile: float) -> float | None:
    """Nearest-rank percentile. Avoids numpy for four numbers."""
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return round(clean[0], 1)
    # nearest-rank: ceil(P/100 * N), 1-based, then clamped
    index = min(len(clean) - 1, max(0, math.ceil(percentile / 100 * len(clean)) - 1))
    return round(clean[index], 1)


def latency_summary(latencies: Sequence[float]) -> dict:
    clean = [v for v in latencies if v is not None]
    if not clean:
        return {}
    return {
        "mean_ms": round(statistics.fmean(clean), 1),
        "p50_ms": pct(clean, 50),
        "p95_ms": pct(clean, 95),
        "max_ms": round(max(clean), 1),
    }
