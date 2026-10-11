"""The plain (non-agent) RAG path.

Retrieve -> stuff the numbered sources into the prompt -> ask the model to
answer with [n] markers -> map the markers back to chunks so the UI can show
a real citation rather than a bare number.
"""

import re
import time

from app import llm
from app.config import settings
from app.models import Answer, Citation
from app.prompts import GROUNDED_SYSTEM
from app.retrieval.search import search

_MARKER = re.compile(r"\[(\d+)\]")


def build_context(chunks) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] {chunk.doc_title} (p. {chunk.page})\n{chunk.text}")
    return "\n\n".join(blocks)


def extract_citations(text: str, chunks) -> list[Citation]:
    """Map [n] markers back onto the source list.

    Markers the model invents (n > len(chunks)) are dropped - they are always
    a sign it tried to cite something it was not given.
    """
    seen: dict[int, Citation] = {}
    for match in _MARKER.finditer(text or ""):
        n = int(match.group(1))
        if n < 1 or n > len(chunks) or n in seen:
            continue
        chunk = chunks[n - 1]
        seen[n] = Citation(
            marker=n,
            doc_id=chunk.doc_id,
            doc_title=chunk.doc_title,
            page=chunk.page,
            snippet=chunk.text[:280].strip(),
        )
    return [seen[k] for k in sorted(seen)]


def strip_unknown_markers(text: str, chunks) -> str:
    def replace(match):
        n = int(match.group(1))
        return match.group(0) if 1 <= n <= len(chunks) else ""

    return _MARKER.sub(replace, text or "")


def grounded_messages(question: str, chunks, notes: str | None = None) -> list[dict]:
    """The prompt shape shared by the RAG path and the agent's writer step.

    Both go through this so a citation-format change lands in one place.
    """
    user = f"Sources:\n\n{build_context(chunks)}\n\nQuestion: {question}"
    if notes and notes.strip():
        user += (
            "\n\nAnalyst notes gathered while searching (structure the answer with "
            "these, but every fact still needs one of the numbered sources above "
            "and its marker):\n"
            + notes.strip()[:2000]
        )
    return [
        {"role": "system", "content": GROUNDED_SYSTEM},
        {"role": "user", "content": user},
    ]


# Used only when the first attempt comes back with no markers at all. The eval
# showed this happening on roughly 3 in 19 questions, always on broad summary
# or comparison questions where the model writes prose and forgets the format.
# Re-asking once with the failure named explicitly fixes most of them, and it
# costs nothing on the answers that were already fine.
_CITATION_RETRY = (
    "That answer had no citation markers. Rewrite it so that every factual "
    "claim ends with the marker of the source it came from, like [2]. "
    "Do not add any fact that is not in the sources above."
)


def write_cited_answer(
    question: str,
    chunks,
    notes: str | None = None,
) -> tuple[str, int, dict]:
    """Write the final answer, retrying once if it comes back uncited.

    Returns (text, attempts, usage).
    """
    messages = grounded_messages(question, chunks, notes=notes)
    response = llm.chat(messages)
    text = llm.content_of(response)
    usage = llm.usage_of(response)

    if _MARKER.search(text or ""):
        return text, 1, usage

    retry = messages + [
        {"role": "assistant", "content": text or "(empty)"},
        {"role": "user", "content": _CITATION_RETRY},
    ]
    second_response = llm.chat(retry)
    second = llm.content_of(second_response)
    usage = llm.merge_usage(usage, llm.usage_of(second_response))

    if _MARKER.search(second or ""):
        return second, 2, usage
    return text, 2, usage


def answer(
    question: str,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
) -> Answer:
    started = time.perf_counter()

    chunks = search(question, k=top_k or settings.top_k, doc_ids=doc_ids)

    if not chunks:
        return Answer(
            question=question,
            answer=(
                "I could not find anything relevant in the uploaded papers. "
                "Either nothing has been uploaded yet, or the question is outside them."
            ),
            mode="rag",
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    text, _attempts, usage = write_cited_answer(question, chunks)

    # the retry above usually fixes this, but not always; don't silently
    # pretend an uncited answer is sourced
    if not _MARKER.search(text):
        text = text + "\n\n(No citation markers were produced for this answer.)"

    latency = round((time.perf_counter() - started) * 1000, 1)
    return Answer(
        question=question,
        answer=strip_unknown_markers(text, chunks),
        citations=extract_citations(text, chunks),
        sources=chunks,
        mode="rag",
        latency_ms=latency,
        usage=usage,
    )
