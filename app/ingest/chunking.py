"""Sentence-aware packing.

Most chunks I saw in tutorials just slice every N characters, which cuts
mid-sentence and gives the retriever garbage to embed. Packing whole
sentences up to a character budget costs a few lines more and noticeably
helps recall, so that's what this does. Overlap is copied from the tail of
the previous chunk so an answer split across a boundary can still be found.
"""

import re

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"[ \t]+")
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}


def clean(text: str) -> str:
    """Tidy PDF extraction noise: hyphenated line breaks, stray whitespace."""
    text = text.replace("\u00ad", "")
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    # "informa-\ntion" -> "information"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    text = _WS.sub(" ", text)
    return text.strip()


def sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENTENCE.split(clean(text))) if s]


def chunk_text(text: str, size: int = 900, overlap: int = 160) -> list[str]:
    """Pack sentences into chunks of roughly `size` characters."""
    sents = sentences(text)
    if not sents:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for sent in sents:
        # a single monster sentence (tables, references) gets hard-split
        if len(sent) > size * 2:
            if current:
                chunks.append(" ".join(current))
                current, length = [], 0
            for i in range(0, len(sent), size):
                chunks.append(sent[i : i + size])
            continue

        if length + len(sent) + 1 > size and current:
            chunks.append(" ".join(current))
            carry = _tail(" ".join(current), overlap)
            current = [carry] if carry else []
            length = len(carry)

        current.append(sent)
        length += len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def _tail(text: str, overlap: int) -> str:
    """Take the last ~overlap chars, snapped to a sentence start if we can."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    dot = tail.find(". ")
    if 0 <= dot < len(tail) - 20:
        tail = tail[dot + 2 :]
    return tail.strip()


def chunk_pages(pages: list[dict], size: int = 900, overlap: int = 160) -> list[dict]:
    """[{page, text}] -> [{page, index, text}]. Index is per-document."""
    out = []
    counter = 0
    for page in pages:
        for text in chunk_text(page["text"], size, overlap):
            out.append({"page": page["page"], "index": counter, "text": text})
            counter += 1
    return out
