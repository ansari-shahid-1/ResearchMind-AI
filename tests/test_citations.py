from app import llm
from app.models import RetrievedChunk
from app.qa import (
    build_context,
    extract_citations,
    strip_unknown_markers,
    write_cited_answer,
)


def chunk(marker, page, title="Paper A"):
    return RetrievedChunk(
        chunk_id=f"c{marker}",
        doc_id="d1",
        doc_title=title,
        page=page,
        text=f"text for source {marker}",
    )


CHUNKS = [chunk(1, 4), chunk(2, 5), chunk(3, 7)]


def test_extract_citations_maps_markers_to_sources():
    citations = extract_citations("SDF scores 78.4 [1] while CRCE reaches 74.1 [3].", CHUNKS)
    assert [c.marker for c in citations] == [1, 3]
    assert [c.page for c in citations] == [4, 7]


def test_extract_citations_deduplicates_repeated_markers():
    citations = extract_citations("See [2] and again [2].", CHUNKS)
    assert len(citations) == 1


def test_extract_citations_drops_invented_markers():
    citations = extract_citations("Made up [9].", CHUNKS)
    assert citations == []


def test_extract_citations_handles_no_markers():
    assert extract_citations("No citation here.", CHUNKS) == []
    assert extract_citations("", CHUNKS) == []


def test_strip_unknown_markers_removes_only_bad_ones():
    text = strip_unknown_markers("good [1] bad [12]", CHUNKS)
    assert "[1]" in text
    assert "[12]" not in text


def fake_chat(responses):
    """Stand in for llm.chat, returning canned replies in order."""
    calls = []

    def _chat(messages, **kwargs):
        calls.append(messages)
        content = responses[min(len(calls) - 1, len(responses) - 1)]
        return {"message": {"content": content}, "eval_count": 5, "prompt_eval_count": 11}

    return _chat, calls


def test_writer_does_not_retry_when_the_answer_is_cited(monkeypatch):
    chat, calls = fake_chat(["the figure is 12.5 [1]"])
    monkeypatch.setattr(llm, "chat", chat)

    text, attempts, usage = write_cited_answer("q", CHUNKS)
    assert attempts == 1
    assert len(calls) == 1
    assert usage["eval_count"] == 5


def test_writer_retries_once_when_no_markers(monkeypatch):
    chat, calls = fake_chat(["an answer with no markers at all", "now with a marker [2]"])
    monkeypatch.setattr(llm, "chat", chat)

    text, attempts, usage = write_cited_answer("q", CHUNKS)
    assert attempts == 2
    assert len(calls) == 2
    assert "[2]" in text
    # usage adds both calls together
    assert usage["eval_count"] == 10
    assert usage["prompt_eval_count"] == 22


def test_writer_keeps_the_first_answer_if_the_retry_also_fails(monkeypatch):
    chat, calls = fake_chat(["first uncited answer", "still uncited"])
    monkeypatch.setattr(llm, "chat", chat)

    text, attempts, _ = write_cited_answer("q", CHUNKS)
    assert attempts == 2
    assert len(calls) == 2
    assert text == "first uncited answer"


def test_writer_passes_notes_through(monkeypatch):
    chat, calls = fake_chat(["cited [1]"])
    monkeypatch.setattr(llm, "chat", chat)

    write_cited_answer("q", CHUNKS, notes="my analysis")
    assert "my analysis" in calls[0][1]["content"]


def test_build_context_numbers_and_labels_sources():
    context = build_context(CHUNKS)
    assert context.startswith("[1] Paper A (p. 4)")
    assert "[3] Paper A (p. 7)" in context
    assert context.count("[") == 3
