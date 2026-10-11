from app.ingest import chunking


def test_clean_joins_hyphenated_line_breaks():
    assert "information" in chunking.clean("informa-\ntion retrieval")
    assert chunking.clean("a  b\tc") == "a b c"


def test_clean_replaces_ligatures():
    assert chunking.clean("bene\ufb01t") == "benefit"


def test_chunks_respect_size_roughly():
    text = " ".join(f"Sentence number {i} about retrieval." for i in range(200))
    chunks = chunking.chunk_text(text, size=400, overlap=0)
    assert len(chunks) > 3
    # packing is sentence based so a chunk can overshoot by one sentence
    assert all(len(c) < 400 + 80 for c in chunks)


def test_overlap_carries_context_forward():
    text = " ".join(f"Sentence {i} with some filler words here." for i in range(120))
    without = chunking.chunk_text(text, size=500, overlap=0)
    with_overlap = chunking.chunk_text(text, size=500, overlap=200)
    assert len(with_overlap) >= len(without)
    # the tail of chunk 0 should reappear near the head of chunk 1
    tail_words = without[0].split()[-4:]
    assert " ".join(tail_words) in with_overlap[1]


def test_no_empty_chunks():
    chunks = chunking.chunk_text("Short text.", size=900, overlap=160)
    assert chunks == ["Short text."]
    assert chunking.chunk_text("", size=900, overlap=160) == []


def test_monster_sentence_is_hard_split():
    text = "x" * 5000
    chunks = chunking.chunk_text(text, size=900, overlap=0)
    assert len(chunks) >= 5
    assert all(len(c) <= 900 for c in chunks)


def test_chunk_pages_keeps_page_numbers_and_running_index():
    pages = [
        {"page": 3, "text": " ".join(f"word{i}" for i in range(400))},
        {"page": 4, "text": " ".join(f"word{i}" for i in range(400))},
    ]
    chunks = chunking.chunk_pages(pages, size=300, overlap=0)
    assert [c["index"] for c in chunks] == list(range(len(chunks)))
    assert {c["page"] for c in chunks} == {3, 4}
    assert all(c["text"].strip() for c in chunks)
