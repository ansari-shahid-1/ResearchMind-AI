from eval import metrics


class FakeChunk:
    def __init__(self, doc_title, page):
        self.doc_title = doc_title
        self.page = page


class FakeCitation:
    def __init__(self, doc_title, page):
        self.doc_title = doc_title
        self.page = page


GOLD = metrics.gold_keys(["Paper A"], {"Paper A": [4]})


def test_gold_keys_is_case_insensitive():
    assert GOLD == {("paper a", 4)}


def test_retrieval_perfect_hit():
    result = metrics.retrieval_metrics([FakeChunk("Paper A", 4)], GOLD, k=6)
    assert result["recall"] == 1.0
    assert result["mrr"] == 1.0
    assert result["precision"] == 1.0


def test_retrieval_mrr_reflects_rank():
    chunks = [FakeChunk("Other", 1), FakeChunk("Other", 2), FakeChunk("Paper A", 4)]
    result = metrics.retrieval_metrics(chunks, GOLD, k=6)
    assert result["mrr"] == round(1 / 3, 4)
    assert result["recall"] == 1.0


def test_retrieval_deduplicates_same_gold_page():
    chunks = [FakeChunk("Paper A", 4), FakeChunk("Paper A", 4)]
    result = metrics.retrieval_metrics(chunks, GOLD, k=6)
    assert result["recall"] == 1.0
    # two chunks, one distinct correct page -> precision is not inflated
    assert result["precision"] == 0.5


def test_retrieval_respects_k():
    chunks = [FakeChunk("Other", 1)] * 5 + [FakeChunk("Paper A", 4)]
    assert metrics.retrieval_metrics(chunks, GOLD, k=3)["hit"] is False
    assert metrics.retrieval_metrics(chunks, GOLD, k=10)["hit"] is True


def test_retrieval_with_no_gold_is_undefined():
    result = metrics.retrieval_metrics([FakeChunk("Paper A", 4)], set(), k=6)
    assert result["recall"] is None


def test_token_f1_identical_is_one():
    assert metrics.token_f1("SDF reaches 78.4.", "SDF reaches 78.4.") == 1.0


def test_token_f1_ignores_articles_and_case():
    assert metrics.token_f1("The SDF reaches 78.4", "sdf reaches 78.4") == 1.0


def test_token_f1_partial():
    score = metrics.token_f1("SDF reaches 78.4 recall", "SDF reaches 78.4 recall and 61.2 F1")
    assert 0.5 < score < 1.0


def test_token_f1_disjoint_is_zero():
    assert metrics.token_f1("completely different", "unrelated answer") == 0.0


def test_numeric_match_finds_numbers():
    assert metrics.numeric_match("reaches 78.4 and 61.2", "78.4 recall, 61.2 F1") == 1.0
    assert metrics.numeric_match("reaches 78.4", "78.4 recall, 61.2 F1") == 0.5
    assert metrics.numeric_match("no digits", "no digits here") is None


def test_numeric_match_ignores_digits_inside_identifiers():
    # "F1" is a metric name, not the number 1
    assert metrics.numeric_match("the F1 score", "the F1 score") is None
    assert metrics.numeric_match("k1 = 1.2", "k1 = 1.2") == 1.0


def test_numeric_match_handles_thousands_separators():
    assert metrics.numeric_match("9,412 questions", "9,412 questions") == 1.0
    assert metrics.numeric_match("9412 questions", "9,412 questions") == 1.0


def test_numeric_match_reads_at_prefixed_numbers():
    assert metrics.numeric_match("Recall@20 of 78.4", "Recall@20 of 78.4") == 1.0


def test_citation_precision_and_recall():
    citations = [FakeCitation("Paper A", 4), FakeCitation("Paper B", 9)]
    result = metrics.citation_metrics(citations, GOLD)
    assert result["citation_precision"] == 0.5
    assert result["citation_recall"] == 1.0


def test_citation_metrics_without_citations():
    assert metrics.citation_metrics([], GOLD)["citation_precision"] == 0.0


def test_citation_metrics_are_undefined_without_a_reference():
    # an unanswerable question: citing nothing is correct, so this must not
    # be scored as a precision failure
    result = metrics.citation_metrics([], set())
    assert result["citation_precision"] is None
    assert result["citation_recall"] is None


def test_citation_metrics_undefined_even_if_it_cited_something():
    result = metrics.citation_metrics([FakeCitation("Paper A", 4)], set())
    assert result["citation_precision"] is None
    assert result["cited"] == 1


def test_unanswerable_questions_do_not_drag_down_the_average():
    rows = [
        {"citation_precision": 1.0},
        {"citation_precision": None},
        {"citation_precision": None},
    ]
    assert metrics.mean([r["citation_precision"] for r in rows]) == 1.0


def test_percentiles():
    values = [10, 20, 30, 40, 50]
    assert metrics.pct(values, 50) == 30
    assert metrics.pct(values, 95) == 50
    assert metrics.pct([], 50) is None


def test_latency_summary():
    summary = metrics.latency_summary([100, 200, 300])
    assert summary["mean_ms"] == 200.0
    assert summary["max_ms"] == 300


def test_mean_skips_none():
    assert metrics.mean([1.0, None, 3.0]) == 2.0
    assert metrics.mean([None]) is None
