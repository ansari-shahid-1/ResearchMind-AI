"""Generate three small sample papers as PDFs.

The point of these is that I know exactly what is in them, including which
page each number is on, so the eval set can have real ground truth instead of
me eyeballing whether an answer "looks right". They are clearly synthetic -
swap in real papers from arXiv for a demo, but keep these for evaluation.

    python scripts/make_sample_papers.py
"""

from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_papers"

# Each paper: (title, [page1, page2, ...]) where a page is a list of paragraphs.
# Page index + 1 is the page number the citations will use, so the ordering
# here IS the ground truth for the eval questions.

SDF = (
    "Sparse-Dense Fusion Retrieval for Scientific Question Answering",
    [
        # page 1
        [
            "Sparse-Dense Fusion Retrieval for Scientific Question Answering",
            "A. Ramanathan, J. Okonkwo, M. Lindqvist - Institute for Document Computing",
            "Abstract. Retrieving evidence from scientific literature is harder than web "
            "question answering because relevant passages are long, terminology is dense, "
            "and a query term that appears once on a page is often the decisive one. We "
            "present SDF (Sparse-Dense Fusion), a retriever that runs BM25 and a dense "
            "bi-encoder in parallel and combines them with reciprocal rank fusion. On "
            "SciQA-9k, SDF reaches 78.4 Recall@20 and 61.2 answer F1, improving over the "
            "dense-only baseline by 6.1 F1 points. The gains are largest on questions "
            "whose answers hinge on a rare identifier such as a gene name or a model "
            "abbreviation, which is exactly where dense retrieval alone tends to fail.",
            "1. Introduction. Dense retrieval has become the default first stage for "
            "open-domain question answering. A dual encoder maps the question and each "
            "passage into a shared space and nearest-neighbour search returns candidates. "
            "This works well when the question and the passage use similar wording. "
            "Scientific questions violate that assumption constantly. A question about "
            "'the FiD architecture' may need a passage that only ever says 'fusion-in-"
            "decoder', and a question about a specific numeric result may need the one "
            "table row where that number appears. Under those conditions the dense "
            "encoder's smooth semantic matching works against it: rare but decisive "
            "terms get washed out by the pooled representation.",
        ],
        # page 2
        [
            "2. Related Work. Hybrid retrieval is not new. Prior systems have combined "
            "lexical and dense scores by linear interpolation, with a weight tuned per "
            "dataset. The difficulty is that BM25 scores are unbounded and depend on "
            "corpus statistics while cosine similarities are bounded, so the "
            "interpolation weight does not transfer between collections. We instead use "
            "reciprocal rank fusion, which only looks at the rank of a document in each "
            "list. RRF has one free parameter, k, and we find performance is flat for k "
            "between 40 and 80, which makes it far easier to set honestly than a "
            "similarity threshold.",
            "Reranking has been the other standard remedy. A cross-encoder scores each "
            "candidate against the query jointly and is much more accurate than the "
            "bi-encoder, but costs a forward pass per candidate. Prior work applies "
            "reranking to the top 100 candidates, which dominates latency. We follow "
            "that recipe only in the ablation, and report a cheaper cascade in a "
            "companion paper.",
        ],
        # page 3
        [
            "3. Method. SDF is a two-stage retriever with a fused first stage and no "
            "learned reranker. The sparse component is BM25 with k1 = 1.2 and b = 0.75 "
            "using a whitespace tokenizer with a 34-word stop list. The dense component "
            "is a bi-encoder initialised from a general-purpose sentence model and "
            "fine-tuned for two epochs on 120k question-passage pairs mined from SciQA-9k "
            "using BM25 hard negatives. Embeddings are L2-normalised and indexed with "
            "an HNSW graph using cosine distance, ef_construction = 200 and M = 32.",
            "At query time we retrieve 20 candidates from each component and fuse with "
            "RRF at k = 60, keeping the top 20 for evaluation. Because both components "
            "are independent, the two lookups run concurrently and the fused stage adds "
            "roughly 3 milliseconds on top of the slower of the two. We deliberately do "
            "not use document-level filtering at retrieval time; long scientific "
            "documents are chunked into overlapping windows of about 900 characters "
            "with a 160 character overlap, and each window keeps its page number so a "
            "citation can point at the page a human would turn to.",
        ],
        # page 4
        [
            "4. Experiments. We evaluate on three collections. SciQA-9k contains 9,412 "
            "questions over 41,000 open-access papers. BioASQ-Lite is a 2,100 question "
            "biomedical set. MultiHopLit is 730 hand-written two-hop questions that "
            "require evidence from two different papers. All systems are evaluated with "
            "Recall@20 and answer F1 using the same reader, so differences reflect "
            "retrieval only.",
            "Table 1. Main results. Recall@20 and answer F1 (%).",
            "System | SciQA-9k R@20 | SciQA-9k F1 | BioASQ R@20 | BioASQ F1 | MultiHopLit F1",
            "BM25 only | 71.2 | 53.9 | 66.8 | 50.1 | 34.2",
            "Dense only | 72.1 | 55.1 | 69.4 | 52.6 | 33.8",
            "SDF (ours) | 78.4 | 61.2 | 74.9 | 57.3 | 41.6",
            "SDF + cross-encoder rerank | 79.1 | 62.4 | 75.6 | 58.0 | 44.9",
            "The fused retriever improves over dense-only by 6.1 F1 on SciQA-9k and "
            "4.7 F1 on BioASQ. Cross-encoder reranking adds 1.2 F1 for 118 ms of extra "
            "latency per query, which we consider a poor trade outside of offline use. "
            "On MultiHopLit all single-stage retrievers are weak, which we attribute to "
            "the questions needing evidence from two documents that share no "
            "vocabulary.",
        ],
        # page 5
        [
            "5. Analysis. We split SciQA-9k by whether the gold passage contains a "
            "token that appears fewer than five times in the whole collection. On the "
            "rare-token subset, which is 38% of questions, dense-only F1 drops to 44.7 "
            "while SDF holds at 57.9. On the frequent-token subset the two are within "
            "1.0 F1 of each other. This supports our hypothesis that the fusion gain "
            "comes almost entirely from lexical recall of rare identifiers, and suggests "
            "that on domains without such identifiers hybrid retrieval buys little.",
            "6. Conclusion. SDF is a simple, well-behaved hybrid retriever with one "
            "tunable parameter and no training beyond the dense adapter. It improves on "
            "dense-only retrieval by 6.1 F1 on scientific question answering and the "
            "gain is concentrated on questions with rare identifying terms. The "
            "remaining weakness is multi-hop questions, which single-stage retrievers of "
            "any kind handle poorly here. Future work should target decomposition rather "
            "than better first-stage ranking.",
        ],
    ],
)

CRCE = (
    "Cascade Reranking with Compact Cross-Encoders for Latency-Bounded Retrieval",
    [
        [
            "Cascade Reranking with Compact Cross-Encoders for Latency-Bounded Retrieval",
            "P. Almeida, R. Bhattacharya, S. Nkemelu - Department of Information Retrieval",
            "Abstract. Cross-encoder reranking is the most reliable way to improve the "
            "precision of a retrieval pipeline, but a single large cross-encoder over "
            "100 candidates costs well over 100 ms per query and makes interactive "
            "search impractical. We present CRCE, a cascade in which a 6-layer compact "
            "cross-encoder scores 60 candidates and a full-size cross-encoder rescores "
            "only the surviving 15. On SciQA-9k the cascade reaches 74.1 Recall@20 and "
            "63.8 answer F1 while adding 12 ms to the median query, against 118 ms for "
            "the single-stage reranker. We show that the cascade recovers 92% of the F1 "
            "improvement of full reranking at 10% of the latency cost.",
            "1. Introduction. The accuracy-versus-latency trade-off in reranking is "
            "usually presented as a choice between reranking and not reranking. We argue "
            "it is a budget allocation problem. Most candidates are easy: they are "
            "either clearly relevant or clearly not, and a small model separates them. "
            "Only a handful of candidates near the decision boundary need the expensive "
            "model. A cascade spends the expensive computation exactly where it "
            "changes the outcome.",
        ],
        [
            "2. Related Work. Cascaded ranking is standard in industrial search, where "
            "a cheap first stage filters millions of documents down to hundreds and "
            "progressively richer models narrow further. Applying the same idea inside "
            "the reranking stage is less common in the literature, largely because "
            "published reranking results report accuracy at a fixed budget of 100 "
            "candidates rather than reporting the accuracy-latency curve. We report the "
            "curve.",
            "Compact cross-encoders have been explored for on-device search. Those "
            "systems distil a large teacher into a small student and accept a moderate "
            "accuracy loss. We take a different route: rather than replacing the large "
            "model, we use the small model as a filter and keep the large model for the "
            "ambiguous cases. No distillation is required, and the large model's "
            "judgements are still available where they matter.",
        ],
        [
            "3. Method. The cascade has three stages. Stage one is the same hybrid "
            "first stage used by SDF, returning 60 candidates. Stage two is a compact "
            "cross-encoder with 6 transformer layers and hidden size 256, which scores "
            "all 60 in a single batched pass. Stage three takes the top 15 by compact "
            "score and rescores them with a 12-layer cross-encoder, and the final order "
            "comes from the stage-three scores.",
            "The stage-two cut-off of 15 was chosen by sweeping 10, 15, 25 and 40 on a "
            "held-out split. Recall of the gold passage within the surviving set is "
            "98.6% at 15 and 99.1% at 25, so raising the cut-off buys 0.5% of headroom "
            "for a 60% increase in stage-three cost. We also quantise the compact model "
            "to 8-bit integers, which halves its memory footprint with no measurable "
            "change in ordering.",
        ],
        [
            "4. Experiments. We compare against the hybrid first stage alone, against "
            "single-stage full reranking over 100 candidates, and against distilling the "
            "large cross-encoder into the compact one. Latency is measured on a single "
            "CPU core with batch size 16, which is the realistic setting for a "
            "self-hosted service.",
            "Table 1. Accuracy and median latency. Recall@20 and answer F1 (%) on "
            "SciQA-9k; latency in milliseconds per query.",
            "System | R@20 | F1 | Median latency (ms)",
            "Hybrid first stage | 72.9 | 59.4 | 21",
            "Distilled compact reranker | 73.4 | 61.0 | 29",
            "CRCE cascade (ours) | 74.1 | 63.8 | 33",
            "Full cross-encoder, 100 cands | 74.6 | 64.6 | 139",
            "CRCE adds 12 ms over the first stage and reaches within 0.8 F1 of full "
            "reranking, which is 106 ms faster. Distillation is the better choice when "
            "memory is the binding constraint and latency is not; the cascade dominates "
            "when a latency budget exists but no accelerator is available.",
        ],
        [
            "5. Analysis. The cascade helps most on questions where the first stage "
            "ranks the gold passage between 20 and 60. On candidates the first stage "
            "already ranks in the top 10, compact and large models agree 94% of the "
            "time, so the expensive stage is doing almost nothing. This is the "
            "quantitative version of the intuition that most candidates are easy: "
            "95.8% of the stage-two budget is spent on candidates where the large model "
            "confirms rather than corrects.",
            "6. Conclusion. Reranking accuracy and reranking latency do not have to be "
            "traded off one-for-one. A two-model cascade recovers most of the accuracy "
            "of full reranking at a tenth of the latency, without distillation, "
            "retraining or a GPU. The design generalises to any setting with a "
            "recoverable ranking function, and we are now testing how far the cascade "
            "can be deepened before the stage-two filter becomes the bottleneck.",
        ],
    ],
)

QDML = (
    "Query Decomposition for Multi-Hop Literature Search",
    [
        [
            "Query Decomposition for Multi-Hop Literature Search",
            "T. Ferreira, L. Yamamoto, D. Owusu - Centre for Scholarly Retrieval",
            "Abstract. Many real research questions cannot be answered from a single "
            "passage. 'Which of these two retrieval methods performs better on "
            "multi-hop questions?' requires locating a result for each method and then "
            "comparing them, and a single retrieval query cannot serve both steps. We "
            "present QDML, a decomposition method that rewrites a multi-hop question "
            "into an ordered set of single-hop sub-queries, retrieves evidence for each, "
            "and merges the evidence before answering. On MultiHopLit, QDML reaches "
            "52.7 answer F1 against 41.6 for fused single-pass retrieval and 34.2 for "
            "BM25, a relative improvement of 27% over the strongest single-pass "
            "baseline. The cost is latency: QDML issues a median of 3.4 retrieval calls "
            "per question.",
            "1. Introduction. Multi-hop questions are the hard case for retrieval-"
            "augmented generation. A single dense query is a single point in embedding "
            "space, and a two-part question has no single good point. Systems that "
            "answer these questions by stuffing more passages into the reader tend to "
            "fail for a different reason: the passages that answer part one and part two "
            "are semantically far apart, so the second hop is not retrieved at all. The "
            "fix is not a better retriever but a better query.",
        ],
        [
            "2. Related Work. Multi-hop question answering has mostly been studied with "
            "hand-built entity chains over a knowledge graph, where the next hop is "
            "determined by a relation edge. That framing does not transfer to a "
            "free-text corpus with no entity linking. Iterative retrieval methods do "
            "transfer: retrieve, read, reformulate, retrieve again. These rely on the "
            "reader to produce a useful follow-up query, which requires a capable model "
            "and degrades badly when the first retrieval returns nothing useful.",
            "Our position is that decomposition should happen before retrieval, not "
            "after it. Splitting 'compare A and B on C' into 'A on C' and 'B on C' is a "
            "syntactic operation that a small model performs reliably, whereas judging "
            "whether a retrieved passage justifies a follow-up hop is a semantic "
            "decision that small models get wrong often.",
        ],
        [
            "3. Method. QDML has three components. The decomposer is a prompted model "
            "that takes the question and emits an ordered list of sub-queries, each "
            "annotated with an operation: one of find, compare or compute. We constrain "
            "output to a fixed JSON shape and reject any decomposition producing more "
            "than four sub-queries, because longer plans were almost always the model "
            "padding a question it did not understand.",
            "Each sub-query is retrieved independently with the hybrid retriever from "
            "SDF, keeping 6 passages each. The merger removes duplicate passages, "
            "orders the remainder so that passages supporting the same sub-query are "
            "adjacent, and numbers them for citation. The answerer then writes the "
            "final answer over the merged evidence and is required to cite at least one "
            "source per sub-query, which we enforce by rejecting answers that leave a "
            "sub-query uncited and regenerating once before giving up.",
        ],
        [
            "4. Experiments. We evaluate on MultiHopLit (730 two-hop questions) and on "
            "SciQA-9k as a check that decomposition does not damage single-hop "
            "performance. Latency is end-to-end including decomposition and answering, "
            "measured with the same reader across systems so that only the retrieval "
            "strategy differs.",
            "Table 1. Results. Answer F1 (%) and median end-to-end latency (seconds).",
            "System | MultiHopLit F1 | SciQA-9k F1 | Retrieval calls | Latency (s)",
            "BM25 single pass | 34.2 | 53.9 | 1.0 | 1.4",
            "Dense single pass | 33.8 | 55.1 | 1.0 | 1.3",
            "SDF fused single pass | 41.6 | 61.2 | 1.0 | 1.7",
            "Iterative reformulation | 44.1 | 60.4 | 2.8 | 4.2",
            "QDML (ours) | 52.7 | 60.9 | 3.4 | 3.9",
            "QDML improves MultiHopLit F1 by 11.1 points over the strongest single-pass "
            "baseline and by 8.6 points over iterative reformulation, while costing less "
            "latency than iterative reformulation despite issuing more retrieval calls, "
            "because decomposition happens once instead of inside the loop. On SciQA-9k "
            "the difference from single-pass is within noise, confirming that "
            "decomposition does not hurt when it is not needed.",
        ],
        [
            "5. Analysis. Decomposition quality, not retrieval quality, is the "
            "bottleneck. On the 22% of questions where the decomposer produced an "
            "incomplete plan, F1 drops to 29.4; on the remainder it is 61.3. Manual "
            "inspection of the failures shows a consistent pattern: questions that "
            "require an implicit comparison ('is the effect larger for A or B?') are "
            "often rewritten as a single find sub-query, losing the comparison "
            "entirely. Future work should detect comparisons syntactically rather than "
            "relying on the model to notice them.",
            "6. Conclusion. Decomposing a multi-hop question before retrieval is a "
            "cheap and effective alternative to iterating after retrieval. It requires "
            "no new retriever, no training, and it costs latency only in proportion to "
            "the number of hops the question actually contains. The open problem is "
            "reliable detection of implicit multi-hop structure, which is where most "
            "remaining errors live.",
        ],
    ],
)

PAPERS = [SDF, CRCE, QDML]

PAGE_RECT = pymupdf.Rect(56, 56, 540, 786)


def write_paper(title: str, pages: list[list[str]], out_dir: Path) -> Path:
    doc = pymupdf.open()
    for paragraphs in pages:
        page = doc.new_page(width=595, height=842)  # A4
        text = "\n\n".join(paragraphs)
        page.insert_textbox(
            PAGE_RECT,
            text,
            fontsize=9.5,
            fontname="helv",
            lineheight=1.35,
        )
    slug = title.lower().replace(" ", "_")[:40].strip("_")
    path = out_dir / f"{slug}.pdf"
    doc.set_metadata({"title": title, "author": "sample corpus"})
    doc.save(path)
    doc.close()
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for title, pages in PAPERS:
        path = write_paper(title, pages, OUT)
        print(f"wrote {path}  ({len(pages)} pages)")


if __name__ == "__main__":
    main()
