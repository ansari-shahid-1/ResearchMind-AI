# ResearchMind AI

Upload a set of research papers, ask questions across all of them, and get
answers with page-level citations that you can check.

The interesting part is not "chat with PDF". It is that a lot of questions I
actually want to ask a pile of papers cannot be answered by retrieving once.
*"Which of these methods does best on multi-hop questions, and how much better
is it?"* needs a search per method, a comparison, and some arithmetic. A
single-shot RAG pipeline will retrieve passages about one method, miss the
other entirely, and then either hallucinate the comparison or refuse.

So this is built as a retrieval system plus a small agent that can decide for
itself how many searches it needs, do its own arithmetic through a tool, and
pull a specific field off a specific page. Everything runs locally on CPU:
Ollama for generation, sentence-transformers for embeddings, Chroma on disk.

## What a run actually looks like

```
$ python scripts/ask.py --mode agent --trace \
    "Which method gets the best answer F1 on MultiHopLit, and by how much does it beat SDF?"

--- steps ---
1. seed retrieval -> 6 chunks
2. tool_call ['search_papers']
3. search_papers({'k': 1, 'query': 'which method gets the best answer F1 on MultiHopLit'}) -> 1 chunks, 14.4ms
4. tool_call ['calculator']
5. calculator({'expression': '52.7 - 41.6'}) -> 0 chunks, 0.0ms
6. answer

QDML (ours) gets the best answer F1 on MultiHopLit with a score of 52.7, which
is 11.1 points higher than the Sparse-Dense Fusion (SDF) method, which has a
score of 41.6. [7]

--- sources ---
[7] Query Decomposition for Multi-Hop Literature Search p.4
```

Two things worth pointing at. The arithmetic went through the calculator
rather than the model's head, so it is visible and checkable in the trace. And
the citation is `p.4`, which is the page of the results table - not the page
that happens to embed closest.

## How it works

Ingest is a straight pipeline:

```
PDF --> per-page text --> sentence-packed chunks --> embeddings --> Chroma
                                       (page number kept on every chunk)
```

Retrieval is hybrid. A dense query and a BM25 query run independently and are
fused with reciprocal rank fusion. This matters more than I expected: dense-only
retrieval could not find "k1 = 1.2" or "52.7 F1" because the encoder pools the
one decisive rare token away, and those are exactly the questions people ask of
papers.

Answering has two modes:

- `rag` - retrieve once, stuff the numbered sources into a prompt, answer with
  markers, map the markers back to chunks.
- `agent` - a LangGraph state machine: `seed -> agent -> act -> agent`, with a
  `finalize` node for when the step cap is hit. The model picks from four
  tools and can loop as many times as it needs.

The tools:

| Tool | What it is for |
|---|---|
| `search_papers` | hybrid retrieval, returns numbered citable passages |
| `calculator` | arithmetic on the AST. Not `eval()` - numeric literals and operators only |
| `extract_field` | pull one named field off one specific page, as JSON |
| `list_documents` | what is in the corpus |

## Decisions I would defend

**One retrieval is seeded before the agent runs.** Without it a 3B model
answers paper questions from memory on step one, which looks fine and is
completely unsourced. Seeding means the worst case is still grounded. It is a
small cheat - the agent never truly starts from nothing.

**Source markers are numbered globally, not per search.** The model can cite
`[7]` from a search it ran three steps ago and the finalizer resolves
everything against one flat list. Per-search numbering had the model citing
`[1]` from the wrong search.

**Ground truth in eval is (document, page), not chunk id.** Chunk ids change
whenever chunk size changes, which would make runs incomparable. Page level is
also what the citation actually promises to the user.

**Citation precision is measured, not just assumed.** A citation pointing at a
page that does not contain the answer is a fabricated source even when the
answer is right, so it gets its own metric.

**The Streamlit UI talks to the API over HTTP.** Two processes to run, but the
front end cannot accidentally import around a broken API contract.

Longer version of all of this, including what went wrong, is in
[NOTES.md](NOTES.md).

## Evaluation

`eval/dataset.jsonl` has 19 questions over three sample papers, tagged
single-hop, comparison or unanswerable, each with a gold answer and gold
pages. Two are deliberately out of scope to check that the system refuses
rather than invents.

```
$ python -m eval.run --modes rag agent
```

Writes `eval/results/<timestamp>.json` (raw, per question) and a markdown
report next to it. The raw file records chunk size, models and `top_k`, because
a score without those is not comparable to anything.

Those runs are committed under `eval/results/`, including the early ones that
scored badly and the small smoke runs. The ranges quoted below are read off
these files - a number I cannot point at is not worth quoting. Files are never
overwritten, so a re-run adds a new timestamp instead of erasing the previous
state.

Measured with `qwen2.5:3b` on CPU, chunk 900/160, `top_k` 6. These are ranges
over the runs of the current code, not one run - a single run is not a
measurement when the judge is a 3B model.

| Metric | `rag` | `agent` | Holds every run? |
|---|---|---|---|
| Recall@6 | 0.951 | 0.951 | identical by construction (see below) |
| Precision@6 | 0.284 | 0.284 | identical by construction |
| MRR | 0.956 | 0.956 | identical by construction |
| Numeric match | 0.793-0.805 | 0.832-0.859 | **yes - agent ahead every run** |
| Token F1 | 0.549-0.553 | 0.533-0.563 | no |
| LLM judge | 0.81-0.90 | 0.86-0.92 | no - the ordering flips |
| Citation precision | 0.818-0.824 | 0.745-0.853 | no - the ordering flips |
| Citation recall | 0.696-0.716 | 0.686-0.706 | no |
| Latency p50 | 0.59-0.76 s | 3.08-3.36 s | **yes - agent ~5x slower** |

Both modes cite on all 19 questions - no answer came back uncited.

What I am actually willing to claim:

- **The calculator earns its keep.** Numeric match is the one accuracy metric
  the agent wins in every run, and it is the only one with a mechanical
  explanation instead of a model-vibe one: the arithmetic happens in Python,
  not in the model.
- **The agent is about 5x slower** at p50, in every run, in exchange for a
  judge and citation difference that sits inside the noise.
- **Everything else is close to a coin flip.** The judge ranking between `rag`
  and `agent` reversed between two runs of identical code. With 19 questions
  and a 3B judge, the noise band swallows most of the interesting differences,
  and I would rather say that than quote the run that flattered the agent.
- **The retrieval rows are not a comparison.** The agent is seeded with the
  same first retrieval and the metric scores the top-6 of the accumulated
  list, which for the agent *is* the seed. It measures the shared first stage,
  not the searches the agent runs afterwards. Scoring best-of-all-searches
  would be a more flattering metric and a less honest one.
- **Only 4 of the 19 questions are multi-hop or comparison** - the case the
  agent exists for. That is nowhere near enough to conclude anything, and it is
  the first thing I would fix.

The `rag` max latency is dominated by a single cold-start answer (Ollama loading
the model on the first question of the run); its p50 is under a second.

The numbers with real signal in them are the before/after ones in
[NOTES.md](NOTES.md). The first version of the agent scored 0.21 citation
precision, left 15 of 19 answers with no citations at all, and once took 427
seconds on a single question. None of that was visible from reading the code.

Metrics: `recall@k` and `precision@k` over gold (doc, page) pairs, MRR, token
F1 against the gold answer, a strict-number match, citation precision and
recall, an LLM judge score, and latency.

Read the numbers honestly. Recall is high because the sample corpus is three
papers and 25 chunks - this is a pipeline test, not a benchmark.

If I could keep only one metric it would be citation precision. It is the only
one that catches a failure mode token F1 and the judge both reward: an answer
that is right and cites a page that does not contain it. But as the table
shows, even it is noisy at 19 questions, so I have stopped treating any single
digit here as a result.

## Setup

Needs Python 3.11+ and [Ollama](https://ollama.com) running.

```bash
ollama pull qwen2.5:3b

python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Windows
# .venv/bin/python -m pip install -r requirements.txt           # macOS/Linux

python scripts/make_sample_papers.py     # three small synthetic papers
python scripts/ingest.py data/sample_papers --reset
```

First ingest downloads the embedding model (~90 MB) and takes a minute. Every
run after that is fast.

## Usage

Command line:

```bash
python scripts/ingest.py mypaper.pdf                     # single file
python scripts/ingest.py ~/papers --title "..."          # a directory
python scripts/ask.py "what is reciprocal rank fusion?" -m rag
python scripts/ask.py --trace "compare the recall of SDF and CRCE"
```

API and UI (both, in two terminals):

```bash
uvicorn app.main:app --reload            # http://localhost:8000/docs
streamlit run ui/streamlit_app.py        # http://localhost:8501
```

The UI has an Evaluation tab that reads the latest `eval/results` run, so you
can see the metrics next to the answers they came from.

```
POST   /documents        multipart upload, deduplicated by content hash
GET    /documents
DELETE /documents/{id}
POST   /documents/{id}/reindex
POST   /query            {question, mode, doc_ids?, top_k?}
GET    /health           includes an Ollama reachability check
```

Config is environment variables with a `RESEARCHMIND_` prefix; see
[.env.example](.env.example). Chunk size is the one to be careful with - change
it and you have to re-ingest, or the index mixes old and new chunk boundaries.

Tests:

```bash
python -m pytest          # 96 tests, under 5 seconds, no model downloads
```

## Limitations

- **CPU latency is bad.** A 3B model on CPU is around 15-25 s per agent
  question. Warm RAG answers are under a second. The step cap exists mostly to
  bound this. A larger model on a GPU would change the picture.
- **No OCR.** Scanned PDFs yield no text; the upload route rejects them with
  zero chunks rather than silently indexing nothing.
- **Chunking is character-based, not token-based**, so effective chunk size
  shifts with the model. Works, but it is a wart.
- **Citation precision is not 100%.** I strip markers the model invents (any
  `[n]` beyond the number of sources it was given), but I do not verify that
  the passage on a cited page actually supports the claim. That check would
  need an entailment model.
- **Eval is small and on a synthetic corpus.** The numbers say the plumbing
  works, nothing more. Points 1-5 of the dataset are the ones to look at for
  retrieval quality.
- **No auth, no tenancy.** It is a single-user local tool.

## Layout

```
app/
  config.py          settings, one place for every tunable
  models.py          pydantic schemas shared by API and services
  llm.py             Ollama wrapper, incl. forgiving JSON parsing
  prompts.py         every prompt, so they can be diffed
  qa.py              plain RAG path + citation parsing
  registry.py        sqlite document registry
  ingest/
    pdf.py           PDF -> page text
    chunking.py      sentence-aware packing with page metadata
    pipeline.py      upload -> chunks -> embeddings -> Chroma
  retrieval/
    embeddings.py    sentence-transformers, loaded once per process
    store.py         Chroma wrapper, revision counter
    search.py        dense + BM25 + reciprocal rank fusion
  agent/
    tools.py         the four tools, incl. the AST calculator
    graph.py         the LangGraph loop
  api/routes.py      HTTP surface
eval/
  dataset.jsonl      questions with gold answers and pages
  metrics.py         pure metric functions
  run.py             runner + report writer
ui/streamlit_app.py
scripts/             ingest, ask, sample-paper generator
tests/
```
