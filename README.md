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

Run of 2026-10-11, `qwen2.5:3b` on CPU, chunk 900/160, `top_k` 6:

| Metric | `rag` | `agent` |
|---|---|---|
| Recall@6 | 0.951 | 0.951 |
| Precision@6 | 0.284 | 0.284 |
| MRR | 0.956 | 0.956 |
| Token F1 | 0.549 | **0.563** |
| Numeric match | 0.793 | **0.859** |
| LLM judge | 0.811 | **0.917** |
| Citation precision | 0.824 | **0.853** |
| Citation recall | 0.696 | **0.706** |
| Latency mean | **1.97 s** | 4.00 s |
| Latency p50 | **0.59 s** | 3.36 s |
| Latency max | 20.4 s | **7.60 s** |

Both modes cite on all 19 questions - no answer came back uncited.

The honest reading:

- **The retrieval rows are identical by construction and should be ignored as a
  comparison.** The agent is seeded with the same first retrieval, and the
  metric scores the top-6 of the accumulated list, which for the agent *is* the
  seed. So the row measures the shared first stage, not the searches the agent
  runs afterwards. Scoring the agent's best-of-all-searches would be a more
  flattering metric and I did not want to report it.
- **The agent now leads on every accuracy metric and trails only on latency.**
  It is about 5.7x slower at p50 and about 4.5x slower on the mean.
- **Do not trust small margins.** I ran the same configuration three times
  while building this and the LLM judge moved by up to 0.09 between runs, with
  citation precision moving by about 0.06. The agent/RAG judge gap above
  (0.106) is only just outside that noise band, so treat it as "probably real"
  rather than settled. The differences that *are* well outside the noise are
  numeric match, the latency columns, and everything in the before/after table
  in [NOTES.md](NOTES.md). A 3B model as judge on 19 questions is a rough
  instrument.
- **Four of the 19 questions are multi-hop or comparison** - the case the agent
  exists for. That is far too few to conclude anything, and it is why I am not
  claiming the agent is the better mode overall.
- The `rag` max of 20.4 s is a single cold-start answer (Ollama loading the
  model); its p50 is 0.59 s.

What the eval actually earned its keep on: the first version of the agent
scored 0.21 citation precision, left 15 of 19 answers with no citations at all,
and once took 427 seconds on a single question. None of that was visible by
reading the code. Finding it, and fixing it, is written up in
[NOTES.md](NOTES.md).

Metrics: `recall@k` and `precision@k` over gold (doc, page) pairs, MRR, token
F1 against the gold answer, a strict-number match, citation precision and
recall, an LLM judge score, and latency.

Read the numbers honestly. Recall is high because the sample corpus is three
papers and 25 chunks - this is a pipeline test, not a benchmark. The
citation-precision numbers are the ones I trust most, because they catch a
failure mode that token F1 happily rewards.

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
python -m pytest          # 62 tests, a few seconds, no model downloads
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
