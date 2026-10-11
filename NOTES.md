# Build notes

Working log, newest last. Mostly here so I don't re-litigate the same
decisions later.

## Why the whole thing is local

I started out planning to call a hosted model for generation and a hosted
embedding endpoint for indexing. Two things pushed me off it. First, the
embedding cost is paid on every ingest and grows with the corpus, which makes
tinkering with chunk sizes annoying. Second, and more importantly, I wanted
the eval numbers to be reproducible: a hosted model can change under you
between runs and then the numbers in the README are meaningless.

So: Ollama for generation, sentence-transformers for embeddings, Chroma
persisted to disk. Everything runs on CPU. The cost is latency, and the
latency is genuinely bad on a 3B model (see the eval section in the README).

## Chunking: sentences, not character windows

First version sliced every 900 characters. It cut mid-sentence constantly and
the retriever got fragments like "...which we attribute to the questions
needing evidence from two documents that share no vocabu". Recall on the
sample corpus was noticeably worse. Rewrote it to pack whole sentences up to
the character budget, with the overlap taken from the tail of the previous
chunk snapped forward to a sentence boundary. More code, better results.

Still character-based rather than token-based, which means the effective
chunk size differs between models. Noted as a limitation rather than fixed.

## Retrieval: keeping BM25 around

Dense-only retrieval could not find "k1 = 1.2" or "MultiHopLit F1 52.7". Those
questions have one decisive rare token and the dense encoder pools it away.
Adding BM25 and fusing with RRF fixed it.

I used RRF rather than score blending on purpose. Cosine distances are
bounded, BM25 scores are not, and normalising per query to blend them was
more fragile than it looked - the same weight did not work across
collections. RRF only cares about rank, and it has one parameter which
turned out to be flat over a wide range, so it is hard to overfit.

## The agent: seeding a retrieval first

The graph is seed -> agent -> act -> agent, with a finalize node for when the
step cap is hit.

The seed node exists because of a failure I kept hitting: on step one the
model would answer a paper question straight from memory without calling
search. The answer looked plausible and was completely unsourced, which is
the worst possible failure mode for this kind of tool. Seeding one retrieval
for the raw question before the loop means the worst case - the model never
calls a tool at all - is still grounded in real passages. It is a bit of a
cheat; the "agent" is never truly starting from nothing.

Source markers are numbered globally across the whole run, not per search.
The model can cite [7] from a search three steps earlier, and the finalizer
resolves everything against one flat list. Per-search numbering meant the
model would cite [1] from the wrong search.

The step cap (6) is about latency more than quality. Without it a confused
run searches the same thing repeatedly.

## Calculator

Deliberately not `eval()`. It walks the AST and only permits numeric literals
and arithmetic operators. The model writes expressions like
`(72.1 - 64.4) / 64.4 * 100` and this evaluates them. Answering "how many
percent better is A than B" needs real arithmetic, and doing it through a tool
is checkable - you can see the expression in the trace.

Floats are rounded to 6 places before returning, otherwise the model reports
0.30000000000000004 for 0.1 + 0.2.

## Eval: page-level ground truth, and a metric bug

Gold answers are (document, page) pairs, not chunk ids, because chunk ids
change whenever the chunk size changes and then nothing is comparable across
runs. Page level is also what the citations actually promise.

The number-matching metric was wrong on first write. My regex was
`-?\d+(?:\.\d+)?`, which happily matches the "1" in "F1" and splits "9,412"
into "9" and "412". That punished correct answers for mentioning a metric
name and under-counted thousands. Rewrote it with lookarounds so digits
inside identifiers are ignored and comma-grouped numbers stay intact. There
are tests for both cases now.

## The agent was worse than the plain RAG path, and the eval caught it

This is the one I would actually talk about, because I did not see it coming.

The first version of the agent returned its own last message as the answer.
Ran the eval and the agent lost to plain RAG on almost everything:

| | citation precision | token F1 | LLM judge |
|---|---|---|---|
| rag | 0.84 | 0.53 | 0.85 |
| agent (first version) | 0.21 | 0.26 | 0.71 |

The cause was dumber than it sounds. On 15 of the 19 questions the agent's
final message contained *zero* citation markers. The answers were often
factually correct - it had found the right numbers - but it had stopped
obeying the "cite every claim" instruction. It was six messages deep into a
tool loop and by then the citation rule from the system prompt had lost.

There was a second, worse problem in the same run. One question took **427
seconds**. The model had generated 41,053 tokens for a single answer - it just
never stopped. Nothing capped `num_predict`, so a confused generation ran
until the model happened to emit an end token.

Both fixes:

1. Set `num_predict`. 450 tokens for answers, 250 for the judge and extractor
   helpers. An uncapped model is not a slow system, it is a hung one.
2. Split gathering from writing. The agent's tools now only collect evidence;
a separate `write` node produces the final answer using the same prompt as the
RAG path, and the agent's own draft is passed in as notes. The writer cannot
see the tool loop, so nothing distracts it from the citation format.

Then a third pass, because 3 of 19 answers were *still* coming back uncited:
if the writer produces no markers, re-ask it once with the failure named
explicitly. That is in `write_cited_answer` and it costs nothing on answers
that were already fine.

Agent mode, first version -> final:

| | uncited answers | citation precision | citation recall | LLM judge | max latency |
|---|---|---|---|---|---|
| first version | 15/19 | 0.21 | 0.14 | 0.71 | 427 s |
| final | 0/19 | 0.76 | 0.73 | 0.85 | 9.2 s |

Lesson I am taking from this: the agent's final message is a bad place to put
your output contract. Whatever the model was doing during the loop, it will
still be doing it when it writes the answer. Make the last step a separate,
boring, heavily-constrained call.

## Things I hit and fixed

- Chroma raises `NotFoundError` when you delete a collection that does not
  exist, so `store.reset()` blew up on a fresh checkout. Wrapped it.
- `import fitz` is deprecated in current PyMuPDF; switched to `import pymupdf`.
- The Streamlit UI calls the API over HTTP instead of importing the services.
  Two processes to run, but it keeps the front end honest - if the API
  contract is broken I find out immediately rather than by importing around
  it.
- Scanned PDFs produce no text and ingest "succeeds" with zero chunks. The
  upload route now treats zero chunks as an error so it fails loudly instead
  of indexing nothing. OCR is not implemented.
