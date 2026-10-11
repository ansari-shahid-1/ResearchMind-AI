"""Run the evaluation set and write a report.

    python -m eval.run
    python -m eval.run --modes rag agent --limit 5
    python -m eval.run --no-judge          # skip the LLM judge, faster

Writes eval/results/<timestamp>.json (raw, per question) and a matching
.md report. The raw file records the settings in force, because a score
without the chunk size and model name that produced it is not comparable to
anything.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# allow `python eval/run.py` as well as `python -m eval.run`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import llm  # noqa: E402
from app.agent import graph as agent_graph  # noqa: E402
from app.config import settings  # noqa: E402
from app.prompts import JUDGE_SYSTEM  # noqa: E402
from app.qa import answer as rag_answer  # noqa: E402
from eval import metrics  # noqa: E402

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "results"


def load_dataset(path: Path = DATASET) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def judge(question: str, reference: str, candidate: str) -> dict:
    prompt = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Candidate answer: {candidate}"
    )
    parsed = llm.generate_json(prompt, system=JUDGE_SYSTEM)
    if not isinstance(parsed, dict) or "score" not in parsed:
        return {"score": None, "reason": "judge returned no usable JSON"}
    try:
        score = max(0.0, min(1.0, float(parsed["score"])))
    except (TypeError, ValueError):
        return {"score": None, "reason": f"unparseable score: {parsed.get('score')!r}"}
    return {"score": round(score, 3), "reason": str(parsed.get("reason", ""))[:200]}


def run_item(item: dict, mode: str, k: int, use_judge: bool) -> dict:
    question = item["question"]
    started = time.perf_counter()

    if mode == "agent":
        result = agent_graph.run(question)
    else:
        result = rag_answer(question, top_k=k)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    gold = metrics.gold_keys(item.get("gold_docs", []), item.get("gold_pages", {}))
    retrieval = metrics.retrieval_metrics(result.sources, gold, k)
    citations = metrics.citation_metrics(result.citations, gold)

    row = {
        "id": item["id"],
        "type": item.get("type", "single_hop"),
        "mode": mode,
        "question": question,
        "answer": result.answer,
        "n_sources": len(result.sources),
        "n_citations": len(result.citations),
        "token_f1": metrics.token_f1(result.answer, item.get("gold_answer", "")),
        "numeric_match": metrics.numeric_match(result.answer, item.get("gold_answer", "")),
        "latency_ms": elapsed_ms,
        "reported_latency_ms": result.latency_ms,
        "steps": len(result.steps),
        "usage": result.usage,
        **retrieval,
        **citations,
    }

    if use_judge:
        graded = judge(question, item.get("gold_answer", ""), result.answer)
        row["judge_score"] = graded["score"]
        row["judge_reason"] = graded["reason"]

    row["wall_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return row


def summarise(rows: list[dict]) -> dict:
    def col(name):
        return [r.get(name) for r in rows]

    return {
        "n": len(rows),
        "recall@k": metrics.mean(col("recall")),
        "precision@k": metrics.mean(col("precision")),
        "mrr": metrics.mean(col("mrr")),
        "token_f1": metrics.mean(col("token_f1")),
        "numeric_match": metrics.mean(col("numeric_match")),
        "judge_score": metrics.mean(col("judge_score")),
        "citation_precision": metrics.mean(col("citation_precision")),
        "citation_recall": metrics.mean(col("citation_recall")),
        **metrics.latency_summary(col("latency_ms")),
    }


def _fmt(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_report(rows: list[dict], summaries: dict[str, dict], path: Path, meta: dict) -> None:
    lines = ["# Evaluation report", ""]
    lines.append(f"Run at `{meta['timestamp']}` with `{meta['chat_model']}` and `{meta['embed_model']}`.")
    lines.append("")
    lines.append(
        f"Chunk size {meta['chunk_size']} (overlap {meta['chunk_overlap']}), "
        f"top_k {meta['top_k']}, k for metrics {meta['k']}."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | " + " | ".join(summaries) + " |")
    lines.append("|---|" + "---|" * len(summaries))
    keys = [
        ("n", "questions"),
        ("recall@k", "Recall@k"),
        ("precision@k", "Precision@k"),
        ("mrr", "MRR"),
        ("token_f1", "Token F1"),
        ("numeric_match", "Numeric match"),
        ("judge_score", "LLM judge"),
        ("citation_precision", "Citation precision"),
        ("citation_recall", "Citation recall"),
        ("mean_ms", "Latency mean (ms)"),
        ("p50_ms", "Latency p50 (ms)"),
        ("p95_ms", "Latency p95 (ms)"),
    ]
    for key, label in keys:
        values = []
        for mode in summaries:
            value = summaries[mode].get(key)
            values.append("-" if value is None else f"{value}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines.append("")

    for mode in summaries:
        subset = [r for r in rows if r["mode"] == mode]
        lines.append(f"## Per question - {mode}")
        lines.append("")
        lines.append("| id | type | R@k | cite P | token F1 | judge | ms |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in subset:
            lines.append(
                f"| {row['id']} | {row['type']} | {_fmt(row.get('recall'))} | "
                f"{_fmt(row.get('citation_precision'))} | {_fmt(row.get('token_f1'))} | "
                f"{_fmt(row.get('judge_score'))} | {row['latency_ms']:.0f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answering quality.")
    parser.add_argument("--modes", nargs="+", default=["rag", "agent"], choices=["rag", "agent"])
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions")
    parser.add_argument("--k", type=int, default=None, help="Cut-off for retrieval metrics")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a table")
    args = parser.parse_args()

    k = args.k or settings.top_k
    items = load_dataset()
    if args.limit:
        items = items[: args.limit]

    use_judge = not args.no_judge
    rows: list[dict] = []

    for mode in args.modes:
        print(f"--- {mode} ---")
        for item in items:
            row = run_item(item, mode, k, use_judge)
            rows.append(row)
            print(
                f"  {row['id']:<18} R@k={_show(row['recall'])} "
                f"citeP={_show(row['citation_precision'])} "
                f"judge={_show(row.get('judge_score'))} "
                f"{row['latency_ms']:.0f}ms"
            )

    summaries = {mode: summarise([r for r in rows if r["mode"] == mode]) for mode in args.modes}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "chat_model": settings.chat_model,
        "embed_model": settings.embed_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "k": k,
        "judge": use_judge,
    }

    raw_path = OUT_DIR / f"{stamp}.json"
    raw_path.write_text(
        json.dumps({"meta": meta, "summaries": summaries, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    report_path = OUT_DIR / f"{stamp}.md"
    write_report(rows, summaries, report_path, meta)

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        print()
        for mode, summary in summaries.items():
            print(f"{mode}: " + "  ".join(f"{k2}={v}" for k2, v in summary.items()))

    print(f"\nraw     {raw_path}")
    print(f"report  {report_path}")
    return 0


def _show(value) -> str:
    return "-" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
