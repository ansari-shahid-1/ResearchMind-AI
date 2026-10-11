"""Ask one question from the command line. Handy for debugging prompts
without waiting on the UI.

    python scripts/ask.py "which method has the best F1 on MultiHopLit?"
    python scripts/ask.py --mode rag "what is reciprocal rank fusion?"
    python scripts/ask.py --trace "compare the recall of SDF and CRCE"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import graph as agent_graph  # noqa: E402
from app.qa import answer as rag_answer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the indexed papers a question.")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["rag", "agent"], default="agent")
    parser.add_argument("--docs", nargs="*", default=None, help="Restrict to these doc ids")
    parser.add_argument("--trace", action="store_true", help="Show the agent's steps")
    args = parser.parse_args()

    if args.mode == "agent":
        result = agent_graph.run(args.question, doc_ids=args.docs)
    else:
        result = rag_answer(args.question, doc_ids=args.docs)

    if args.trace and result.steps:
        print("--- steps ---")
        for i, step in enumerate(result.steps, 1):
            kind = step.get("type")
            if kind == "tool":
                print(f"{i}. {step['tool']}({step.get('args')}) -> {step.get('results')} chunks, {step.get('ms')}ms")
            elif kind == "seed":
                print(f"{i}. seed retrieval -> {step.get('results')} chunks")
            else:
                print(f"{i}. {kind} {step.get('tools') or ''}")
        print()

    print(result.answer)
    if result.citations:
        print("\n--- sources ---")
        for citation in result.citations:
            print(f"[{citation.marker}] {citation.doc_title} p.{citation.page}")

    print(f"\n[{result.mode}] {result.latency_ms:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
