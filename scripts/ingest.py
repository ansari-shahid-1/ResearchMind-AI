"""Ingest PDFs from the command line.

    python scripts/ingest.py data/sample_papers
    python scripts/ingest.py paper.pdf --title "My Title"
    python scripts/ingest.py data/sample_papers --reset
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import registry  # noqa: E402
from app.ingest.pipeline import ingest_pdf  # noqa: E402


def collect(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(p for p in path.glob("*.pdf")))
        elif path.suffix.lower() == ".pdf":
            found.append(path)
        else:
            print(f"skipping {path} (not a PDF)")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector store.")
    parser.add_argument("paths", nargs="+", help="PDF files or directories")
    parser.add_argument("--title", default=None, help="Override the title (single file only)")
    parser.add_argument("--reset", action="store_true", help="Wipe the index and registry first")
    args = parser.parse_args()

    registry.init()
    if args.reset:
        from app.retrieval import store

        store.reset()
        registry.clear()
        print("index reset\n")

    files = collect(args.paths)
    if not files:
        print("nothing to ingest")
        return 1

    total_chunks = 0
    for path in files:
        title = args.title if len(files) == 1 else None
        try:
            doc, duplicate = ingest_pdf(path, title=title)
        except Exception as exc:
            print(f"  FAILED {path.name}: {exc}")
            continue

        if duplicate:
            print(f"  skip   {path.name} (already indexed as {doc.doc_id})")
        else:
            total_chunks += doc.chunks
            print(f"  ok     {doc.title}")
            print(f"         {doc.pages} pages -> {doc.chunks} chunks   id={doc.doc_id}")

    print(f"\n{len(registry.list_all())} documents, {total_chunks} new chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
