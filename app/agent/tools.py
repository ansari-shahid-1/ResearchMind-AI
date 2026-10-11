"""Tools the agent can call.

Each tool returns an observation string (what the model reads) plus any
chunks it should be allowed to cite. Search is the only one that produces
citable chunks; the calculator and extractor feed numbers and facts back
into the loop but cite through the retrieval that surfaced them.
"""

import ast
import operator
from typing import Any, NamedTuple

from app import llm, registry
from app.config import settings
from app.models import RetrievedChunk
from app.prompts import EXTRACT_SYSTEM
from app.retrieval import store
from app.retrieval.search import search


class ToolResult(NamedTuple):
    observation: str
    chunks: list[RetrievedChunk]


# --------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    """Deliberately does not call eval(). Only arithmetic on literals."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"not a number: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {type(node).__name__}")


def calculate(expression: str) -> float:
    if len(expression) > 200:
        raise ValueError("expression too long")
    return _eval_node(ast.parse(expression, mode="eval").body)


# --------------------------------------------------------------------------
# tool implementations
# --------------------------------------------------------------------------

def search_papers(
    query: str,
    k: int | None = None,
    doc_ids: list[str] | None = None,
    start_index: int = 1,
) -> ToolResult:
    chunks = search(query, k=k or settings.top_k, doc_ids=doc_ids)
    if not chunks:
        return ToolResult("No results. Nothing in the corpus matched that query.", [])

    lines = []
    for offset, chunk in enumerate(chunks):
        marker = start_index + offset
        # the model gets a trimmed preview; the full text stays in the state
        # so the final answer can be built from untruncated context
        preview = chunk.text[:700]
        lines.append(f"[{marker}] {chunk.doc_title} (p. {chunk.page})\n{preview}")
    return ToolResult("\n\n".join(lines), chunks)


def calculator(expression: str) -> ToolResult:
    try:
        value = calculate(expression)
    except Exception as exc:
        return ToolResult(f"Calculator error: {exc}", [])
    # round float noise so the model doesn't report 0.30000000000000004
    if isinstance(value, float):
        value = round(value, 6)
    return ToolResult(f"{expression} = {value}", [])


def extract_field(doc_id: str, page: int, field: str) -> ToolResult:
    rows = store.get_page(doc_id, page)
    if not rows:
        return ToolResult(f"No text found for document {doc_id} page {page}.", [])

    text = "\n".join(r["text"] for r in rows)
    title = rows[0]["doc_title"]
    prompt = f"Text from '{title}', page {page}:\n\n{text[:6000]}\n\nField to extract: {field}"
    parsed = llm.generate_json(prompt, system=EXTRACT_SYSTEM)

    if not isinstance(parsed, dict):
        return ToolResult(f"Could not extract '{field}' as JSON. Raw page text:\n{text[:700]}", [])

    value = parsed.get("value")
    quote = (parsed.get("quote") or "").strip()
    observation = f"{title} p.{page} -> {field}: {value}"
    if quote:
        observation += f"\nSupporting text: \"{quote}\""

    chunk = RetrievedChunk(
        chunk_id=rows[0]["chunk_id"],
        doc_id=doc_id,
        doc_title=title,
        page=int(page),
        text=quote or text[:400],
        score=1.0,
    )
    return ToolResult(observation, [chunk])


def list_documents() -> ToolResult:
    docs = registry.list_all()
    if not docs:
        return ToolResult("No documents have been uploaded.", [])
    lines = [f"- {d.title} ({d.pages} pages, id={d.doc_id})" for d in docs]
    return ToolResult("Uploaded documents:\n" + "\n".join(lines), [])


# --------------------------------------------------------------------------
# schemas handed to the model
# --------------------------------------------------------------------------

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": (
            "Search the uploaded papers for passages relevant to a query. "
            "Returns numbered sources that can be cited as [n]. Call this "
            "several times with different queries for multi-part questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for. Be specific - 'FiD F1 on Natural Questions' beats 'results'.",
                },
                "k": {"type": "integer", "description": "How many passages to return (default 6)."},
            },
            "required": ["query"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '(72.1 - 64.4) / 64.4 * 100'.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic using + - * / // % ** and parentheses. Numbers only.",
                }
            },
            "required": ["expression"],
        },
    },
}

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_field",
        "description": (
            "Pull one named field out of a specific page of a specific paper. "
            "Use when you know which page holds the value and want it verbatim, "
            "e.g. a reported metric, a dataset size or a model name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document id from list_documents or a search result."},
                "page": {"type": "integer", "description": "1-based page number."},
                "field": {"type": "string", "description": "What to extract, e.g. 'reported F1'."},
            },
            "required": ["doc_id", "page", "field"],
        },
    },
}

LIST_TOOL = {
    "type": "function",
    "function": {
        "name": "list_documents",
        "description": "List the papers that have been uploaded, with their ids.",
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_SCHEMAS = [SEARCH_TOOL, CALCULATOR_TOOL, EXTRACT_TOOL, LIST_TOOL]
TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}


def dispatch(
    name: str,
    args: dict[str, Any],
    doc_ids: list[str] | None,
    start_index: int,
) -> ToolResult:
    """Route a tool call. Unknown names and bad arguments become an
    observation the model can recover from rather than a crash."""
    try:
        if name == "search_papers":
            return search_papers(
                query=str(args.get("query", "")),
                k=args.get("k"),
                doc_ids=doc_ids,
                start_index=start_index,
            )
        if name == "calculator":
            return calculator(str(args.get("expression", "")))
        if name == "extract_field":
            return extract_field(
                doc_id=str(args.get("doc_id", "")),
                page=int(args.get("page", 0)),
                field=str(args.get("field", "")),
            )
        if name == "list_documents":
            return list_documents()
    except Exception as exc:
        return ToolResult(f"{name} failed: {exc}", [])
    return ToolResult(f"Unknown tool: {name}", [])
