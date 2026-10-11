"""The agent loop, built as an explicit LangGraph state machine.

    START -> seed -> agent -> (act | write) -> END
                      ^        |
                      +-- act -+

Three decisions worth explaining, because all three came out of watching a 3B
model actually behave rather than from theory.

1. The loop is seeded with one retrieval for the raw question before the agent
   runs. A small model often answers a paper question straight from memory on
   step one, which looks fine and is completely unsourced. Seeding means the
   worst case - the model never calls a tool at all - is still grounded.

2. Evidence gathering and answer writing are separate steps. The first version
   let the agent's final message be the answer, and the eval caught it: on 14
   of 19 questions the agent produced zero citation markers, and citation
   precision fell to 0.21 against 0.84 for the plain RAG path. The model, deep
   into a tool loop, just stops obeying the "cite everything" instruction. So
   the tools now only gather; a `write` node produces the answer with the same
   prompt the RAG path uses, and the agent's own draft is handed over as
   loosely-trusted notes.

3. Source markers are numbered globally across the whole run, not per search
   call. The model can cite [7] from a search it ran three steps earlier and
   the writer resolves everything against one flat list.

The step cap matters for latency more than quality - without it a confused run
will search the same thing six times - and hitting the cap routes straight to
`write` rather than to a separate finalise node.
"""

import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app import llm
from app.agent.tools import TOOL_SCHEMAS, dispatch
from app.config import settings
from app.models import Answer, RetrievedChunk
from app.prompts import AGENT_SYSTEM
from app.qa import (
    build_context,
    extract_citations,
    strip_unknown_markers,
    write_cited_answer,
)
from app.retrieval.search import search

_MARKER = re.compile(r"\[\d+\]")

NO_EVIDENCE = (
    "I could not find anything relevant in the uploaded papers. Either nothing "
    "has been uploaded yet, or the question is outside them."
)


class AgentState(TypedDict, total=False):
    question: str
    doc_ids: list[str] | None
    messages: list[dict]
    chunks: list[RetrievedChunk]
    steps: list[dict]
    draft: str
    answer: str
    usage: dict


def _seed(state: AgentState) -> AgentState:
    """Fetch one round of context for the raw question."""
    question = state["question"]
    chunks = search(question, k=settings.top_k, doc_ids=state.get("doc_ids"))

    messages = [{"role": "system", "content": AGENT_SYSTEM}]
    if chunks:
        messages[0]["content"] += (
            "\n\nSources already retrieved for this question. Cite them by number:\n\n"
            + build_context(chunks)
        )
    messages.append({"role": "user", "content": question})

    return {
        "messages": messages,
        "chunks": chunks,
        "steps": [{"type": "seed", "query": question, "results": len(chunks)}],
    }


def _agent(state: AgentState) -> AgentState:
    started = time.perf_counter()
    response = llm.chat(state["messages"], tools=TOOL_SCHEMAS)
    message = llm.message_of(response)
    calls = llm.tool_calls_of(response)

    assistant: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    if calls:
        assistant["tool_calls"] = calls

    steps = list(state.get("steps", []))
    steps.append(
        {
            "type": "tool_call" if calls else "draft",
            "tools": [c.get("function", {}).get("name") for c in calls],
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }
    )

    update: AgentState = {
        "messages": state["messages"] + [assistant],
        "steps": steps,
        "usage": llm.merge_usage(state.get("usage"), llm.usage_of(response)),
    }
    if not calls:
        update["draft"] = (message.get("content") or "").strip()
    return update


def _act(state: AgentState) -> AgentState:
    """Execute every tool call in the last assistant message."""
    calls = state["messages"][-1].get("tool_calls") or []

    messages = list(state["messages"])
    chunks = list(state.get("chunks", []))
    steps = list(state.get("steps", []))

    for call in calls:
        fn = call.get("function", {}) or {}
        name = fn.get("name") or ""
        args = fn.get("arguments")
        if isinstance(args, str):
            args = llm.parse_json(args, default={}) or {}
        if not isinstance(args, dict):
            args = {}

        started = time.perf_counter()
        result = dispatch(name, args, state.get("doc_ids"), start_index=len(chunks) + 1)
        elapsed = round((time.perf_counter() - started) * 1000, 1)

        chunks.extend(result.chunks)
        messages.append({"role": "tool", "content": result.observation, "tool_name": name})
        steps.append(
            {
                "type": "tool",
                "tool": name,
                "args": _short_args(args),
                "ms": elapsed,
                "observation": result.observation[:300],
                "results": len(result.chunks),
            }
        )

    return {"messages": messages, "chunks": chunks, "steps": steps}


def _write(state: AgentState) -> AgentState:
    """Produce the final answer from the evidence that was gathered."""
    chunks = state.get("chunks", [])
    steps = list(state.get("steps", []))

    if not chunks:
        steps.append({"type": "write", "sources": 0})
        return {"answer": state.get("draft") or NO_EVIDENCE, "steps": steps}

    started = time.perf_counter()
    # the agent's own draft, if it wrote one, is passed as notes: good for
    # structure, not trusted for facts
    text, attempts, usage = write_cited_answer(
        state["question"], chunks, notes=state.get("draft")
    )

    steps.append(
        {
            "type": "write",
            "sources": len(chunks),
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "notes": bool(state.get("draft")),
            "attempts": attempts,
        }
    )
    return {
        "answer": text,
        "steps": steps,
        "usage": llm.merge_usage(state.get("usage"), usage),
    }


def _route(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.get("role") != "assistant" or not last.get("tool_calls"):
        return "write"
    rounds = sum(1 for s in state.get("steps", []) if s.get("type") == "tool")
    return "write" if rounds >= settings.max_agent_steps else "act"


def _short_args(args: dict) -> dict:
    return {k: (v[:120] if isinstance(v, str) else v) for k, v in args.items()}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("seed", _seed)
    graph.add_node("agent", _agent)
    graph.add_node("act", _act)
    graph.add_node("write", _write)

    graph.add_edge(START, "seed")
    graph.add_edge("seed", "agent")
    graph.add_conditional_edges("agent", _route, {"act": "act", "write": "write"})
    graph.add_edge("act", "agent")
    graph.add_edge("write", END)
    return graph.compile()


_compiled = None


def compiled():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run(question: str, doc_ids: list[str] | None = None) -> Answer:
    started = time.perf_counter()
    state = compiled().invoke(
        {"question": question, "doc_ids": doc_ids, "chunks": [], "steps": [], "usage": {}},
        {"recursion_limit": 2 * settings.max_agent_steps + 6},
    )

    chunks = state.get("chunks", [])
    text = state.get("answer", "") or NO_EVIDENCE

    if chunks and text and not _MARKER.search(text):
        text += "\n\n(No citation markers were produced for this answer.)"

    return Answer(
        question=question,
        answer=strip_unknown_markers(text, chunks),
        citations=extract_citations(text, chunks),
        sources=chunks,
        mode="agent",
        steps=state.get("steps", []),
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        usage=state.get("usage", {}),
    )
