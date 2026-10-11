"""Tests for the agent graph's control flow.

Only the pure parts - routing and prompt assembly. Exercising the actual loop
needs a model, so that lives in the eval harness rather than the unit suite.
"""

from app.agent.graph import _route, _short_args
from app.config import settings
from app.models import RetrievedChunk
from app.qa import grounded_messages


def tool_round(n: int) -> list[dict]:
    return [{"type": "tool"}] * n


def assistant_with_tools():
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "search_papers"}}],
    }


def chunks():
    return [
        RetrievedChunk(
            chunk_id="a-0000",
            doc_id="a",
            doc_title="Paper One",
            page=2,
            text="the gold figure is 12.5",
            score=1.0,
        ),
        RetrievedChunk(
            chunk_id="b-0001",
            doc_id="b",
            doc_title="Paper Two",
            page=7,
            text="the other figure is 9.1",
            score=0.9,
        ),
    ]


def test_routes_to_act_when_tools_are_pending():
    state = {"messages": [assistant_with_tools()], "steps": [{"type": "tool"}]}
    assert _route(state) == "act"


def test_routes_to_write_when_the_model_stops_calling_tools():
    state = {"messages": [{"role": "assistant", "content": "done"}], "steps": []}
    assert _route(state) == "write"


def test_routes_to_write_at_the_step_cap():
    state = {
        "messages": [assistant_with_tools()],
        "steps": tool_round(settings.max_agent_steps),
    }
    assert _route(state) == "write"


def test_step_cap_boundary():
    below = {
        "messages": [assistant_with_tools()],
        "steps": tool_round(settings.max_agent_steps - 1),
    }
    assert _route(below) == "act"


def test_short_args_truncates_long_strings():
    result = _short_args({"query": "x" * 400, "k": 3})
    assert len(result["query"]) == 120
    assert result["k"] == 3


def test_grounded_messages_labels_sources_with_markers():
    messages = grounded_messages("what is the figure?", chunks())
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "[1] Paper One (p. 2)" in user
    assert "[2] Paper Two (p. 7)" in user
    assert user.rstrip().endswith("what is the figure?")


def test_grounded_messages_includes_notes_when_given():
    with_notes = grounded_messages("q", chunks(), notes="my draft reasoning")[1]["content"]
    without = grounded_messages("q", chunks())[1]["content"]
    assert "Analyst notes" in with_notes
    assert "Analyst notes" not in without


def test_grounded_messages_clips_huge_notes():
    notes = grounded_messages("q", chunks(), notes="z" * 5000)[1]["content"]
    assert len(notes) < 4000


def test_grounded_messages_ignores_blank_notes():
    user = grounded_messages("q", chunks(), notes="   ")[1]["content"]
    assert "Analyst notes" not in user
