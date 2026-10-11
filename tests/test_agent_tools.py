import pytest

from app.agent import tools


def test_calculator_arithmetic():
    assert tools.calculate("2 + 3 * 4") == 14
    assert tools.calculate("(72.1 - 64.4) / 64.4 * 100") == pytest.approx(11.9565, rel=1e-3)
    assert tools.calculate("-5 + 2 ** 3") == 3


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('file').read()",
        "x + 1",
        "[1, 2, 3]",
        "'a' * 3",
        "1 if True else 2",
    ],
)
def test_calculator_rejects_non_arithmetic(expression):
    with pytest.raises(ValueError):
        tools.calculate(expression)


def test_calculator_tool_reports_errors_as_observation():
    result = tools.calculator("x + 1")
    assert "error" in result.observation.lower()
    assert result.chunks == []


def test_calculator_tool_rounds_float_noise():
    result = tools.calculator("0.1 + 0.2")
    assert "0.3" in result.observation
    assert "0000000" not in result.observation


def test_calculator_length_guard():
    with pytest.raises(ValueError):
        tools.calculate("1+" * 200 + "1")


def test_dispatch_unknown_tool_is_recoverable():
    result = tools.dispatch("nonexistent", {}, doc_ids=None, start_index=1)
    assert "Unknown tool" in result.observation


def test_dispatch_swallows_tool_failure(temp_env):
    result = tools.dispatch("extract_field", {"doc_id": "nope", "page": "1", "field": "x"}, None, 1)
    assert isinstance(result.observation, str)
    assert result.chunks == []


def test_tool_schemas_are_wellformed():
    for schema in tools.TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] in tools.TOOL_NAMES
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"
