from typing import Any

import pytest
from pydantic import ValidationError

from software_factory.domain.llm.blocks import TextBlock, ToolUseBlock

TEXT = TextBlock(text="Hello!")
TOOL_USE = ToolUseBlock(
    tool_name="search", args={"query": "pydantic"}, tool_use_id="toolu_1"
)


# --- TextBlock --------------------------------------------------------------


def test_text_block_has_text_type() -> None:
    assert TEXT.type == "text"
    assert TEXT.text == "Hello!"


@pytest.mark.parametrize("text", ["", "   "])
def test_text_block_rejects_empty_text(text: str) -> None:
    with pytest.raises(ValidationError, match="text"):
        TextBlock(text=text)


def test_text_block_rejects_missing_text() -> None:
    with pytest.raises(ValidationError, match=r"text\s+Field required"):
        TextBlock.model_validate({"type": "text"})


def test_text_block_rejects_other_type() -> None:
    with pytest.raises(ValidationError, match="type"):
        TextBlock.model_validate({"type": "tool_use", "text": "Hi"})


def test_text_block_is_immutable() -> None:
    with pytest.raises(ValidationError):
        TEXT.text = "Changed"


# --- ToolUseBlock -----------------------------------------------------------


def test_tool_use_block_has_tool_use_type() -> None:
    assert TOOL_USE.type == "tool_use"
    assert TOOL_USE.tool_name == "search"
    assert TOOL_USE.args == {"query": "pydantic"}
    assert TOOL_USE.tool_use_id == "toolu_1"


def test_tool_use_block_accepts_empty_args() -> None:
    block = ToolUseBlock(tool_name="now", args={}, tool_use_id="toolu_1")

    assert block.args == {}


def test_tool_use_block_accepts_nested_json_args() -> None:
    args: dict[str, Any] = {
        "path": "src/app.py",
        "lines": [1, 2, 3],
        "options": {"recursive": True, "depth": None, "ratio": 0.5},
    }

    block = ToolUseBlock(tool_name="read", args=args, tool_use_id="toolu_1")

    assert block.args == args


@pytest.mark.parametrize(
    "args",
    [None, [], "query", {1: "not a string key"}, {"value": object()}],
)
def test_tool_use_block_rejects_non_json_object_args(args: Any) -> None:
    with pytest.raises(ValidationError, match="args"):
        ToolUseBlock(tool_name="search", args=args, tool_use_id="toolu_1")


@pytest.mark.parametrize("field", ["tool_name", "tool_use_id"])
@pytest.mark.parametrize("value", ["", "   "])
def test_tool_use_block_rejects_empty_identifiers(field: str, value: str) -> None:
    data: dict[str, Any] = {
        "tool_name": "search",
        "args": {},
        "tool_use_id": "toolu_1",
        field: value,
    }

    with pytest.raises(ValidationError, match=field):
        ToolUseBlock(**data)


@pytest.mark.parametrize("field", ["tool_name", "args", "tool_use_id"])
def test_tool_use_block_rejects_missing_field(field: str) -> None:
    data: dict[str, Any] = {"tool_name": "search", "args": {}, "tool_use_id": "t"}
    del data[field]

    with pytest.raises(ValidationError, match=rf"{field}\s+Field required"):
        ToolUseBlock(**data)


def test_tool_use_block_is_immutable() -> None:
    with pytest.raises(ValidationError):
        TOOL_USE.tool_name = "other"
