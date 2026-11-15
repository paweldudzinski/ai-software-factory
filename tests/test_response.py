from typing import Any

import pytest
from pydantic import ValidationError

from software_factory.domain.llm.response import (
    LLMResponse,
    LLMUsage,
    StopReason,
    TextBlock,
    ToolUseBlock,
)

USAGE = LLMUsage(input_tokens=120, output_tokens=30)
TEXT = TextBlock(text="Hello!")
TOOL_USE = ToolUseBlock(
    tool_name="search", args={"query": "pydantic"}, tool_use_id="toolu_1"
)


def make_response(**overrides: Any) -> LLMResponse:
    data: dict[str, Any] = {
        "content": [TEXT],
        "model": "claude-sonnet-5-5",
        "stop_reason": StopReason.END_TURN,
        "usage": USAGE,
        "request_id": "req_018EeWyXxfu5pfWkrYcMdjWG",
    }
    data.update(overrides)
    return LLMResponse(**data)


# --- LLMUsage ---------------------------------------------------------------


def test_usage_computes_total_tokens() -> None:
    assert USAGE.total_tokens == 150


def test_usage_total_tokens_is_serialized() -> None:
    assert USAGE.model_dump() == {
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
    }


def test_usage_accepts_zero_tokens() -> None:
    usage = LLMUsage(input_tokens=0, output_tokens=0)

    assert usage.total_tokens == 0


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("value", [-1, "10", 1.5, True, None])
def test_usage_rejects_invalid_token_count(field: str, value: Any) -> None:
    data: dict[str, Any] = {"input_tokens": 1, "output_tokens": 1, field: value}

    with pytest.raises(ValidationError, match=field):
        LLMUsage(**data)


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_usage_rejects_missing_field(field: str) -> None:
    data: dict[str, Any] = {"input_tokens": 1, "output_tokens": 1}
    del data[field]

    with pytest.raises(ValidationError, match=rf"{field}\s+Field required"):
        LLMUsage(**data)


def test_usage_is_immutable() -> None:
    with pytest.raises(ValidationError):
        USAGE.input_tokens = 1


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


# --- LLMResponse ------------------------------------------------------------


def test_creates_valid_response() -> None:
    response = make_response()

    assert response.content == [TEXT]
    assert response.model == "claude-sonnet-5-5"
    assert response.stop_reason is StopReason.END_TURN
    assert response.usage == USAGE
    assert response.request_id == "req_018EeWyXxfu5pfWkrYcMdjWG"


def test_parses_from_plain_data() -> None:
    response = LLMResponse.model_validate(
        {
            "content": [
                {"type": "text", "text": "Let me search."},
                {
                    "type": "tool_use",
                    "tool_name": "search",
                    "args": {"query": "pydantic"},
                    "tool_use_id": "toolu_1",
                },
            ],
            "model": "claude-sonnet-5-5",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 120, "output_tokens": 30},
            "request_id": "req_1",
        }
    )

    assert response.content == [TextBlock(text="Let me search."), TOOL_USE]
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.usage.total_tokens == 150


def test_keeps_order_of_mixed_blocks() -> None:
    content = [TEXT, TOOL_USE, TextBlock(text="Done.")]

    response = make_response(content=content, stop_reason=StopReason.TOOL_USE)

    assert response.content == content
    assert [type(block) for block in response.content] == [
        TextBlock,
        ToolUseBlock,
        TextBlock,
    ]


def test_round_trips_through_dump() -> None:
    response = make_response(content=[TEXT, TOOL_USE])

    assert LLMResponse.model_validate(response.model_dump()) == response


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image", "source": {}},
        {"text": "missing type"},
        {"type": "text"},
        {"type": "tool_use", "tool_name": "search", "args": {}},
    ],
)
def test_rejects_invalid_content_block(block: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match=r"content\.0"):
        make_response(content=[block])


@pytest.mark.parametrize("content", ["Hello!", None, TEXT])
def test_rejects_content_that_is_not_a_list(content: Any) -> None:
    with pytest.raises(ValidationError, match="content"):
        make_response(content=content)


@pytest.mark.parametrize("stop_reason", list(StopReason))
def test_accepts_every_stop_reason(stop_reason: StopReason) -> None:
    response = make_response(stop_reason=stop_reason.value)

    assert response.stop_reason is stop_reason


@pytest.mark.parametrize("stop_reason", ["", "finished", "END_TURN", None])
def test_rejects_invalid_stop_reason(stop_reason: Any) -> None:
    with pytest.raises(ValidationError, match="stop_reason"):
        make_response(stop_reason=stop_reason)


def test_allows_empty_content() -> None:
    response = make_response(content=[], stop_reason=StopReason.REFUSAL)

    assert response.content == []


@pytest.mark.parametrize("model", ["", "   "])
def test_rejects_empty_model(model: str) -> None:
    with pytest.raises(ValidationError, match="model"):
        make_response(model=model)


def test_request_id_is_optional() -> None:
    response = make_response(request_id=None)

    assert response.request_id is None


@pytest.mark.parametrize("request_id", ["", "   "])
def test_rejects_empty_request_id(request_id: str) -> None:
    with pytest.raises(ValidationError, match="request_id"):
        make_response(request_id=request_id)


@pytest.mark.parametrize("field", ["content", "model", "stop_reason", "usage"])
def test_rejects_missing_required_field(field: str) -> None:
    data: dict[str, Any] = {
        "content": [TEXT],
        "model": "claude-sonnet-5-5",
        "stop_reason": StopReason.END_TURN,
        "usage": USAGE,
    }
    del data[field]

    with pytest.raises(ValidationError, match=rf"{field}\s+Field required"):
        LLMResponse(**data)


def test_rejects_invalid_usage() -> None:
    with pytest.raises(ValidationError, match=r"usage\.input_tokens"):
        make_response(usage={"input_tokens": -1, "output_tokens": 0})


def test_response_is_immutable() -> None:
    response = make_response()

    with pytest.raises(ValidationError):
        response.content = []
