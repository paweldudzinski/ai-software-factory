from typing import Any

import pytest
from pydantic import ValidationError

from software_factory.domain.llm.blocks import (
    MessageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from software_factory.domain.llm.message import LLMMessage, MessageRole

HELLO: list[MessageBlock] = [TextBlock(text="Hello")]
TOOL_USE = ToolUseBlock(tool_name="search", args={"q": "x"}, tool_use_id="toolu_1")
TOOL_RESULT = ToolResultBlock(tool_use_id="toolu_1", content="3 results")


# --- role -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("user", MessageRole.USER),
        ("assistant", MessageRole.ASSISTANT),
        (MessageRole.USER, MessageRole.USER),
        (MessageRole.ASSISTANT, MessageRole.ASSISTANT),
    ],
)
def test_creates_message_with_valid_role(role: Any, expected: MessageRole) -> None:
    message = LLMMessage(role=role, content=HELLO)

    assert message.role is expected
    assert message.content == HELLO


@pytest.mark.parametrize("role", ["", "   ", None, "system", "USER", 1])
def test_rejects_invalid_role(role: Any) -> None:
    with pytest.raises(ValidationError, match="role"):
        LLMMessage(role=role, content=HELLO)


def test_rejects_missing_role() -> None:
    with pytest.raises(ValidationError, match=r"role\s+Field required"):
        LLMMessage(content=HELLO)  # type: ignore[call-arg]


# --- content ----------------------------------------------------------------


def test_accepts_every_message_block_type() -> None:
    content: list[MessageBlock] = [TextBlock(text="Found it."), TOOL_USE, TOOL_RESULT]

    message = LLMMessage(role=MessageRole.USER, content=content)

    assert message.content == content


def test_parses_blocks_from_dicts_by_type() -> None:
    message = LLMMessage.model_validate(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "3 results",
                },
                {"type": "text", "text": "Hello"},
            ],
        }
    )

    assert message.content == [TOOL_RESULT, TextBlock(text="Hello")]


@pytest.mark.parametrize("content", ["Hello", "", None, TextBlock(text="Hello")])
def test_rejects_content_that_is_not_a_list(content: Any) -> None:
    with pytest.raises(ValidationError, match="content"):
        LLMMessage(role=MessageRole.USER, content=content)


@pytest.mark.parametrize(
    "block",
    [
        {"type": "image", "source": {}},
        {"text": "missing type"},
        {"type": "text", "text": "   "},
        {"type": "tool_result", "content": "no tool_use_id"},
    ],
)
def test_rejects_invalid_block(block: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match=r"content\.0"):
        LLMMessage(role=MessageRole.USER, content=[block])  # type: ignore[list-item]


def test_rejects_empty_content() -> None:
    with pytest.raises(ValidationError, match=r"content\s+List should have at least 1"):
        LLMMessage(role=MessageRole.USER, content=[])


def test_rejects_missing_content() -> None:
    with pytest.raises(ValidationError, match=r"content\s+Field required"):
        LLMMessage(role=MessageRole.USER)  # type: ignore[call-arg]


def test_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="name"):
        LLMMessage.model_validate(
            {"role": "user", "content": [{"type": "text", "text": "Hi"}], "name": "bob"}
        )


def test_is_immutable() -> None:
    message = LLMMessage(role=MessageRole.USER, content=HELLO)

    with pytest.raises(ValidationError):
        message.content = []
