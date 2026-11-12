from typing import Any

import pytest
from pydantic import ValidationError

from software_factory.domain.llm.message import LLMMessage, MessageRole


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
    message = LLMMessage(role=role, content="Hello")

    assert message.role is expected
    assert message.content == "Hello"


@pytest.mark.parametrize("role", ["", "   ", None, "system", "USER", 1])
def test_rejects_invalid_role(role: Any) -> None:
    with pytest.raises(ValidationError, match="role"):
        LLMMessage(role=role, content="Hello")


def test_rejects_missing_role() -> None:
    with pytest.raises(ValidationError, match=r"role\s+Field required"):
        LLMMessage(content="Hello")  # type: ignore[call-arg]


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_rejects_empty_content(content: str) -> None:
    with pytest.raises(ValidationError, match="content"):
        LLMMessage(role=MessageRole.USER, content=content)


def test_rejects_missing_content() -> None:
    with pytest.raises(ValidationError, match=r"content\s+Field required"):
        LLMMessage(role=MessageRole.USER)  # type: ignore[call-arg]


def test_strips_whitespace_from_content() -> None:
    message = LLMMessage(role=MessageRole.USER, content="  Hello  ")

    assert message.content == "Hello"


def test_is_immutable() -> None:
    message = LLMMessage(role=MessageRole.USER, content="Hello")

    with pytest.raises(ValidationError):
        message.content = "Changed"
