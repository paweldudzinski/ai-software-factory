from typing import Any

import pytest
from pydantic import ValidationError

from software_factory.domain.llm.message import LLMMessage, MessageRole
from software_factory.domain.llm.request import LLMRequest

USER_MESSAGE = LLMMessage(role=MessageRole.USER, content="Hello")


def make_request(**overrides: Any) -> LLMRequest:
    data: dict[str, Any] = {
        "model": "claude-sonnet-5-5",
        "messages": [USER_MESSAGE],
        "max_tokens": 1024,
    }
    data.update(overrides)
    return LLMRequest(**data)


def test_creates_valid_request() -> None:
    request = make_request(system_prompt="You are a helpful assistant.")

    assert request.model == "claude-sonnet-5-5"
    assert request.system_prompt == "You are a helpful assistant."
    assert request.messages == [USER_MESSAGE]
    assert request.max_tokens == 1024


def test_system_prompt_is_optional() -> None:
    request = make_request()

    assert request.system_prompt is None


def test_accepts_explicit_none_system_prompt() -> None:
    request = make_request(system_prompt=None)

    assert request.system_prompt is None


def test_accepts_multiple_messages() -> None:
    messages = [
        USER_MESSAGE,
        LLMMessage(role=MessageRole.ASSISTANT, content="Hi! How can I help?"),
        LLMMessage(role=MessageRole.USER, content="Write a test."),
    ]

    request = make_request(messages=messages)

    assert request.messages == messages


def test_parses_messages_from_dicts() -> None:
    request = make_request(messages=[{"role": "user", "content": "Hello"}])

    assert request.messages == [USER_MESSAGE]


@pytest.mark.parametrize("model", ["", "   "])
def test_rejects_empty_model(model: str) -> None:
    with pytest.raises(ValidationError, match="model"):
        make_request(model=model)


@pytest.mark.parametrize("system_prompt", ["", "   "])
def test_rejects_empty_system_prompt(system_prompt: str) -> None:
    with pytest.raises(ValidationError, match="system_prompt"):
        make_request(system_prompt=system_prompt)


def test_rejects_empty_messages() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        make_request(messages=[])


@pytest.mark.parametrize(
    "messages",
    [
        None,
        [{"role": "system", "content": "Hello"}],
        [{"role": "user", "content": ""}],
        ["Hello"],
    ],
)
def test_rejects_invalid_messages(messages: Any) -> None:
    with pytest.raises(ValidationError, match="messages"):
        make_request(messages=messages)


@pytest.mark.parametrize("max_tokens", [0, -1, "1024", 1.5, True, None])
def test_rejects_invalid_max_tokens(max_tokens: Any) -> None:
    with pytest.raises(ValidationError, match="max_tokens"):
        make_request(max_tokens=max_tokens)


@pytest.mark.parametrize("field", ["model", "messages", "max_tokens"])
def test_rejects_missing_required_field(field: str) -> None:
    data: dict[str, Any] = {
        "model": "claude-sonnet-5-5",
        "messages": [USER_MESSAGE],
        "max_tokens": 1024,
    }
    del data[field]

    with pytest.raises(ValidationError, match=rf"{field}\s+Field required"):
        LLMRequest(**data)


def test_is_immutable() -> None:
    request = make_request()

    with pytest.raises(ValidationError):
        request.max_tokens = 2048
