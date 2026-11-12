import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx2
import pytest
from anthropic import AsyncAnthropic, omit
from anthropic.types import Message

from software_factory.application.ports.llm_provider import (
    LLMProvider,
    LLMProviderError,
)
from software_factory.domain.llm.message import LLMMessage, MessageRole
from software_factory.domain.llm.request import LLMRequest
from software_factory.domain.llm.response import LLMResponse, StopReason
from software_factory.infrastructure.llm.anthropic_provider import (
    AnthropicProvider,
    AnthropicProviderError,
)

MODEL = "claude-sonnet-5-5"
REQUEST_ID = "req_018EeWyXxfu5pfWkrYcMdjWG"
PROVIDER_MODULE = "software_factory.infrastructure.llm.anthropic_provider"

TEXT = {"type": "text", "text": "Hello!", "citations": None}
THINKING = {"type": "thinking", "thinking": "", "signature": "sig"}
TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}


def make_message(
    content: list[dict[str, Any]] | None = None,
    stop_reason: str | None = "end_turn",
    request_id: str | None = REQUEST_ID,
) -> Message:
    message = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [TEXT] if content is None else content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 120, "output_tokens": 30},
        }
    )
    message._request_id = request_id
    return message


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the SDK client created by AnthropicProvider, so no API is called."""
    fake = MagicMock(spec=AsyncAnthropic)
    monkeypatch.setattr(f"{PROVIDER_MODULE}.AsyncAnthropic", lambda: fake)
    return fake


def make_provider(
    client: MagicMock, message: Message
) -> tuple[AnthropicProvider, AsyncMock]:
    create = AsyncMock(return_value=message)
    client.messages.create = create
    return AnthropicProvider(), create


def make_request(**overrides: Any) -> LLMRequest:
    data: dict[str, Any] = {
        "model": MODEL,
        "messages": [LLMMessage(role=MessageRole.USER, content="Hi")],
        "max_tokens": 1024,
    }
    data.update(overrides)
    return LLMRequest(**data)


def generate(provider: AnthropicProvider, request: LLMRequest) -> LLMResponse:
    return asyncio.run(provider.generate(request))


# --- request mapping --------------------------------------------------------


def test_maps_request_to_messages_api_params(client: MagicMock) -> None:
    provider, create = make_provider(client, make_message())
    request = make_request(
        system_prompt="You are a helpful assistant.",
        messages=[
            LLMMessage(role=MessageRole.USER, content="Hi"),
            LLMMessage(role=MessageRole.ASSISTANT, content="Hello!"),
            LLMMessage(role=MessageRole.USER, content="Write a test."),
        ],
    )

    generate(provider, request)

    create.assert_awaited_once_with(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Write a test."},
        ],
        system="You are a helpful assistant.",
    )


def test_omits_system_when_no_system_prompt(client: MagicMock) -> None:
    provider, create = make_provider(client, make_message())

    generate(provider, make_request())

    assert create.await_args is not None
    assert create.await_args.kwargs["system"] is omit


# --- response mapping -------------------------------------------------------


def test_maps_response(client: MagicMock) -> None:
    provider, _ = make_provider(client, make_message())

    response = generate(provider, make_request())

    assert response.content == "Hello!"
    assert response.model == MODEL
    assert response.stop_reason is StopReason.END_TURN
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 30
    assert response.request_id == REQUEST_ID


def test_skips_non_text_blocks_before_text(client: MagicMock) -> None:
    provider, _ = make_provider(client, make_message(content=[THINKING, TEXT]))

    response = generate(provider, make_request())

    assert response.content == "Hello!"


def test_concatenates_multiple_text_blocks(client: MagicMock) -> None:
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": "The answer ", "citations": None},
        THINKING,
        {"type": "text", "text": "is 42.", "citations": None},
    ]
    provider, _ = make_provider(client, make_message(content=blocks))

    response = generate(provider, make_request())

    assert response.content == "The answer is 42."


@pytest.mark.parametrize(
    ("content", "stop_reason"),
    [
        ([], "refusal"),
        ([THINKING], "max_tokens"),
        ([THINKING, TOOL_USE], "tool_use"),
    ],
)
def test_returns_empty_content_when_no_text(
    client: MagicMock, content: list[dict[str, Any]], stop_reason: str
) -> None:
    provider, _ = make_provider(
        client, make_message(content=content, stop_reason=stop_reason)
    )

    response = generate(provider, make_request())

    assert response.content == ""
    assert response.stop_reason is StopReason(stop_reason)


def test_allows_missing_request_id(client: MagicMock) -> None:
    provider, _ = make_provider(client, make_message(request_id=None))

    response = generate(provider, make_request())

    assert response.request_id is None


def test_raises_on_missing_stop_reason(client: MagicMock) -> None:
    provider, _ = make_provider(client, make_message(stop_reason=None))

    with pytest.raises(AnthropicProviderError, match="stop_reason") as exc_info:
        generate(provider, make_request())

    assert exc_info.value.request_id == REQUEST_ID


# --- error translation ------------------------------------------------------

API_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def make_failing_provider(client: MagicMock, error: Exception) -> AnthropicProvider:
    client.messages.create = AsyncMock(side_effect=error)
    return AnthropicProvider()


def test_translates_api_status_error(client: MagicMock) -> None:
    error = anthropic.RateLimitError(
        "rate limited",
        response=httpx2.Response(
            429, headers={"request-id": REQUEST_ID}, request=API_REQUEST
        ),
        body={"error": {"type": "rate_limit_error", "message": "Slow down"}},
    )
    provider = make_failing_provider(client, error)

    with pytest.raises(LLMProviderError, match="429: Slow down") as exc_info:
        generate(provider, make_request())

    assert exc_info.value.request_id == REQUEST_ID
    assert exc_info.value.__cause__ is error


def test_translates_connection_error(client: MagicMock) -> None:
    provider = make_failing_provider(
        client, anthropic.APIConnectionError(request=API_REQUEST)
    )

    with pytest.raises(LLMProviderError, match="Cannot connect") as exc_info:
        generate(provider, make_request())

    assert exc_info.value.request_id is None


# --- wiring -----------------------------------------------------------------


def test_creates_default_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    provider = AnthropicProvider()

    assert isinstance(provider._client, AsyncAnthropic)


def test_satisfies_llm_provider_port(client: MagicMock) -> None:
    provider: LLMProvider = AnthropicProvider()

    assert provider is not None


def test_translates_missing_credentials(client: MagicMock) -> None:
    error = TypeError(
        '"Could not resolve authentication method. Expected one of api_key, '
        'auth_token, or credentials to be set."'
    )
    provider = make_failing_provider(client, error)

    with pytest.raises(LLMProviderError, match="No Anthropic credentials found"):
        generate(provider, make_request())


def test_reraises_unrelated_type_error(client: MagicMock) -> None:
    error = TypeError("unexpected keyword argument 'foo'")
    provider = make_failing_provider(client, error)

    with pytest.raises(TypeError, match="foo"):
        generate(provider, make_request())
