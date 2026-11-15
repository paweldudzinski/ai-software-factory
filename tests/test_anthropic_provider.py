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
from software_factory.domain.llm.response import (
    LLMResponse,
    StopReason,
    TextBlock,
    ToolUseBlock,
)
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


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text, "citations": None}


def test_maps_response(client: MagicMock) -> None:
    provider, _ = make_provider(client, make_message())

    response = generate(provider, make_request())

    assert response.content == [TextBlock(text="Hello!")]
    assert response.model == MODEL
    assert response.stop_reason is StopReason.END_TURN
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 30
    assert response.request_id == REQUEST_ID


def test_keeps_each_text_block_separate(client: MagicMock) -> None:
    blocks = [text_block("First part."), text_block("Second part.")]
    provider, _ = make_provider(client, make_message(content=blocks))

    response = generate(provider, make_request())

    assert response.content == [
        TextBlock(text="First part."),
        TextBlock(text="Second part."),
    ]


def test_keeps_order_of_supported_blocks_and_skips_thinking(
    client: MagicMock,
) -> None:
    blocks = [text_block("Let me search."), THINKING, TOOL_USE, text_block("Done.")]
    provider, _ = make_provider(
        client, make_message(content=blocks, stop_reason="tool_use")
    )

    response = generate(provider, make_request())

    assert response.content == [
        TextBlock(text="Let me search."),
        ToolUseBlock(tool_name="search", args={}, tool_use_id="toolu_1"),
        TextBlock(text="Done."),
    ]


@pytest.mark.parametrize(
    "blocks",
    [
        [THINKING],
        [THINKING, {"type": "redacted_thinking", "data": "encrypted"}],
    ],
)
def test_returns_empty_content_when_only_thinking(
    client: MagicMock, blocks: list[dict[str, Any]]
) -> None:
    provider, _ = make_provider(
        client, make_message(content=blocks, stop_reason="max_tokens")
    )

    response = generate(provider, make_request())

    assert response.content == []
    assert response.stop_reason is StopReason.MAX_TOKENS


def test_maps_tool_use_block(client: MagicMock) -> None:
    args = {"query": "pydantic", "limit": 5, "filters": {"lang": ["en", "pl"]}}
    block = {"type": "tool_use", "id": "toolu_42", "name": "search", "input": args}
    provider, _ = make_provider(
        client, make_message(content=[block], stop_reason="tool_use")
    )

    response = generate(provider, make_request())

    [tool_use] = response.content
    assert isinstance(tool_use, ToolUseBlock)
    assert tool_use.tool_name == "search"
    assert tool_use.args == args
    assert tool_use.tool_use_id == "toolu_42"


def test_raises_on_unsupported_block_type(client: MagicMock) -> None:
    server_tool_use = {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "pydantic"},
    }
    provider, _ = make_provider(
        client, make_message(content=[TEXT, server_tool_use])
    )

    with pytest.raises(
        AnthropicProviderError, match="Unsupported content block type 'server_tool_use'"
    ) as exc_info:
        generate(provider, make_request())

    assert exc_info.value.request_id == REQUEST_ID


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
