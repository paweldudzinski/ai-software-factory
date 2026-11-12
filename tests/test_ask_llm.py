import asyncio

import pytest

from software_factory.application.ports.llm_provider import LLMProviderError
from software_factory.application.use_cases.ask_llm import AskLLM
from software_factory.domain.llm.message import LLMMessage, MessageRole
from software_factory.domain.llm.request import LLMRequest
from software_factory.domain.llm.response import LLMResponse, LLMUsage, StopReason

REQUEST = LLMRequest(
    model="claude-sonnet-5-5",
    messages=[LLMMessage(role=MessageRole.USER, content="Hi")],
    max_tokens=1024,
)
RESPONSE = LLMResponse(
    content="Hello!",
    model="claude-sonnet-5-5",
    stop_reason=StopReason.END_TURN,
    usage=LLMUsage(input_tokens=12, output_tokens=3),
    request_id="req_1",
)


class FakeProvider:
    def __init__(
        self, response: LLMResponse = RESPONSE, error: Exception | None = None
    ) -> None:
        self.response = response
        self.error = error
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


def test_passes_request_to_provider_and_returns_response() -> None:
    provider = FakeProvider()

    response = asyncio.run(AskLLM(provider).execute(REQUEST))

    assert provider.requests == [REQUEST]
    assert response is RESPONSE


def test_propagates_provider_error() -> None:
    provider = FakeProvider(error=LLMProviderError("boom", request_id="req_1"))

    with pytest.raises(LLMProviderError, match="boom"):
        asyncio.run(AskLLM(provider).execute(REQUEST))
