from typing import Protocol

from software_factory.domain.llm.request import LLMRequest
from software_factory.domain.llm.response import LLMResponse


class LLMProviderError(Exception):
    """
    Raised by an LLMProvider when a request cannot be completed.

    Adapters translate vendor SDK errors into this exception, so callers never
    depend on a specific SDK. `request_id` is set when the provider returned one.
    """

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class LLMProvider(Protocol):
    """
    Port for sending requests to an LLM.

    The application depends only on this interface; concrete providers
    (e.g. Anthropic) are adapters in the infrastructure layer that translate
    the domain models to and from the vendor SDK.
    """

    async def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Send a single request to the model and return its full response.

        Args:
            request: The model, messages, optional system prompt and output
                token limit to send.

        Returns:
            The model's reply together with its stop reason, token usage and
            the provider's request ID. A response that stopped early (e.g.
            `max_tokens` or `refusal`) is returned, not raised; check
            `stop_reason` before using `content`.

        Raises:
            LLMProviderError: The request failed (authentication, rate limit,
                network, API error) or the response could not be mapped.
        """
        ...
