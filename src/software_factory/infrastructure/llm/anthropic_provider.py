import anthropic
from anthropic import AsyncAnthropic, omit
from anthropic.types import (
    AssistantMessageParam,
    Message,
    MessageParam,
    TextBlock,
    UserMessageParam,
)

from software_factory.application.ports.llm_provider import LLMProviderError
from software_factory.domain.llm.message import LLMMessage, MessageRole
from software_factory.domain.llm.request import LLMRequest
from software_factory.domain.llm.response import LLMResponse, LLMUsage, StopReason

_MISSING_CREDENTIALS = (
    "No Anthropic credentials found. Set the ANTHROPIC_API_KEY environment "
    "variable or log in with `ant auth login`."
)


class AnthropicProviderError(LLMProviderError):
    """
    Raised when a call to Claude fails or its response cannot be mapped.
    """


class AnthropicProvider:
    """
    LLMProvider adapter for the Claude Messages API.

    Translates LLMRequest into a Messages API call made through the official
    async SDK, and the returned Message back into LLMResponse.
    """

    def __init__(self) -> None:
        # Credentials are resolved by the SDK
        # (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile).
        self._client = AsyncAnthropic()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            message = await self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                messages=self._to_api_messages(request),
                system=(
                    request.system_prompt if request.system_prompt is not None else omit
                ),
            )
        except anthropic.APIStatusError as exc:
            raise AnthropicProviderError(
                f"Anthropic API error {exc.status_code}: {exc.api_message or exc}",
                request_id=exc.request_id,
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise AnthropicProviderError(
                f"Cannot connect to Anthropic API: {exc}"
            ) from exc
        except anthropic.AnthropicError as exc:
            raise AnthropicProviderError(f"Anthropic request failed: {exc}") from exc
        except TypeError as exc:
            # The SDK only checks credentials when sending, and reports missing
            # ones as a plain TypeError. Any other TypeError is a bug: re-raise.
            if "authentication method" not in str(exc):
                raise
            raise AnthropicProviderError(_MISSING_CREDENTIALS) from exc
        return self._to_response(message)

    @classmethod
    def _to_api_messages(cls, request: LLMRequest) -> list[MessageParam]:
        return [cls._to_api_message(message) for message in request.messages]

    @staticmethod
    def _to_api_message(message: LLMMessage) -> MessageParam:
        match message.role:
            case MessageRole.USER:
                return UserMessageParam(role="user", content=message.content)
            case MessageRole.ASSISTANT:
                return AssistantMessageParam(role="assistant", content=message.content)

    @classmethod
    def _to_response(cls, message: Message) -> LLMResponse:
        request_id = message._request_id
        return LLMResponse(
            content=cls._extract_text(message),
            model=message.model,
            stop_reason=cls._to_stop_reason(message, request_id),
            usage=LLMUsage(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            ),
            request_id=request_id,
        )

    @staticmethod
    def _extract_text(message: Message) -> str:
        # `content` is a list of blocks of mixed types (thinking, text, tool_use,
        # ...), in no guaranteed order. Only text blocks form the reply; they are
        # concatenated as-is because the API may split one answer into several
        # blocks (e.g. around citations). No text at all yields "" - stop_reason
        # tells the caller why.
        return "".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        )

    @staticmethod
    def _to_stop_reason(message: Message, request_id: str | None) -> StopReason:
        try:
            return StopReason(message.stop_reason)
        except ValueError:
            raise AnthropicProviderError(
                f"Unsupported stop_reason {message.stop_reason!r}",
                request_id=request_id,
            ) from None
