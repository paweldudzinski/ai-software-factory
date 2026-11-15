from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, computed_field

from ..types import Model, NonEmptyStr
from .blocks import ContentBlock

TokenCount = Annotated[StrictInt, Field(ge=0)]


class StopReason(Enum):
    """
    Why the model stopped generating (mirrors Anthropic's `stop_reason`).
    """

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"
    MODEL_CONTEXT_WINDOW_EXCEEDED = "model_context_window_exceeded"


class LLMUsage(BaseModel):
    """
    Token usage reported by the provider for a single request.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: TokenCount
    output_tokens: TokenCount

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """
    A single, immutable response returned by an LLM provider.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: list[ContentBlock]
    model: Model
    stop_reason: StopReason
    usage: LLMUsage
    request_id: NonEmptyStr | None = None
