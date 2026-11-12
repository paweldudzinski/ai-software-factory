from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
)

from ..types import NonEmptyStr


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    """
    A single, immutable message in a conversation with an LLM.
    """

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: NonEmptyStr
