from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .blocks import MessageBlock


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    """
    A single, immutable message in a conversation with an LLM.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: MessageRole
    content: Annotated[list[MessageBlock], Field(min_length=1)]
