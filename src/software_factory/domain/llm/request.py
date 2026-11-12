from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from ..types import Model, NonEmptyStr
from .message import LLMMessage


class LLMRequest(BaseModel):
    """
    A single, immutable request sent to an LLM.
    """

    model_config = ConfigDict(frozen=True)

    model: Model
    system_prompt: NonEmptyStr | None = None
    messages: Annotated[list[LLMMessage], Field(min_length=1)]
    max_tokens: Annotated[StrictInt, Field(gt=0)]
