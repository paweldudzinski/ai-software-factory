from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..types import JSONValue, NonEmptyStr


class TextBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["text"] = "text"
    text: NonEmptyStr


class ToolUseBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["tool_use"] = "tool_use"
    tool_name: NonEmptyStr
    args: dict[str, JSONValue]
    tool_use_id: NonEmptyStr


ContentBlock = Annotated[
    TextBlock | ToolUseBlock,
    Field(discriminator="type"),
]
