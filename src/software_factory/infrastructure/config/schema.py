from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Settings(BaseModel):
    # forbid: a typo such as `max_iteration` fails instead of being silently ignored.
    model_config = ConfigDict(frozen=True, extra="forbid")


class FactorySettings(_Settings):
    name: NonEmptyStr
    environment: Literal["local"]


class LLMSettings(_Settings):
    provider: Literal["anthropic"]
    model: NonEmptyStr
    max_output_tokens: Annotated[StrictInt, Field(gt=0)]


class RuntimeSettings(_Settings):
    workspace: Path
    # StrictInt: rejects "10", 10.5 and booleans instead of coercing them.
    max_iterations: Annotated[StrictInt, Field(gt=0)]

    @field_validator("workspace", mode="before")
    @classmethod
    def _workspace_not_empty(cls, value: object) -> object:
        # Path("") silently becomes Path("."), so reject blank strings before conversion.
        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be empty")
        return value


class FactoryConfig(_Settings):
    factory: FactorySettings
    llm: LLMSettings
    runtime: RuntimeSettings
