from typing import Annotated

from pydantic import StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Model = NonEmptyStr
type JSONValue = (
    str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
)
