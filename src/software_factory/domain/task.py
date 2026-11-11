from enum import Enum
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TaskStatus(Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: NonEmptyStr
    title: NonEmptyStr
    description: NonEmptyStr
    created_at: AwareDatetime
    status: TaskStatus = TaskStatus.QUEUED
