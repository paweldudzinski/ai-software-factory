from enum import Enum

from pydantic import AwareDatetime, BaseModel, ConfigDict

from .types import NonEmptyStr


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
