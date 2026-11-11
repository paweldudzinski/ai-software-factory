from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from software_factory.domain.task import Task, TaskStatus

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


def make_task(**overrides: Any) -> Task:
    data: dict[str, Any] = {
        "id": "task-1",
        "title": "Write tests",
        "description": "Cover the Task model with tests",
        "created_at": NOW,
    }
    data.update(overrides)
    return Task(**data)


def test_creates_valid_task() -> None:
    task = make_task()

    assert task.id == "task-1"
    assert task.title == "Write tests"
    assert task.description == "Cover the Task model with tests"
    assert task.created_at == NOW
    assert task.status is TaskStatus.QUEUED


def test_accepts_non_utc_timezone() -> None:
    created_at = datetime(2026, 11, 11, 13, 0, tzinfo=timezone(timedelta(hours=1)))

    task = make_task(created_at=created_at)

    assert task.created_at.utcoffset() == timedelta(hours=1)


def test_parses_iso_string_with_offset() -> None:
    task = make_task(created_at="2026-11-11T12:00:00Z")

    assert task.created_at == NOW


@pytest.mark.parametrize("field", ["id", "title", "description"])
@pytest.mark.parametrize("value", ["", "   "])
def test_rejects_empty_string(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match=field):
        make_task(**{field: value})


@pytest.mark.parametrize(
    "created_at",
    [datetime(2026, 11, 11, 12, 0), "2026-11-11T12:00:00"],
)
def test_rejects_naive_datetime(created_at: datetime | str) -> None:
    with pytest.raises(ValidationError, match="timezone"):
        make_task(created_at=created_at)


def test_is_immutable() -> None:
    task = make_task()

    with pytest.raises(ValidationError):
        task.title = "Changed"
