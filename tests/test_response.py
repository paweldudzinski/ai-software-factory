from typing import Any

import pytest
from pydantic import ValidationError

from software_factory.domain.llm.response import LLMResponse, LLMUsage, StopReason

USAGE = LLMUsage(input_tokens=120, output_tokens=30)


def make_response(**overrides: Any) -> LLMResponse:
    data: dict[str, Any] = {
        "content": "Hello!",
        "model": "claude-sonnet-5-5",
        "stop_reason": StopReason.END_TURN,
        "usage": USAGE,
        "request_id": "req_018EeWyXxfu5pfWkrYcMdjWG",
    }
    data.update(overrides)
    return LLMResponse(**data)


# --- LLMUsage ---------------------------------------------------------------


def test_usage_computes_total_tokens() -> None:
    assert USAGE.total_tokens == 150


def test_usage_total_tokens_is_serialized() -> None:
    assert USAGE.model_dump() == {
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
    }


def test_usage_accepts_zero_tokens() -> None:
    usage = LLMUsage(input_tokens=0, output_tokens=0)

    assert usage.total_tokens == 0


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("value", [-1, "10", 1.5, True, None])
def test_usage_rejects_invalid_token_count(field: str, value: Any) -> None:
    data: dict[str, Any] = {"input_tokens": 1, "output_tokens": 1, field: value}

    with pytest.raises(ValidationError, match=field):
        LLMUsage(**data)


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_usage_rejects_missing_field(field: str) -> None:
    data: dict[str, Any] = {"input_tokens": 1, "output_tokens": 1}
    del data[field]

    with pytest.raises(ValidationError, match=rf"{field}\s+Field required"):
        LLMUsage(**data)


def test_usage_is_immutable() -> None:
    with pytest.raises(ValidationError):
        USAGE.input_tokens = 1


# --- LLMResponse ------------------------------------------------------------


def test_creates_valid_response() -> None:
    response = make_response()

    assert response.content == "Hello!"
    assert response.model == "claude-sonnet-5-5"
    assert response.stop_reason is StopReason.END_TURN
    assert response.usage == USAGE
    assert response.request_id == "req_018EeWyXxfu5pfWkrYcMdjWG"


def test_parses_from_plain_data() -> None:
    response = LLMResponse.model_validate(
        {
            "content": "Hello!",
            "model": "claude-sonnet-5-5",
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 120, "output_tokens": 30},
            "request_id": "req_1",
        }
    )

    assert response.stop_reason is StopReason.MAX_TOKENS
    assert response.usage.total_tokens == 150


@pytest.mark.parametrize("stop_reason", list(StopReason))
def test_accepts_every_stop_reason(stop_reason: StopReason) -> None:
    response = make_response(stop_reason=stop_reason.value)

    assert response.stop_reason is stop_reason


@pytest.mark.parametrize("stop_reason", ["", "finished", "END_TURN", None])
def test_rejects_invalid_stop_reason(stop_reason: Any) -> None:
    with pytest.raises(ValidationError, match="stop_reason"):
        make_response(stop_reason=stop_reason)


def test_allows_empty_content() -> None:
    response = make_response(content="", stop_reason=StopReason.REFUSAL)

    assert response.content == ""


def test_preserves_content_whitespace() -> None:
    response = make_response(content="  indented\n")

    assert response.content == "  indented\n"


@pytest.mark.parametrize("model", ["", "   "])
def test_rejects_empty_model(model: str) -> None:
    with pytest.raises(ValidationError, match="model"):
        make_response(model=model)


def test_request_id_is_optional() -> None:
    response = make_response(request_id=None)

    assert response.request_id is None


@pytest.mark.parametrize("request_id", ["", "   "])
def test_rejects_empty_request_id(request_id: str) -> None:
    with pytest.raises(ValidationError, match="request_id"):
        make_response(request_id=request_id)


@pytest.mark.parametrize("field", ["content", "model", "stop_reason", "usage"])
def test_rejects_missing_required_field(field: str) -> None:
    data: dict[str, Any] = {
        "content": "Hello!",
        "model": "claude-sonnet-5-5",
        "stop_reason": StopReason.END_TURN,
        "usage": USAGE,
    }
    del data[field]

    with pytest.raises(ValidationError, match=rf"{field}\s+Field required"):
        LLMResponse(**data)


def test_rejects_invalid_usage() -> None:
    with pytest.raises(ValidationError, match=r"usage\.input_tokens"):
        make_response(usage={"input_tokens": -1, "output_tokens": 0})


def test_response_is_immutable() -> None:
    response = make_response()

    with pytest.raises(ValidationError):
        response.content = "Changed"
