from pathlib import Path

import pytest
from typer.testing import CliRunner

from software_factory.application.ports.llm_provider import LLMProviderError
from software_factory.application.use_cases.ask_llm import AskLLM
from software_factory.cli.app import app
from software_factory.domain.llm.message import LLMMessage, MessageRole
from software_factory.domain.llm.response import LLMResponse, LLMUsage, StopReason
from software_factory.infrastructure.config.schema import FactoryConfig
from tests.test_ask_llm import FakeProvider
from tests.test_cli import SAMPLE_CONFIG

RESPONSE = LLMResponse(
    content="Event-driven architecture is...",
    model="claude-sonnet-5-5",
    stop_reason=StopReason.END_TURN,
    usage=LLMUsage(input_tokens=12, output_tokens=150),
    request_id="req_018EeWyXxfu5pfWkrYcMdjWG",
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "factory.yaml"
    path.write_text(SAMPLE_CONFIG)
    return path


def use_provider(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> None:
    """Replace the composition root so no real LLM is called."""

    def build_ask_llm(config: FactoryConfig) -> AskLLM:
        return AskLLM(provider)

    monkeypatch.setattr("software_factory.cli.app.build_ask_llm", build_ask_llm)


def ask(runner: CliRunner, config_file: Path, prompt: str) -> tuple[int, str]:
    result = runner.invoke(app, ["ask", prompt, "--config", str(config_file)])
    return result.exit_code, result.output


def test_prints_response(
    runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_provider(monkeypatch, FakeProvider(RESPONSE))

    exit_code, output = ask(runner, config_file, "Explain event-driven architecture")

    assert exit_code == 0, output
    assert output == (
        "\n"
        "AI Software Factory\n"
        "-------------------\n"
        "\n"
        "Provider: Anthropic\n"
        "Model: claude-sonnet-5-5\n"
        "\n"
        "Response:\n"
        "Event-driven architecture is...\n"
        "\n"
        "-------------------\n"
        "Input tokens: 12\n"
        "Output tokens: 150\n"
        "Request ID: req_018EeWyXxfu5pfWkrYcMdjWG\n"
    )


def test_builds_request_from_config_and_prompt(
    runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(RESPONSE)
    use_provider(monkeypatch, provider)

    ask(runner, config_file, "Explain event-driven architecture")

    [request] = provider.requests
    assert request.model == "test-model"
    assert request.max_tokens == 1024
    assert request.system_prompt is None
    assert request.messages == [
        LLMMessage(role=MessageRole.USER, content="Explain event-driven architecture")
    ]


def test_warns_when_response_stopped_early(
    runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = RESPONSE.model_copy(update={"stop_reason": StopReason.MAX_TOKENS})
    use_provider(monkeypatch, FakeProvider(response))

    exit_code, output = ask(runner, config_file, "Hi")

    assert exit_code == 0, output
    assert "Warning: response stopped early (stop reason: max_tokens)" in output


def test_handles_response_without_text(
    runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = RESPONSE.model_copy(
        update={"content": "", "stop_reason": StopReason.REFUSAL, "request_id": None}
    )
    use_provider(monkeypatch, FakeProvider(response))

    exit_code, output = ask(runner, config_file, "Hi")

    assert exit_code == 0, output
    assert "(no text in response)" in output
    assert "stop reason: refusal" in output
    assert "Request ID: n/a" in output


def test_reports_provider_error_with_request_id(
    runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = LLMProviderError("Anthropic API error 529: Overloaded", request_id="req_9")
    use_provider(monkeypatch, FakeProvider(error=error))

    exit_code, output = ask(runner, config_file, "Hi")

    assert exit_code == 1
    assert "Anthropic API error 529: Overloaded" in output
    assert "Request ID: req_9" in output


def test_reports_error_from_composition_root(
    runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build_ask_llm(config: FactoryConfig) -> AskLLM:
        raise LLMProviderError("Cannot create Anthropic client: no credentials")

    monkeypatch.setattr("software_factory.cli.app.build_ask_llm", build_ask_llm)

    exit_code, output = ask(runner, config_file, "Hi")

    assert exit_code == 1
    assert "Cannot create Anthropic client" in output


@pytest.mark.parametrize("prompt", ["", "   "])
def test_rejects_empty_prompt(
    runner: CliRunner, config_file: Path, monkeypatch: pytest.MonkeyPatch, prompt: str
) -> None:
    provider = FakeProvider(RESPONSE)
    use_provider(monkeypatch, provider)

    exit_code, output = ask(runner, config_file, prompt)

    assert exit_code == 1
    assert "Prompt must not be empty" in output
    assert provider.requests == []


def test_fails_on_invalid_config(runner: CliRunner, tmp_path: Path) -> None:
    exit_code, output = ask(runner, tmp_path / "missing.yaml", "Hi")

    assert exit_code == 1
    assert "Config file not found" in output
