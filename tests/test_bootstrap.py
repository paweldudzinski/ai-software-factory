import pytest

from software_factory.application.use_cases.ask_llm import AskLLM
from software_factory.bootstrap import build_ask_llm, build_llm_provider
from software_factory.infrastructure.config.schema import FactoryConfig
from software_factory.infrastructure.llm.anthropic_provider import AnthropicProvider

CONFIG = FactoryConfig.model_validate(
    {
        "factory": {"name": "test-factory", "environment": "local"},
        "llm": {
            "provider": "anthropic",
            "model": "claude-sonnet-5-5",
            "max_output_tokens": 1024,
        },
        "runtime": {"workspace": "./workspace", "max_iterations": 3},
    }
)


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def test_builds_anthropic_provider() -> None:
    assert isinstance(build_llm_provider(CONFIG), AnthropicProvider)


def test_builds_ask_llm_with_configured_provider() -> None:
    use_case = build_ask_llm(CONFIG)

    assert isinstance(use_case, AskLLM)
    assert isinstance(use_case._provider, AnthropicProvider)
