"""
Composition root: the only place that picks concrete implementations
(infrastructure adapters) for the application's ports.
"""

from software_factory.application.ports.llm_provider import LLMProvider
from software_factory.application.use_cases.ask_llm import AskLLM
from software_factory.infrastructure.config.schema import FactoryConfig
from software_factory.infrastructure.llm.anthropic_provider import AnthropicProvider


def build_llm_provider(config: FactoryConfig) -> LLMProvider:
    match config.llm.provider:
        case "anthropic":
            return AnthropicProvider()


def build_ask_llm(config: FactoryConfig) -> AskLLM:
    return AskLLM(provider=build_llm_provider(config))
