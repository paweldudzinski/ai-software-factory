from software_factory.application.ports.llm_provider import LLMProvider
from software_factory.domain.llm.request import LLMRequest
from software_factory.domain.llm.response import LLMResponse


class AskLLM:
    """
    Use case: send a single request to an LLM and return its response.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def execute(self, request: LLMRequest) -> LLMResponse:
        return await self._provider.complete(request)
