import os

from providers.base import LLMProvider
from providers.claude import ClaudeProvider
from providers.ollama import OllamaProvider
from providers.openai import OpenAIProvider


def get_provider(model_name: str | None = None) -> LLMProvider:
    model = (model_name or "").strip().lower()
    if model.startswith("gpt"):
        return OpenAIProvider()
    if model.startswith("claude"):
        return ClaudeProvider()
    return OllamaProvider()


class ProviderRouter:
    def __init__(self, default_model: str | None = None) -> None:
        self.default_model = default_model or os.getenv("MANTIS_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        self._openai = OpenAIProvider()
        self._claude = ClaudeProvider()
        self._ollama = OllamaProvider(model=self.default_model)

    def _select(self, model_name: str | None) -> LLMProvider:
        model = (model_name or self.default_model or "").lower()
        if model.startswith("gpt"):
            return self._openai
        if model.startswith("claude"):
            return self._claude
        return self._ollama

    async def chat(self, messages, model: str | None = None) -> str:
        provider = self._select(model)
        target_model = model or self.default_model
        return await provider.chat(messages, model=target_model)
