import os
from typing import Any, Dict, List

import httpx
from providers.detection import has_anthropic, has_ollama, has_openai


def _auto_detect_default_model() -> str | None:
    if has_openai():
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if has_anthropic():
        return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    if has_ollama():
        return os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    return None


class ProviderRouter:
    """
    Minimal provider router.
    Routes by model prefix:
    - gpt* -> OpenAI
    - claude* -> Anthropic
    - fallback -> Ollama
    """

    def __init__(self) -> None:
        self.default_model = os.getenv("MANTIS_MODEL") or _auto_detect_default_model()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

    async def chat(self, messages: List[Dict[str, Any]], model: str | None = None) -> str:
        target_model = model or self.default_model
        if not target_model:
            return "[Mantis error: No LLM provider configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or install Ollama.]"
        lowered = target_model.lower()
        if lowered.startswith("gpt"):
            return await self._openai_chat(messages, target_model)
        if lowered.startswith("claude"):
            return await self._anthropic_chat(messages, target_model)
        return await self._ollama_chat(messages, target_model)

    async def _ollama_chat(self, messages: List[Dict[str, Any]], model: str) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(base_url=self.ollama_base_url, timeout=120.0) as client:
                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                return (data.get("message") or {}).get("content", "").strip()
        except Exception as exc:
            return f"[LLM error: {exc}]"

    async def _openai_chat(self, messages: List[Dict[str, Any]], model: str) -> str:
        if not self.openai_api_key:
            return "[LLM error: OPENAI_API_KEY is not set]"

        payload = {"model": model or self.openai_model, "messages": messages}
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return ""
                return (choices[0].get("message") or {}).get("content", "").strip()
        except Exception as exc:
            return f"[LLM error: {exc}]"

    async def _anthropic_chat(self, messages: List[Dict[str, Any]], model: str) -> str:
        if not self.anthropic_api_key:
            return "[LLM error: ANTHROPIC_API_KEY is not set]"

        system = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
        chat_messages = []
        for m in messages:
            role = m.get("role", "user")
            if role in {"user", "assistant"}:
                chat_messages.append({"role": role, "content": m.get("content", "")})

        payload: Dict[str, Any] = {
            "model": model or self.anthropic_model,
            "max_tokens": 1000,
            "messages": chat_messages,
        }
        if system:
            payload["system"] = system

        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                blocks = data.get("content", [])
                text = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text")
                return text.strip()
        except Exception as exc:
            return f"[LLM error: {exc}]"
