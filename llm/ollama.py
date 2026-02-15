import os
from typing import List, Dict, Any

import httpx


class OllamaClient:
    """
    Minimal wrapper around the local Ollama HTTP API using OpenAI-style messages.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
        self.timeout = timeout

    async def chat(self, messages: List[Dict[str, Any]], model: str | None = None) -> str:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                message = data.get("message") or {}
                return message.get("content", "").strip()
        except Exception as exc:  # pragma: no cover - defensive guardrail
            return f"[LLM error: {exc}]"
