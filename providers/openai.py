import os
from typing import Any, Dict, List

import httpx


class OpenAIProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    async def chat(self, messages: List[Dict[str, Any]], model: str | None = None) -> str:
        if not self.api_key:
            return "[LLM error: OPENAI_API_KEY is not set]"

        payload = {
            "model": model or self.model,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    return ""
                message = choices[0].get("message") or {}
                return message.get("content", "").strip()
        except Exception as exc:  # pragma: no cover
            return f"[LLM error: {exc}]"
