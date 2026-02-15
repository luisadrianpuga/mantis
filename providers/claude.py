import os
from typing import Any, Dict, List

import httpx


class ClaudeProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")).rstrip("/")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        self.timeout = timeout

    async def chat(self, messages: List[Dict[str, Any]], model: str | None = None) -> str:
        if not self.api_key:
            return "[LLM error: ANTHROPIC_API_KEY is not set]"

        system_messages: list[str] = []
        anthro_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            if role == "system":
                system_messages.append(content)
            elif role in {"assistant", "user"}:
                anthro_messages.append({"role": role, "content": content})
            else:
                anthro_messages.append({"role": "user", "content": content})

        payload: Dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": 1000,
            "messages": anthro_messages,
        }
        if system_messages:
            payload["system"] = "\n".join(system_messages)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/v1/messages", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content_blocks = data.get("content", [])
                texts = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
                return "\n".join(texts).strip()
        except Exception as exc:  # pragma: no cover
            return f"[LLM error: {exc}]"
