from typing import Any, Dict, List, Protocol


class LLMProvider(Protocol):
    async def chat(self, messages: List[Dict[str, Any]], model: str | None = None) -> str:
        ...
