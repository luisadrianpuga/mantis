import inspect
from typing import Any, Callable, Dict, List

from tools import http, python_exec


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {
            "python": {
                "name": "python",
                "description": "Execute Python code locally. WARNING: runs with full system access.",
                "handler": python_exec.run,
            },
            "http": {
                "name": "http",
                "description": "Fetch text content from a URL.",
                "handler": http.fetch,
            },
        }

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": spec["name"], "description": spec["description"]} for spec in self.tools.values()]

    async def run(self, name: str, tool_input: str) -> str:
        spec = self.tools.get(name)
        if not spec:
            return f"Unknown tool: {name}"
        handler: Callable = spec["handler"]
        if inspect.iscoroutinefunction(handler):
            return await handler(tool_input)
        return handler(tool_input)
