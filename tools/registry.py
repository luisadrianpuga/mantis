import inspect
from typing import Any, Callable, Dict, List

from tools import filesystem, http, python_exec


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
            "read_file": {
                "name": "read_file",
                "description": "Read a UTF-8 text file from the workspace.",
                "handler": filesystem.read_file,
            },
            "write_file": {
                "name": "write_file",
                "description": "Write content to a file in the workspace. Input should include path and content.",
                "handler": filesystem.write_file,
            },
            "list_files": {
                "name": "list_files",
                "description": "List files and directories under a workspace path.",
                "handler": filesystem.list_files,
            },
            "search_files": {
                "name": "search_files",
                "description": "Search for text across files in the workspace.",
                "handler": filesystem.search_files,
            },
        }

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": spec["name"], "description": spec["description"]} for spec in self.tools.values()]

    async def run(self, name: str, tool_input: str) -> str:
        spec = self.tools.get(name)
        if not spec:
            return f"Unknown tool: {name}"
        handler: Callable = spec["handler"]
        try:
            if inspect.iscoroutinefunction(handler):
                return await handler(tool_input)
            return handler(tool_input)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            return f"Tool execution error ({name}): {exc}"
