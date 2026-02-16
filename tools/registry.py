import inspect
from typing import Any, Callable, Dict, List

from tools import http, python_exec
from tools import workspace_tools, filesystem, tests


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
            "workspace.tree": {
                "name": "workspace.tree",
                "description": "Return repository file tree metadata (path, size, language). Input can be empty.",
                "handler": workspace_tools.tree,
            },
            "workspace.read_file": {
                "name": "workspace.read_file",
                "description": "Read a UTF-8 workspace file. Input: relative file path.",
                "handler": workspace_tools.read_file,
            },
            "workspace.search": {
                "name": "workspace.search",
                "description": "Search workspace paths and text content. Input: query string.",
                "handler": workspace_tools.search,
            },
            "create_file": {
                "name": "create_file",
                "description": "Create a new file. Input JSON: {\"path\":\"...\",\"content\":\"...\"}.",
                "handler": filesystem.create_file,
            },
            "write_file": {
                "name": "write_file",
                "description": "Write file content. Existing files require explicit overwrite flag. Input JSON: {\"path\":\"...\",\"content\":\"...\",\"overwrite\":true}.",
                "handler": filesystem.write_file,
            },
            "patch_file": {
                "name": "patch_file",
                "description": "Apply unified diff to an existing file. Input JSON: {\"path\":\"...\",\"diff\":\"...\"}.",
                "handler": filesystem.patch_file,
            },
            "delete_file": {
                "name": "delete_file",
                "description": "Delete a file only with explicit confirmation. Input JSON: {\"path\":\"...\",\"confirm\":true}.",
                "handler": filesystem.delete_file,
            },
            "run_tests": {
                "name": "run_tests",
                "description": "Auto-detect and execute tests (pytest, npm test, make test). Input can be empty.",
                "handler": tests.run_tests,
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
