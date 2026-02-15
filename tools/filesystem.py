import json
import os
from pathlib import Path


MAX_READ_CHARS = 12_000


def read_file(tool_input: str) -> str:
    payload = _parse_payload(tool_input)
    path = payload.get("path") or tool_input.strip()
    if not path:
        return "read_file error: missing path"

    target = _resolve_workspace_path(path)
    if not target.exists() or not target.is_file():
        return f"read_file error: file not found: {path}"

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = target.read_text(encoding="utf-8", errors="replace")
    return content[:MAX_READ_CHARS]


def write_file(tool_input: str) -> str:
    payload = _parse_payload(tool_input)
    path = payload.get("path", "")
    content = payload.get("content", "")

    if not path:
        lines = tool_input.splitlines()
        if lines:
            path = lines[0].strip()
            content = "\n".join(lines[1:])

    if not path:
        return "write_file error: missing path"

    target = _resolve_workspace_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


def list_files(tool_input: str) -> str:
    payload = _parse_payload(tool_input)
    path = payload.get("path") or tool_input.strip() or "."
    target = _resolve_workspace_path(path)

    if not target.exists() or not target.is_dir():
        return f"list_files error: directory not found: {path}"

    entries = sorted(item.name for item in target.iterdir())
    return "\n".join(entries[:500]) if entries else "(empty directory)"


def search_files(tool_input: str) -> str:
    payload = _parse_payload(tool_input)
    query = payload.get("query") or tool_input.strip()
    base_path = payload.get("path") or "."
    if not query:
        return "search_files error: missing query"

    base_dir = _resolve_workspace_path(base_path)
    if not base_dir.exists() or not base_dir.is_dir():
        return f"search_files error: directory not found: {base_path}"

    matches: list[str] = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", "__pycache__", ".venv", ".mantis"}]
        for name in files:
            file_path = Path(root) / name
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if query.lower() in text.lower():
                rel_path = file_path.relative_to(Path.cwd())
                matches.append(str(rel_path))
            if len(matches) >= 200:
                return "\n".join(matches)
    return "\n".join(matches) if matches else "No matches found."


def _parse_payload(tool_input: str) -> dict:
    stripped = tool_input.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {}


def _resolve_workspace_path(path: str) -> Path:
    workspace_root = Path.cwd().resolve()
    target = (workspace_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()

    if workspace_root not in target.parents and target != workspace_root:
        raise ValueError(f"Path outside workspace is not allowed: {path}")

    return target
