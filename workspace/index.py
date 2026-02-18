import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".md": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".sh": "shell",
    ".html": "html",
    ".css": "css",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
}

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}


@dataclass
class WorkspaceFile:
    path: str
    size: int
    language: str


class WorkspaceIndex:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.snapshot: List[WorkspaceFile] = []
        self.refresh()

    def refresh(self) -> None:
        files: List[WorkspaceFile] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in filenames:
                full_path = Path(dirpath) / filename
                try:
                    relative = full_path.relative_to(self.root).as_posix()
                    stat = full_path.stat()
                except OSError:
                    continue
                files.append(
                    WorkspaceFile(
                        path=relative,
                        size=stat.st_size,
                        language=self._detect_language(full_path),
                    )
                )
        self.snapshot = sorted(files, key=lambda item: item.path)

    def tree(self) -> str:
        self.refresh()
        payload: Dict[str, Any] = {
            "root": str(self.root),
            "file_count": len(self.snapshot),
            "files": [item.__dict__ for item in self.snapshot],
        }
        return json.dumps(payload, indent=2)

    def read_file(self, path: str) -> str:
        resolved = self._resolve_path(path)
        try:
            return resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Binary file cannot be decoded as UTF-8: {path}"
        except OSError as exc:
            return f"Workspace read error: {exc}"

    def search(self, query: str, max_hits: int = 20) -> str:
        self.refresh()
        lowered = query.lower().strip()
        if not lowered:
            return json.dumps({"query": query, "matches": []}, indent=2)

        matches: List[Dict[str, Any]] = []
        for item in self.snapshot:
            if lowered in item.path.lower():
                matches.append({"path": item.path, "type": "path"})
                if len(matches) >= max_hits:
                    break

        if len(matches) < max_hits:
            for item in self.snapshot:
                if any(existing.get("path") == item.path for existing in matches):
                    continue
                if item.size > 512_000:
                    continue
                resolved = self._resolve_path(item.path)
                try:
                    text = resolved.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                index = text.lower().find(lowered)
                if index >= 0:
                    start = max(0, index - 80)
                    end = min(len(text), index + len(lowered) + 80)
                    snippet = text[start:end].replace("\n", " ").strip()
                    matches.append({"path": item.path, "type": "content", "snippet": snippet})
                    if len(matches) >= max_hits:
                        break

        return json.dumps({"query": query, "matches": matches}, indent=2)

    def _resolve_path(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise ValueError(f"Path escapes workspace root: {path}")
        return candidate

    @staticmethod
    def _detect_language(path: Path) -> str:
        return LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "unknown")
