import os
from pathlib import Path
from typing import Any


TEXT_EXTENSIONS = {".md", ".txt", ".py", ".json", ".toml", ".yaml", ".yml", ".js", ".ts"}


def scan_workspace(root_dir: str) -> dict[str, Any]:
    root = Path(root_dir).resolve()
    files: list[str] = []
    tree: dict[str, list[str]] = {}

    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules", ".mantis"}]
        rel_root = os.path.relpath(current_root, root)
        rel_root = "." if rel_root == "." else rel_root
        sorted_files = sorted(filenames)
        tree[rel_root] = sorted_files
        for filename in sorted_files:
            files.append(str(Path(rel_root) / filename if rel_root != "." else Path(filename)))

    return {"root": str(root), "file_count": len(files), "files": files, "tree": tree}


def summarize_repository(root_dir: str = ".") -> str:
    root = Path(root_dir).resolve()
    summary_parts: list[str] = [f"Repository: {root.name}"]

    readme_content = _read_if_exists(root / "README.md")
    if readme_content:
        summary_parts.append(f"README summary: {readme_content[:1200]}")

    package_json = _read_if_exists(root / "package.json")
    if package_json:
        summary_parts.append(f"package.json excerpt: {package_json[:600]}")

    pyproject = _read_if_exists(root / "pyproject.toml")
    if pyproject:
        summary_parts.append(f"pyproject.toml excerpt: {pyproject[:600]}")

    scan = scan_workspace(str(root))
    top_files = ", ".join(scan["files"][:50]) if scan["files"] else "none"
    summary_parts.append(f"File count: {scan['file_count']}")
    summary_parts.append(f"Top files: {top_files}")

    return "\n".join(summary_parts)


def index_files_into_memory(memory_manager, root_dir: str = ".") -> str:
    root = Path(root_dir).resolve()
    summary = summarize_repository(str(root))
    memory_manager.store_memory(
        summary,
        {
            "source": "workspace",
            "type": "project",
            "root": str(root),
        },
    )
    return summary


def _read_if_exists(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
