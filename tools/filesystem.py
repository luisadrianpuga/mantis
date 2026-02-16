import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import MANTIS_ALLOW_FILE_WRITE, MANTIS_SANDBOX

CHANGE_LOG_DIR = Path(".mantis") / "changes"
WORKSPACE_ROOT = Path(".").resolve()
SANDBOX_ROOT = (WORKSPACE_ROOT / "workspace").resolve()
DENYLIST = {
    ".env",
    ".git/config",
}


def _json_input(tool_input: str) -> Dict[str, Any]:
    try:
        payload = json.loads(tool_input)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected JSON tool input. Error: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object input.")
    return payload


def _ensure_writes_allowed() -> None:
    if not MANTIS_ALLOW_FILE_WRITE:
        raise PermissionError("File write tools are disabled. Set MANTIS_ALLOW_FILE_WRITE=true to enable.")


def _resolve_path(path: str) -> Path:
    normalized = path.strip()
    if normalized in DENYLIST:
        raise ValueError(f"Path is denied by policy: {path}")
    if normalized.startswith("~/") or normalized == "~":
        raise ValueError(f"Path is denied by policy: {path}")
    if Path(normalized).is_absolute():
        raise ValueError(f"Absolute paths are not allowed: {path}")

    candidate = (WORKSPACE_ROOT / path).resolve()
    if WORKSPACE_ROOT not in candidate.parents and candidate != WORKSPACE_ROOT:
        raise ValueError(f"Path escapes workspace root: {path}")
    if MANTIS_SANDBOX:
        if candidate == SANDBOX_ROOT:
            return candidate
        if SANDBOX_ROOT not in candidate.parents:
            raise ValueError(f"Sandbox mode restricts writes to workspace/: {path}")
    return candidate


def _ensure_log_dir() -> None:
    CHANGE_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_change(action: str, path: str, before: str, after: str) -> None:
    _ensure_log_dir()
    now_utc = datetime.now(timezone.utc)
    now = now_utc.strftime("%Y%m%dT%H%M%S%f")
    log_path = CHANGE_LOG_DIR / f"{now}_{action}.json"
    payload = {
        "timestamp": now_utc.isoformat(),
        "action": action,
        "path": path,
        "before": before,
        "after": after,
    }
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_file(tool_input: str) -> str:
    try:
        _ensure_writes_allowed()
        payload = _json_input(tool_input)
        path = str(payload.get("path", "")).strip()
        content = str(payload.get("content", ""))
        if not path:
            return "create_file requires JSON: {\"path\": \"...\", \"content\": \"...\"}"

        file_path = _resolve_path(path)
        if file_path.exists():
            return f"create_file refused: file already exists at {path}"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        _log_change("create_file", path, "", content)
        return f"Created file: {path}"
    except Exception as exc:
        return f"create_file error: {exc}"


def write_file(tool_input: str) -> str:
    try:
        _ensure_writes_allowed()
        payload = _json_input(tool_input)
        path = str(payload.get("path", "")).strip()
        content = str(payload.get("content", ""))
        overwrite = bool(payload.get("overwrite", False))
        if not path:
            return "write_file requires JSON: {\"path\": \"...\", \"content\": \"...\", \"overwrite\": true}"

        file_path = _resolve_path(path)
        if file_path.exists() and not overwrite:
            return f"write_file refused: {path} exists. Set overwrite=true for explicit intent."

        before = ""
        if file_path.exists():
            before = file_path.read_text(encoding="utf-8")
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        _log_change("write_file", path, before, content)
        return f"Wrote file: {path}"
    except Exception as exc:
        return f"write_file error: {exc}"


def patch_file(tool_input: str) -> str:
    try:
        _ensure_writes_allowed()
        payload = _json_input(tool_input)
        path = str(payload.get("path", "")).strip()
        diff = str(payload.get("diff", ""))
        if not path or not diff:
            return "patch_file requires JSON: {\"path\": \"...\", \"diff\": \"unified diff\"}"

        file_path = _resolve_path(path)
        if not file_path.exists():
            return f"patch_file refused: file does not exist at {path}"

        before = file_path.read_text(encoding="utf-8")
        after = _apply_unified_diff(before, diff)
        if after == before:
            return f"patch_file applied no changes to {path}"

        file_path.write_text(after, encoding="utf-8")
        _log_change("patch_file", path, before, after)
        return f"Patched file: {path}"
    except Exception as exc:
        return f"patch_file error: {exc}"


def delete_file(tool_input: str) -> str:
    try:
        _ensure_writes_allowed()
        payload = _json_input(tool_input)
        path = str(payload.get("path", "")).strip()
        confirm = bool(payload.get("confirm", False))
        if not path:
            return "delete_file requires JSON: {\"path\": \"...\", \"confirm\": true}"
        if not confirm:
            return "delete_file refused: set confirm=true for explicit intent."

        file_path = _resolve_path(path)
        if not file_path.exists():
            return f"delete_file skipped: file not found at {path}"

        before = file_path.read_text(encoding="utf-8")
        os.remove(file_path)
        _log_change("delete_file", path, before, "")
        return f"Deleted file: {path}"
    except Exception as exc:
        return f"delete_file error: {exc}"


def _parse_hunk_header(header: str) -> Tuple[int, int, int, int]:
    # Example: @@ -3,2 +3,3 @@
    parts = header.split("@@")
    if len(parts) < 3:
        raise ValueError(f"Invalid hunk header: {header}")
    body = parts[1].strip()
    old_part, new_part = body.split()
    old_start, old_count = _parse_range(old_part)
    new_start, new_count = _parse_range(new_part)
    return old_start, old_count, new_start, new_count


def _parse_range(token: str) -> Tuple[int, int]:
    # token format: -l,s or +l,s or -l or +l
    token = token[1:]
    if "," in token:
        start, count = token.split(",", 1)
        return int(start), int(count)
    return int(token), 1


def _apply_unified_diff(original: str, diff_text: str) -> str:
    original_lines = original.splitlines(keepends=True)
    diff_lines = diff_text.splitlines(keepends=True)

    # Skip file header lines.
    index = 0
    while index < len(diff_lines) and not diff_lines[index].startswith("@@"):
        index += 1

    result: List[str] = []
    source_index = 0

    while index < len(diff_lines):
        header = diff_lines[index].rstrip("\n")
        if not header.startswith("@@"):
            raise ValueError(f"Expected hunk header, got: {header}")
        old_start, _, _, _ = _parse_hunk_header(header)
        old_start_index = old_start - 1

        while source_index < old_start_index:
            result.append(original_lines[source_index])
            source_index += 1

        index += 1
        while index < len(diff_lines) and not diff_lines[index].startswith("@@"):
            line = diff_lines[index]
            if not line:
                index += 1
                continue
            marker = line[:1]
            body = line[1:]
            if marker == " ":
                if source_index >= len(original_lines):
                    raise ValueError("Patch context exceeds source length.")
                if original_lines[source_index] != body:
                    raise ValueError("Patch context mismatch.")
                result.append(original_lines[source_index])
                source_index += 1
            elif marker == "-":
                if source_index >= len(original_lines):
                    raise ValueError("Patch removal exceeds source length.")
                if original_lines[source_index] != body:
                    raise ValueError("Patch removal mismatch.")
                source_index += 1
            elif marker == "+":
                result.append(body)
            elif marker == "\\":
                # Handle "\\ No newline at end of file"
                pass
            else:
                raise ValueError(f"Unsupported diff marker: {marker}")
            index += 1

    result.extend(original_lines[source_index:])
    return "".join(result)
