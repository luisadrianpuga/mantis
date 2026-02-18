import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List
import difflib


def _run_git(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def _ensure_repo() -> str | None:
    probe = _run_git(["rev-parse", "--is-inside-work-tree"])
    if probe.returncode != 0 or "true" not in probe.stdout.lower():
        return "Git tool error: workspace is not a git repository."
    return None


def _json_input(tool_input: str) -> Dict[str, Any]:
    tool_input = (tool_input or "").strip()
    if not tool_input:
        return {}
    try:
        payload = json.loads(tool_input)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Expected JSON input. Error: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object input.")
    return payload


def status(_: str = "") -> str:
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    result = _run_git(["status", "--porcelain"])
    if result.returncode != 0:
        return f"git.status error: {result.stderr.strip() or result.stdout.strip()}"
    return result.stdout.strip() or "Clean working tree."


def create_branch(tool_input: str) -> str:
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    try:
        payload = _json_input(tool_input)
    except Exception as exc:
        return f"git.create_branch error: {exc}"

    name = str(payload.get("name", "")).strip() or _default_branch_name(payload.get("goal", ""))
    safe_name = _sanitize_branch_name(name)
    if not safe_name:
        return "git.create_branch error: branch name is required."

    result = _run_git(["checkout", "-b", safe_name])
    if result.returncode != 0 and "already exists" in (result.stderr or ""):
        result = _run_git(["checkout", safe_name])

    if result.returncode != 0:
        return f"git.create_branch error: {result.stderr.strip() or result.stdout.strip()}"
    return f"Checked out branch: {safe_name}"


def commit(tool_input: str) -> str:
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    try:
        payload = _json_input(tool_input)
    except Exception as exc:
        return f"git.commit error: {exc}"

    message = str(payload.get("message", "")).strip()
    if not message:
        return "git.commit error: message is required."

    add = _run_git(["add", "-A"])
    if add.returncode != 0:
        return f"git.commit error during add: {add.stderr.strip() or add.stdout.strip()}"

    status_result = _run_git(["status", "--porcelain"])
    if status_result.returncode != 0:
        return f"git.commit error during status: {status_result.stderr.strip() or status_result.stdout.strip()}"
    if not status_result.stdout.strip():
        return "git.commit skipped: no changes to commit."

    commit_result = _run_git(["commit", "-m", message])
    if commit_result.returncode != 0:
        return f"git.commit error: {commit_result.stderr.strip() or commit_result.stdout.strip()}"
    return commit_result.stdout.strip() or "Commit created."


def diff(tool_input: str) -> str:
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error

    tool_input = (tool_input or "").strip()
    if not tool_input:
        result = _run_git(["diff"])
        if result.returncode != 0:
            return f"git.diff error: {result.stderr.strip() or result.stdout.strip()}"
        return result.stdout.strip() or "No diff."

    try:
        payload = _json_input(tool_input)
    except Exception as exc:
        return f"git.diff error: {exc}"

    if "path" in payload and ("proposed_content" in payload or payload.get("delete") is True):
        return _preview_diff(payload)

    args = ["diff"]
    if payload.get("cached"):
        args.append("--cached")
    if payload.get("name_only"):
        args.append("--name-only")
    path = str(payload.get("path", "")).strip()
    if path:
        args.extend(["--", path])
    result = _run_git(args)
    if result.returncode != 0:
        return f"git.diff error: {result.stderr.strip() or result.stdout.strip()}"
    return result.stdout.strip() or "No diff."


def log(tool_input: str) -> str:
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    try:
        payload = _json_input(tool_input)
    except Exception as exc:
        return f"git.log error: {exc}"

    n = int(payload.get("n", 10))
    n = min(max(n, 1), 100)
    result = _run_git(["log", f"-n{n}", "--oneline"])
    if result.returncode != 0:
        return f"git.log error: {result.stderr.strip() or result.stdout.strip()}"
    return result.stdout.strip() or "No commits found."


def checkout(tool_input: str) -> str:
    repo_error = _ensure_repo()
    if repo_error:
        return repo_error
    try:
        payload = _json_input(tool_input)
    except Exception as exc:
        return f"git.checkout error: {exc}"

    target = str(payload.get("target", "")).strip()
    if not target:
        return "git.checkout error: target is required."

    result = _run_git(["checkout", target])
    if result.returncode != 0:
        return f"git.checkout error: {result.stderr.strip() or result.stdout.strip()}"
    return result.stdout.strip() or f"Checked out {target}."


def _default_branch_name(goal: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")
    if not base:
        base = "task"
    return f"mantis/{base[:40]}"


def _sanitize_branch_name(name: str) -> str:
    clean = re.sub(r"\s+", "-", name.strip())
    clean = re.sub(r"[^A-Za-z0-9_./-]", "", clean)
    clean = clean.strip("./")
    return clean


def _preview_diff(payload: Dict[str, Any]) -> str:
    path = str(payload.get("path", "")).strip()
    if not path:
        return "git.diff preview error: path is required."

    target = Path(path)
    before = ""
    if target.exists() and target.is_file():
        try:
            before = target.read_text(encoding="utf-8")
        except Exception as exc:
            return f"git.diff preview error: {exc}"

    if payload.get("delete") is True:
        after = ""
        from_name = f"a/{path}"
        to_name = "/dev/null"
    else:
        after = str(payload.get("proposed_content", ""))
        from_name = f"a/{path}" if target.exists() else "/dev/null"
        to_name = f"b/{path}"

    diff_lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=from_name,
        tofile=to_name,
    )
    text = "".join(diff_lines).strip()
    return text or "No diff."
