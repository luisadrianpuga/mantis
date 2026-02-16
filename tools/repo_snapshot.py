import json
from typing import Any, Dict, List

from tools import git_tools, tests, workspace_tools


def snapshot(_: str = "") -> str:
    tree = workspace_tools.tree("")
    commits = git_tools.log(json.dumps({"n": 5}))
    tests_output = tests.run_tests("")

    todo_matches: List[Dict[str, Any]] = []
    seen_paths = set()
    for token in ["TODO", "FIXME", "BUG"]:
        result = workspace_tools.search(token)
        try:
            payload = json.loads(result)
            for match in payload.get("matches", []):
                path = match.get("path")
                key = (path, match.get("type"), match.get("snippet", ""))
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                todo_matches.append(match)
        except Exception:
            continue

    readme = workspace_tools.read_file("README.md")
    if readme.startswith("Workspace read error"):
        readme = ""

    payload = {
        "tree": _safe_json(tree),
        "recent_commits": commits,
        "failing_tests": tests_output,
        "todo_comments": {"matches": todo_matches},
        "readme": readme[:2000],
    }
    return json.dumps(payload, indent=2)


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw
