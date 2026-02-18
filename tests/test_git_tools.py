import json
import subprocess

from tools import git_tools


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def test_git_tool_flow(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "mantis@example.com"], repo)
    _run(["git", "config", "user.name", "Mantis"], repo)
    (repo / "a.txt").write_text("v1\n", encoding="utf-8")
    _run(["git", "add", "a.txt"], repo)
    _run(["git", "commit", "-m", "init"], repo)

    monkeypatch.chdir(repo)

    branch_result = git_tools.create_branch(json.dumps({"name": "mantis/test-flow"}))
    assert "Checked out branch" in branch_result

    (repo / "a.txt").write_text("v2\n", encoding="utf-8")
    status = git_tools.status("")
    assert "a.txt" in status

    diff = git_tools.diff("")
    assert "diff --git" in diff

    commit = git_tools.commit(json.dumps({"message": "Mantis: update file\nTests: PASS"}))
    assert "Mantis: update file" in commit

    logs = git_tools.log(json.dumps({"n": 1}))
    assert "Mantis: update file" in logs
