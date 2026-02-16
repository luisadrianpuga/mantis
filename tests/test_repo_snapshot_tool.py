import json

from tools import repo_snapshot


def test_snapshot_contains_expected_fields(monkeypatch):
    monkeypatch.setattr(repo_snapshot.workspace_tools, "tree", lambda _: json.dumps({"file_count": 2}))
    monkeypatch.setattr(repo_snapshot.git_tools, "log", lambda _: "abc123 init")
    monkeypatch.setattr(repo_snapshot.tests, "run_tests", lambda _: "EXIT_CODE: 1")

    def fake_search(query: str):
        return json.dumps({"matches": [{"path": f"{query}.md", "type": "content", "snippet": query}]})

    monkeypatch.setattr(repo_snapshot.workspace_tools, "search", fake_search)
    monkeypatch.setattr(repo_snapshot.workspace_tools, "read_file", lambda _: "# README\nhello")

    raw = repo_snapshot.snapshot("")
    payload = json.loads(raw)

    assert payload["tree"]["file_count"] == 2
    assert payload["recent_commits"] == "abc123 init"
    assert payload["failing_tests"] == "EXIT_CODE: 1"
    assert payload["readme"].startswith("# README")
    assert len(payload["todo_comments"]["matches"]) == 3
