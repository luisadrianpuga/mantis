import json

from workspace import WorkspaceIndex


def test_workspace_tree_and_search(tmp_path):
    project = tmp_path / "repo"
    project.mkdir()
    (project / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (project / "README.md").write_text("hello repo\n", encoding="utf-8")

    index = WorkspaceIndex(project)
    tree = json.loads(index.tree())

    assert tree["file_count"] == 2
    assert any(item["path"] == "main.py" and item["language"] == "python" for item in tree["files"])

    matches = json.loads(index.search("hello"))
    assert any(match["path"] == "README.md" for match in matches["matches"])
