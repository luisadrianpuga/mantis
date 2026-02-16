import json

from tools import filesystem


def test_write_blocked_when_flag_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(filesystem, "MANTIS_ALLOW_FILE_WRITE", False)

    payload = json.dumps({"path": "a.txt", "content": "hi", "overwrite": True})
    result = filesystem.write_file(payload)

    assert "disabled" in result


def test_create_patch_delete_with_change_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(filesystem, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(filesystem, "CHANGE_LOG_DIR", tmp_path / ".mantis" / "changes")
    monkeypatch.setattr(filesystem, "MANTIS_ALLOW_FILE_WRITE", True)

    create_payload = json.dumps({"path": "sample.txt", "content": "line1\nline2\n"})
    assert "Created file" in filesystem.create_file(create_payload)

    diff = "@@ -1,2 +1,2 @@\n line1\n-line2\n+line2-updated\n"
    patch_payload = json.dumps({"path": "sample.txt", "diff": diff})
    assert "Patched file" in filesystem.patch_file(patch_payload)

    delete_payload = json.dumps({"path": "sample.txt", "confirm": True})
    assert "Deleted file" in filesystem.delete_file(delete_payload)

    change_logs = list((tmp_path / ".mantis" / "changes").glob("*.json"))
    assert len(change_logs) == 3
