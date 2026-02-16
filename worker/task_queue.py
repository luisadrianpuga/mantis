import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import uuid


class TaskQueue:
    def __init__(self, path: Path | str = Path(".mantis") / "queue.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def enqueue(self, goal: str) -> Dict[str, Any]:
        items = self._read()
        payload = {
            "id": uuid.uuid4().hex,
            "goal": goal,
            "status": "pending",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        items.append(payload)
        self._write(items)
        return payload

    def next_pending(self) -> Dict[str, Any] | None:
        items = self._read()
        for item in items:
            if item.get("status") == "pending":
                item["status"] = "running"
                item["updated_at"] = _now_iso()
                self._write(items)
                return item
        return None

    def set_status(self, task_id: str, status: str) -> Dict[str, Any] | None:
        items = self._read()
        for item in items:
            if item.get("id") == task_id:
                item["status"] = status
                item["updated_at"] = _now_iso()
                self._write(items)
                return item
        return None

    def list_items(self) -> List[Dict[str, Any]]:
        return self._read()

    def has_goal(self, goal: str) -> bool:
        normalized = goal.strip().lower()
        if not normalized:
            return False
        return any(item.get("goal", "").strip().lower() == normalized for item in self._read())

    def _read(self) -> List[Dict[str, Any]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
        except Exception:
            pass
        return []

    def _write(self, items: List[Dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
