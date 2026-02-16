import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import uuid


class PlanStore:
    def __init__(self, root: Path | str = Path(".mantis") / "plans") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_plan(self, goal: str, tasks: List[str]) -> Dict[str, Any]:
        plan_id = uuid.uuid4().hex
        payload = {
            "id": plan_id,
            "goal": goal,
            "tasks": tasks,
            "completed_tasks": [],
            "status": "running",
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        self._write(plan_id, payload)
        return payload

    def mark_task_completed(self, plan_id: str, task_index: int) -> Dict[str, Any]:
        payload = self.load(plan_id)
        completed = set(payload.get("completed_tasks", []))
        completed.add(task_index)
        payload["completed_tasks"] = sorted(completed)
        payload["updated_at"] = _now_iso()
        self._write(plan_id, payload)
        return payload

    def set_status(self, plan_id: str, status: str) -> Dict[str, Any]:
        payload = self.load(plan_id)
        payload["status"] = status
        payload["updated_at"] = _now_iso()
        self._write(plan_id, payload)
        return payload

    def load(self, plan_id: str) -> Dict[str, Any]:
        path = self.root / f"{plan_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Plan not found: {plan_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def latest_running(self) -> Dict[str, Any] | None:
        running = []
        for path in self.root.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("status") == "running":
                running.append(payload)
        if not running:
            return None
        running.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return running[0]

    def persist(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "id" not in payload:
            raise ValueError("Plan payload must include id")
        payload["updated_at"] = _now_iso()
        self._write(payload["id"], payload)
        return payload

    def _write(self, plan_id: str, payload: Dict[str, Any]) -> None:
        path = self.root / f"{plan_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
