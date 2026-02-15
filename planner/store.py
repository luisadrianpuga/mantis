import json
import os
from dataclasses import asdict
from typing import Any, List
from uuid import uuid4

from planner.models import Goal, Task, TaskStatus

TASKS_FILE = os.path.join(".mantis", "tasks.json")


def _ensure_store_dir() -> None:
    os.makedirs(os.path.dirname(TASKS_FILE), exist_ok=True)


def _goal_from_dict(data: dict[str, Any]) -> Goal:
    tasks = [Task(**{**task, "status": TaskStatus(task.get("status", TaskStatus.pending))}) for task in data.get("tasks", [])]
    return Goal(id=data["id"], title=data["title"], tasks=tasks)


def _task_to_dict(task: Task) -> dict[str, Any]:
    payload = asdict(task)
    payload["status"] = task.status.value
    return payload


def _goal_to_dict(goal: Goal) -> dict[str, Any]:
    return {
        "id": goal.id,
        "title": goal.title,
        "tasks": [_task_to_dict(task) for task in goal.tasks],
    }


def load_goals() -> List[Goal]:
    if not os.path.exists(TASKS_FILE):
        return []

    with open(TASKS_FILE, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw_goals = data.get("goals", []) if isinstance(data, dict) else []
    return [_goal_from_dict(goal) for goal in raw_goals]


def save_goals(goals: List[Goal]) -> None:
    _ensure_store_dir()
    payload = {"goals": [_goal_to_dict(goal) for goal in goals]}
    with open(TASKS_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def add_goal(title: str, goal_id: str | None = None) -> Goal:
    goals = load_goals()
    goal = Goal(id=goal_id or str(uuid4()), title=title)
    goals.append(goal)
    save_goals(goals)
    return goal


def add_task(
    goal_id: str,
    title: str,
    description: str,
    priority: int = 5,
    task_id: str | None = None,
) -> Task:
    goals = load_goals()
    for goal in goals:
        if goal.id == goal_id:
            task = Task(id=task_id or str(uuid4()), title=title, description=description, priority=priority)
            goal.tasks.append(task)
            save_goals(goals)
            return task
    raise ValueError(f"Goal not found: {goal_id}")


def update_task(task_id: str, status: TaskStatus | None = None, result: str | None = None) -> Task | None:
    goals = load_goals()
    for goal in goals:
        for task in goal.tasks:
            if task.id == task_id:
                if status is not None:
                    task.status = status
                if result is not None:
                    task.result = result
                save_goals(goals)
                return task
    return None


def get_pending_tasks() -> List[Task]:
    goals = load_goals()
    tasks: List[Task] = []
    for goal in goals:
        tasks.extend(task for task in goal.tasks if task.status == TaskStatus.pending)
    return sorted(tasks, key=lambda task: task.priority)
