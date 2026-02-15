import json
import re
from typing import Any

from planner.models import Task, TaskStatus
from planner.store import add_task, get_pending_tasks, load_goals, update_task


async def generate_tasks_from_goals(agent) -> int:
    """
    Generate missing tasks for goals that currently have no tasks.
    Returns number of created tasks.
    """
    goals = load_goals()
    created = 0

    for goal in goals:
        if goal.tasks:
            continue

        prompt = (
            "Break this goal into small executable tasks. "
            "Return strict JSON as a list of objects with keys: title, description, priority (1-10).\n"
            f"Goal: {goal.title}"
        )
        reply = await agent.run([{"role": "user", "content": prompt}])
        task_specs = _parse_task_specs(reply)

        for spec in task_specs:
            priority = _coerce_priority(spec.get("priority", 5))
            add_task(
                goal_id=goal.id,
                title=spec.get("title", "Untitled task"),
                description=spec.get("description", spec.get("title", "")),
                priority=priority,
            )
            created += 1

    return created


async def pick_next_task() -> Task | None:
    pending = get_pending_tasks()
    return pending[0] if pending else None


async def complete_task(task_id: str, result: str) -> Task | None:
    return update_task(task_id=task_id, status=TaskStatus.done, result=result)


async def fail_task(task_id: str, result: str) -> Task | None:
    return update_task(task_id=task_id, status=TaskStatus.failed, result=result)


async def start_task(task_id: str) -> Task | None:
    return update_task(task_id=task_id, status=TaskStatus.running)


def _parse_task_specs(reply: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(reply)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", reply)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            pass

    lines = [line.strip("- ").strip() for line in reply.splitlines() if line.strip()]
    return [{"title": line, "description": line, "priority": 5} for line in lines[:10]]


def _coerce_priority(value: Any) -> int:
    try:
        return max(1, min(10, int(value)))
    except (TypeError, ValueError):
        return 5
