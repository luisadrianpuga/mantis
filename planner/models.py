from dataclasses import dataclass, field
from enum import Enum
from typing import List


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


@dataclass
class Task:
    id: str
    title: str
    description: str
    priority: int = 5
    status: TaskStatus = TaskStatus.pending
    result: str | None = None


@dataclass
class Goal:
    id: str
    title: str
    tasks: List[Task] = field(default_factory=list)
