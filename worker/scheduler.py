import time
from dataclasses import dataclass
from typing import List


@dataclass
class Job:
    name: str
    description: str
    interval_seconds: int
    last_run: float = 0


class Scheduler:
    def __init__(self, jobs: List[Job]):
        self.jobs = jobs

    def due_jobs(self) -> List[Job]:
        now = time.time()
        ready: List[Job] = []
        for job in self.jobs:
            if now - job.last_run >= job.interval_seconds:
                job.last_run = now
                ready.append(job)
        return ready
