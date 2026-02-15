from worker.scheduler import Job


def get_default_jobs() -> list[Job]:
    return [
        Job(
            name="task_planning",
            description="Break goals into actionable tasks.",
            interval_seconds=30 * 60,
        ),
        Job(
            name="task_execution",
            description="Execute the next pending task.",
            interval_seconds=60,
        ),
        Job(
            name="workspace_index",
            description="Refresh repository understanding in memory.",
            interval_seconds=6 * 60 * 60,
        ),
        Job(
            name="journal",
            description="Write a concise operational journal entry.",
            interval_seconds=15 * 60,
        ),
        Job(
            name="review_goals",
            description="Review goal and task progress, then summarize.",
            interval_seconds=60 * 60,
        ),
    ]
