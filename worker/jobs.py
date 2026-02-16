from worker.scheduler import Job


def get_default_jobs() -> list[Job]:
    return [
        Job(
            name="repo_task_discovery",
            description="Analyze repository and enqueue development tasks.",
            interval_seconds=60 * 10,
        ),
        Job(
            name="self_reflection",
            description=(
                "Reflect on recent activity and produce a short journal entry describing:\n"
                "- what happened\n"
                "- what was learned\n"
                "- what should improve next"
            ),
            interval_seconds=30 * 60,
        ),
        Job(
            name="web_exploration",
            description="Search the web for interesting AI or software engineering news.\nSummarize findings.",
            interval_seconds=60 * 60,
        ),
        Job(
            name="workspace_cleanup",
            description="Review stored memories and create a concise summary of long-term context.",
            interval_seconds=6 * 60 * 60,
        ),
    ]
