from worker.scheduler import Job


def get_default_jobs() -> list[Job]:
    return [
        Job(
            name="self_reflection",
            description="Reflect on recent memories and summarize important things.",
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
