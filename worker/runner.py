import logging
import os
from datetime import datetime

from agent.repo_analyzer import RepoAnalyzer
from agent.loop import AgentLoop
from identity.manager import IdentityManager
from worker.task_queue import TaskQueue
from worker.scheduler import Job

logger = logging.getLogger("mantis.worker")
LOG_DIR = os.path.join(".mantis", "logs")
LOG_FILE = os.path.join(LOG_DIR, "worker.log")


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _summarize_result(result: str, max_length: int = 500) -> str:
    """
    Collapse whitespace and trim very long outputs to keep log lines readable.
    """
    single_line = " ".join(result.split())
    if len(single_line) <= max_length:
        return single_line
    return single_line[: max_length - 3] + "..."


async def run_job(agent: AgentLoop, job: Job, identity: IdentityManager | None = None) -> str:
    if job.name == "repo_task_discovery":
        result = await run_repo_discovery(agent, identity)
        _log_job_result(job.name, result)
        agent.memory.store_memory(result, {"source": "worker", "job": job.name, "ran_at": datetime.utcnow().isoformat()})
        return result

    messages = [
        {"role": "system", "content": "You are running an autonomous scheduled task."},
        {"role": "user", "content": job.description},
    ]
    result = await agent.run(messages)

    # Persist the result to long-term memory with job metadata.
    agent.memory.store_memory(result, {"source": "worker", "job": job.name, "ran_at": datetime.utcnow().isoformat()})

    _log_job_result(job.name, result)

    if identity:
        if job.name == "self_reflection":
            identity.append_journal(result)
        else:
            summary = _summarize_result(result, max_length=240)
            identity.append_journal(f"Completed job '{job.name}'. Summary: {summary}")

    return result


async def run_repo_discovery(agent: AgentLoop, identity: IdentityManager | None = None) -> str:
    snapshot = await agent.tools.run("workspace.snapshot", "")
    analyzer = RepoAnalyzer(agent.llm)
    goals = await analyzer.discover_goals(snapshot)

    queue = TaskQueue()
    enqueued = 0
    for goal in goals:
        goal = goal.strip()
        if not goal:
            continue
        if queue.has_goal(goal):
            continue
        queue.enqueue(goal)
        enqueued += 1

    if identity:
        identity.append_journal("Discovered new development tasks.")

    return f"Task discovery completed. Enqueued {enqueued} tasks."


def _log_job_result(job_name: str, result: str) -> None:
    # Log to disk in the required format.
    _ensure_log_dir()
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(LOG_FILE) for handler in logger.handlers):
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(file_handler)
    logger.info("%s | %s", job_name, _summarize_result(result))
