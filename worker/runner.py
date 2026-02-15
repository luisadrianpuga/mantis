import logging
import os
from datetime import datetime

from agent.loop import AgentLoop
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


async def run_job(agent: AgentLoop, job: Job) -> str:
    messages = [
        {"role": "system", "content": "You are running an autonomous scheduled task."},
        {"role": "user", "content": job.description},
    ]
    result = await agent.run(messages)

    # Persist the result to long-term memory with job metadata.
    agent.memory.store_memory(result, {"source": "worker", "job": job.name, "ran_at": datetime.utcnow().isoformat()})

    # Log to disk in the required format.
    _ensure_log_dir()
    if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename == os.path.abspath(LOG_FILE) for handler in logger.handlers):
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(file_handler)
    logger.info("%s | %s", job.name, _summarize_result(result))

    return result
