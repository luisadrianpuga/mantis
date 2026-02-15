import asyncio
import logging
import os

from agent.loop import AgentLoop
from agent.memory import MemoryManager
from llm.ollama import OllamaClient
from storage.vectordb import VectorStore
from tools.registry import ToolRegistry
from worker.jobs import get_default_jobs
from worker.runner import LOG_FILE, run_job, _ensure_log_dir
from worker.scheduler import Scheduler

logger = logging.getLogger("mantis.worker")


def _configure_logging() -> None:
    _ensure_log_dir()
    log_format = logging.Formatter("%(asctime)s | %(message)s")

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_format)
        logger.addHandler(stream_handler)

    file_path = os.path.abspath(LOG_FILE)
    has_file_handler = any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == file_path for handler in logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False


async def main() -> None:
    _configure_logging()

    vector_store = VectorStore()
    memory_manager = MemoryManager(vector_store)
    tool_registry = ToolRegistry()
    ollama_client = OllamaClient()
    agent_loop = AgentLoop(ollama_client, tool_registry, memory_manager)

    scheduler = Scheduler(get_default_jobs())

    print("Mantis Worker started. Autonomous mode active.")
    logger.info("Worker booted with %d jobs", len(scheduler.jobs))

    while True:
        try:
            jobs = scheduler.due_jobs()
            for job in jobs:
                logger.info("Running job: %s", job.name)
                await run_job(agent_loop, job)
        except Exception as exc:  # pragma: no cover - long-running guard
            logger.exception("Worker loop encountered an error: %s", exc)

        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
