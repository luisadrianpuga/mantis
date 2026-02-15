import asyncio
import logging
import os
from datetime import datetime

from agent.loop import AgentLoop
from agent.memory import MemoryManager
from planner.planner import complete_task, fail_task, generate_tasks_from_goals, pick_next_task, start_task
from planner.store import load_goals
from providers.router import ProviderRouter
from storage.vectordb import VectorStore
from tools.registry import ToolRegistry
from workspace.scanner import index_files_into_memory
from worker.jobs import get_default_jobs
from worker.runner import LOG_FILE, _ensure_log_dir
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
    provider_router = ProviderRouter()
    agent_loop = AgentLoop(provider_router, tool_registry, memory_manager)

    scheduler = Scheduler(get_default_jobs())

    print("Mantis Worker started. Autonomous mode active.")
    logger.info("Worker booted with %d jobs", len(scheduler.jobs))

    while True:
        try:
            # Always run a quick autonomy cycle.
            await _plan_and_execute_once(agent_loop)

            jobs = scheduler.due_jobs()
            for job in jobs:
                logger.info("Running job: %s", job.name)
                await _run_scheduled_job(agent_loop, memory_manager, job.name)
        except Exception as exc:  # pragma: no cover - long-running guard
            logger.exception("Worker loop encountered an error: %s", exc)

        await asyncio.sleep(5)


async def _plan_and_execute_once(agent: AgentLoop) -> None:
    await generate_tasks_from_goals(agent)

    task = await pick_next_task()
    if not task:
        return

    await start_task(task.id)
    prompt = (
        f"Execute this task and produce a concise result.\n"
        f"Task title: {task.title}\n"
        f"Task description: {task.description}"
    )
    result = await agent.run(prompt)

    if result.startswith("[LLM error"):
        await fail_task(task.id, result)
        return

    await complete_task(task.id, result)
    agent.memory.store_memory(
        f"Completed task: {task.title}\nResult: {result}",
        {
            "source": "planner",
            "type": "task_result",
            "task_id": task.id,
            "recorded_at": datetime.utcnow().isoformat(),
        },
    )


async def _run_scheduled_job(agent: AgentLoop, memory: MemoryManager, job_name: str) -> None:
    if job_name == "task_planning":
        created = await generate_tasks_from_goals(agent)
        logger.info("Planner created %d tasks", created)
        return

    if job_name == "task_execution":
        await _plan_and_execute_once(agent)
        return

    if job_name == "workspace_index":
        summary = index_files_into_memory(memory)
        logger.info("Workspace indexed: %s", " ".join(summary.split())[:300])
        return

    if job_name == "journal":
        journal = await agent.run(
            "Write a one-paragraph operator journal entry summarizing recent progress and blockers."
        )
        memory.store_memory(journal, {"source": "worker", "type": "journal", "recorded_at": datetime.utcnow().isoformat()})
        logger.info("Journal entry recorded")
        return

    if job_name == "review_goals":
        goals = load_goals()
        lines = []
        for goal in goals:
            done = len([task for task in goal.tasks if task.status.value == "done"])
            total = len(goal.tasks)
            lines.append(f"{goal.title}: {done}/{total} done")
        summary = "\n".join(lines) if lines else "No goals found."
        memory.store_memory(summary, {"source": "planner", "type": "goal_review", "recorded_at": datetime.utcnow().isoformat()})
        logger.info("Goal review completed")
        return


if __name__ == "__main__":
    asyncio.run(main())
