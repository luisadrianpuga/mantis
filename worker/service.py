import asyncio
import logging
import os
import sys

from agent.loop import AgentLoop
from agent.memory import MemoryManager
from identity.bootstrap import bootstrap_identity
from identity.manager import IdentityManager
from providers.detection import has_anthropic, has_ollama, has_openai, has_openai_compat
from providers.router import ProviderRouter
from storage.vectordb import VectorStore
from tools.registry import ToolRegistry
from worker.jobs import get_default_jobs
from worker.runner import LOG_FILE, run_job, _ensure_log_dir
from worker.scheduler import Scheduler

logger = logging.getLogger("mantis.worker")
PID_FILE = os.path.join(".mantis", "worker.pid")


def _configure_logging() -> None:
    _ensure_log_dir()
    log_format = logging.Formatter("%(asctime)s | %(message)s")

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_format)
        logger.addHandler(stream_handler)

    file_path = os.path.abspath(LOG_FILE)
    has_file_handler = any(
        isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == file_path
        for handler in logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_pid_lock() -> None:
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)

    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r", encoding="utf-8") as handle:
                old_pid = int(handle.read().strip())
        except (ValueError, OSError):
            old_pid = None

        if old_pid and _is_pid_alive(old_pid):
            print("Worker already running.")
            sys.exit(1)

    with open(PID_FILE, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def _release_pid_lock() -> None:
    if not os.path.exists(PID_FILE):
        return

    try:
        with open(PID_FILE, "r", encoding="utf-8") as handle:
            pid_text = handle.read().strip()
        if pid_text == str(os.getpid()):
            os.remove(PID_FILE)
    except OSError:
        pass


def _print_first_run_banner(workspace: str) -> None:
    print("First run detected.")
    print(f"Workspace: {workspace}")
    print("Memory initialized.")


def _print_worker_health_banner(default_model: str | None) -> None:
    safe_mode = _env_bool("MANTIS_SAFE_MODE", default=True)
    dangerous_enabled = _env_bool("MANTIS_APPROVE_DANGEROUS", default=False) and not safe_mode

    print("🪲 Mantis Worker running (autonomous mode)")
    print(f"Safety mode: {'ON' if safe_mode else 'OFF'}")
    print(f"Dangerous tools: {'ENABLED' if dangerous_enabled else 'DISABLED'}")
    print(f"Default model: {default_model or 'None detected'}")

    print("\nProviders detected:")
    print(f"Local OpenAI-compatible: {'YES' if has_openai_compat() else 'NO'}")
    print(f"OpenAI cloud: {'YES' if has_openai() else 'NO'}")
    print(f"Anthropic: {'YES' if has_anthropic() else 'NO'}")
    print(f"Ollama: {'YES' if has_ollama() else 'NO'}")

    if has_openai_compat():
        print("\nActive provider: LOCAL LLM")
    elif has_openai():
        print("\nActive provider: OpenAI")
    elif has_anthropic():
        print("\nActive provider: Anthropic")
    elif has_ollama():
        print("\nActive provider: Ollama")
    else:
        print("\nActive provider: NONE")

    print("Memory location: .mantis/")


async def main() -> None:
    first_run = not os.path.exists(".mantis")

    _acquire_pid_lock()
    _configure_logging()

    try:
        identity: IdentityManager = bootstrap_identity(workspace=os.getcwd())
        vector_store = VectorStore()
        memory_manager = MemoryManager(vector_store)
        tool_registry = ToolRegistry()
        provider_router = ProviderRouter()
        agent_loop = AgentLoop(provider_router, tool_registry, memory_manager, identity=identity)

        scheduler = Scheduler(get_default_jobs())

        if first_run:
            _print_first_run_banner(os.getcwd())
        _print_worker_health_banner(provider_router.default_model)
        logger.info("Worker booted with %d jobs", len(scheduler.jobs))

        while True:
            try:
                jobs = scheduler.due_jobs()
                for job in jobs:
                    logger.info("Running job: %s", job.name)
                    await run_job(agent_loop, job, identity=identity)
            except Exception as exc:  # pragma: no cover - long-running guard
                logger.exception("Worker loop encountered an error: %s", exc)

            await asyncio.sleep(5)
    finally:
        _release_pid_lock()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Worker shutting down gracefully.")
