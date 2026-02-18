import asyncio

from worker.runner import run_repo_discovery


class FakeTools:
    async def run(self, name, tool_input):
        assert name == "workspace.snapshot"
        return '{"tree":{"file_count":1}}'


class FakeLLM:
    async def chat(self, messages, model=None):
        return "1. Add tests for queue\n2. Improve README setup section\n3. add tests for queue"


class FakeAgent:
    def __init__(self):
        self.tools = FakeTools()
        self.llm = FakeLLM()


class FakeIdentity:
    def __init__(self):
        self.entries = []

    def append_journal(self, entry: str) -> None:
        self.entries.append(entry)


def test_run_repo_discovery_enqueues_goals(tmp_path):
    from worker import runner

    queue_path = tmp_path / "queue.json"

    class LocalQueue(runner.TaskQueue):
        def __init__(self):
            super().__init__(queue_path)

    original = runner.TaskQueue
    runner.TaskQueue = LocalQueue
    try:
        identity = FakeIdentity()
        result = asyncio.run(run_repo_discovery(FakeAgent(), identity))

        assert "Enqueued 2 tasks" in result
        assert identity.entries and "Discovered new development tasks." in identity.entries[-1]

        items = LocalQueue().list_items()
        assert len(items) == 2
        assert items[0]["goal"] == "Add tests for queue"
    finally:
        runner.TaskQueue = original
