import asyncio

from worker.runner import run_idle_improvement


class FakeTools:
    async def run(self, name, tool_input):
        assert name == "workspace.snapshot"
        return '{"tree":{"file_count":1}}'


class FakeLLM:
    async def chat(self, messages, model=None):
        return "1. Improve README examples\n2. Add tests"


class FakeAgent:
    def __init__(self):
        self.tools = FakeTools()
        self.llm = FakeLLM()


class FakeIdentity:
    def __init__(self):
        self.entries = []

    def append_journal(self, entry: str) -> None:
        self.entries.append(entry)


def test_run_idle_improvement_enqueues_one_task(tmp_path):
    from worker import runner

    queue_path = tmp_path / "queue.json"

    class LocalQueue(runner.TaskQueue):
        def __init__(self):
            super().__init__(queue_path)

    original = runner.TaskQueue
    runner.TaskQueue = LocalQueue
    try:
        identity = FakeIdentity()
        result = asyncio.run(run_idle_improvement(FakeAgent(), identity))

        assert result == "Idle improvement task enqueued."
        items = LocalQueue().list_items()
        assert len(items) == 1
        assert items[0]["goal"] == "Improve README examples"
        assert identity.entries and "Idle improvement task generated." in identity.entries[-1]
    finally:
        runner.TaskQueue = original
