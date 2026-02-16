import asyncio

from agent.loop import AgentLoop


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    async def chat(self, messages, model=None):
        if not self.replies:
            return "No more replies"
        return self.replies.pop(0)


class FakeTools:
    def __init__(self):
        self.calls = []

    def list_tools(self):
        return [
            {"name": "write_file", "description": "..."},
            {"name": "run_tests", "description": "..."},
        ]

    async def run(self, name, tool_input):
        self.calls.append((name, tool_input))
        if name == "write_file":
            return "Wrote file: app.py"
        if name == "run_tests":
            return "COMMAND: pytest -q\nEXIT_CODE: 0\nOUTPUT:\npassed"
        return "ok"


class FakeMemory:
    def retrieve_memory(self, query):
        return []

    def store_memory(self, text, metadata):
        return None


class FakeIdentity:
    def load_identity_block(self):
        return "identity"


def test_agent_loop_project_goal_runs_tests_after_edits():
    llm = FakeLLM(
        [
            "PLAN:\n1. Modify source",  # planner output
            'TOOL: write_file | {"path":"app.py","content":"print(1)","overwrite":true}',
            "Task 1 complete.",
            "All done. Tests are green.",
        ]
    )
    tools = FakeTools()
    memory = FakeMemory()
    loop = AgentLoop(llm, tools, memory, identity=FakeIdentity(), max_iterations=3)

    result = asyncio.run(loop.run("Add a feature and update tests in this repository"))

    assert "green" in result.lower()
    tool_names = [name for name, _ in tools.calls]
    assert "write_file" in tool_names
    assert tool_names.count("run_tests") >= 2
