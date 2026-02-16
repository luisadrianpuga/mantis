import asyncio

from agent.planner import Planner


class FakeLLM:
    async def chat(self, messages, model=None):
        return "PLAN:\n1. Inspect repository\n2. Add endpoint\n3. Run tests"


def test_planner_extracts_tasks():
    planner = Planner(FakeLLM())
    plan = asyncio.run(planner.create_plan("Add a health endpoint", [{"name": "workspace.tree", "description": "..."}]))

    assert plan.tasks == ["Inspect repository", "Add endpoint", "Run tests"]
