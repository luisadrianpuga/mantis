import asyncio

from agent.repo_analyzer import RepoAnalyzer


class FakeLLM:
    async def chat(self, messages, model=None):
        return "1. Fix failing tests in API routes\n2. Add unit tests for task queue\n3. Resolve TODO in worker service"


def test_repo_analyzer_extracts_goals():
    analyzer = RepoAnalyzer(FakeLLM())
    goals = asyncio.run(analyzer.discover_goals('{"tree":{}}'))

    assert goals == [
        "Fix failing tests in API routes",
        "Add unit tests for task queue",
        "Resolve TODO in worker service",
    ]
