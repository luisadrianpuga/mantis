import re
from typing import List


class RepoAnalyzer:
    def __init__(self, llm) -> None:
        self.llm = llm

    async def discover_goals(self, snapshot_json: str) -> List[str]:
        prompt = """
You are a senior software engineer performing repo triage.

Given repository snapshot data, produce a list of
HIGH VALUE development goals.

Look for:
- failing tests
- missing tests
- TODO / FIXME comments
- missing docs
- obvious improvements
- bugs or tech debt

Return ONLY a numbered list.
Each goal must be actionable and specific.
""".strip()

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": snapshot_json},
        ]

        raw = await self.llm.chat(messages)
        return self._extract_goals(raw)

    @staticmethod
    def _extract_goals(raw: str) -> List[str]:
        goals: List[str] = []
        for line in raw.splitlines():
            match = re.match(r"^\s*\d+\.\s+(.*)$", line.strip())
            if match:
                goal = match.group(1).strip()
                if goal:
                    goals.append(goal)
        return goals
