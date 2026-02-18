import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Plan:
    raw: str
    tasks: List[str]


class Planner:
    def __init__(self, llm_client) -> None:
        self.llm = llm_client

    async def create_plan(self, user_goal: str, tool_descriptions: List[Dict[str, str]], model: str | None = None) -> Plan:
        tools_text = "\n".join(f"- {tool['name']}: {tool['description']}" for tool in tool_descriptions)
        planner_prompt = (
            "You are a senior software engineer creating an execution plan. "
            "Produce only this format:\n"
            "PLAN:\n"
            "1. ...\n"
            "2. ...\n"
            "3. ...\n"
            "Keep tasks concrete, sequential, and minimal-diff focused.\n"
            f"Available tools:\n{tools_text or '- none'}"
        )
        messages = [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": user_goal},
        ]
        raw = await self.llm.chat(messages, model=model)
        tasks = self._extract_tasks(raw)
        if not tasks:
            tasks = [f"Implement the goal safely: {user_goal}"]
            raw = "PLAN:\n1. " + tasks[0]
        return Plan(raw=raw, tasks=tasks)

    @staticmethod
    def _extract_tasks(raw: str) -> List[str]:
        tasks: List[str] = []
        for line in raw.splitlines():
            match = re.match(r"^\s*\d+\.\s+(.*)$", line.strip())
            if match:
                item = match.group(1).strip()
                if item:
                    tasks.append(item)
        return tasks
