import re
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from agent.planner import Planner
from agent.prompt import build_system_prompt
from config import MANTIS_DEV_MODE
from identity.manager import IdentityManager
from tools.registry import ToolRegistry

if TYPE_CHECKING:
    from agent.memory import MemoryManager


WRITE_TOOLS = {"create_file", "write_file", "patch_file", "delete_file"}


class AgentLoop:
    """
    Agent loop with planning support for project-level goals.

    Flow for project goals:
    user goal -> planner -> sequential task execution -> tests -> fix loop
    """

    def __init__(
        self,
        llm_client,
        tools: ToolRegistry,
        memory: "MemoryManager",
        identity: IdentityManager | None = None,
        max_iterations: int = 5,
    ) -> None:
        self.llm = llm_client
        self.tools = tools
        self.memory = memory
        self.identity = identity
        self.max_iterations = max_iterations
        self.planner = Planner(llm_client)

    async def run(self, messages: List[Dict[str, str]] | str, model: str | None = None) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        latest_user_message = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        memories = self.memory.retrieve_memory(latest_user_message) if latest_user_message else []
        identity_block = self.identity.load_identity_block() if self.identity else ""
        system_prompt = build_system_prompt(self.tools.list_tools(), memories, identity_block)
        conversation: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}] + messages

        final_reply = ""
        if self._is_project_level_goal(latest_user_message):
            final_reply, conversation = await self._run_project_goal(conversation, latest_user_message, model=model)
        else:
            final_reply, _, conversation = await self._run_chat_loop(
                conversation,
                model=model,
                iteration_limit=self.max_iterations,
            )

        if latest_user_message and len(latest_user_message) > 40 and final_reply:
            summary = f"User: {latest_user_message}\nAssistant: {final_reply}"
            self.memory.store_memory(summary, {"source": "chat", "type": "exchange"})

        return final_reply or "Reached iteration limit without final answer."

    async def _run_project_goal(
        self,
        conversation: List[Dict[str, str]],
        user_goal: str,
        model: str | None = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        plan = await self.planner.create_plan(user_goal, self.tools.list_tools(), model=model)
        conversation.append({"role": "assistant", "content": plan.raw})

        last_reply = plan.raw
        tests_ran = False

        for index, task in enumerate(plan.tasks, start=1):
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"Execute plan task {index}/{len(plan.tasks)}: {task}\n"
                        "Use tools as needed. Keep changes minimal and safe."
                    ),
                }
            )
            last_reply, used_tools, conversation = await self._run_chat_loop(
                conversation,
                model=model,
                iteration_limit=self.max_iterations,
            )

            changed_files = bool(used_tools.intersection(WRITE_TOOLS))
            if changed_files:
                tests_ran = True
                test_result = await self.tools.run("run_tests", "")
                conversation.append({"role": "user", "content": f"Automatic test result:\n{test_result}"})

                if self._extract_exit_code(test_result) != 0:
                    fixed = await self._run_fix_loop(conversation, model=model)
                    if not fixed:
                        return (
                            "Stopped after reaching iteration limit while fixing failing tests.\n"
                            f"Last test output:\n{test_result}",
                            conversation,
                        )

        if tests_ran:
            final_test_result = await self.tools.run("run_tests", "")
            conversation.append({"role": "user", "content": f"Final test verification:\n{final_test_result}"})
            if self._extract_exit_code(final_test_result) != 0:
                fixed = await self._run_fix_loop(conversation, model=model)
                if not fixed:
                    return (
                        "Plan execution finished but tests remain failing after max fix iterations.\n"
                        f"Last test output:\n{final_test_result}",
                        conversation,
                    )

        conversation.append(
            {
                "role": "user",
                "content": "Provide the final user-facing summary with what changed and current test status.",
            }
        )
        last_reply, _, conversation = await self._run_chat_loop(
            conversation,
            model=model,
            iteration_limit=self.max_iterations,
        )
        return last_reply, conversation

    async def _run_fix_loop(self, conversation: List[Dict[str, str]], model: str | None = None) -> bool:
        for _ in range(self.max_iterations):
            conversation.append(
                {
                    "role": "user",
                    "content": "Tests are failing. Fix the failure with minimal diffs, then stop for test execution.",
                }
            )
            _, used_tools, conversation = await self._run_chat_loop(
                conversation,
                model=model,
                iteration_limit=self.max_iterations,
            )

            if not used_tools.intersection(WRITE_TOOLS):
                return False

            test_result = await self.tools.run("run_tests", "")
            conversation.append({"role": "user", "content": f"Test rerun result:\n{test_result}"})
            if self._extract_exit_code(test_result) == 0:
                return True

        return False

    async def _run_chat_loop(
        self,
        conversation: List[Dict[str, str]],
        model: str | None = None,
        iteration_limit: int = 5,
    ) -> Tuple[str, Set[str], List[Dict[str, str]]]:
        used_tools: Set[str] = set()
        last_reply = ""

        for _ in range(iteration_limit):
            reply = await self.llm.chat(conversation, model=model)
            if reply.startswith("[LLM error"):
                return reply, used_tools, conversation

            tool_call = self._parse_tool_call(reply)
            if tool_call:
                tool_name, tool_input = tool_call
                used_tools.add(tool_name)
                conversation.append({"role": "assistant", "content": reply})
                tool_result = await self.tools.run(tool_name, tool_input)
                conversation.append({"role": "user", "content": f"Tool {tool_name} result:\n{tool_result}"})
                continue

            conversation.append({"role": "assistant", "content": reply})
            last_reply = reply
            break

        return last_reply, used_tools, conversation

    @staticmethod
    def _parse_tool_call(text: str) -> Optional[Tuple[str, str]]:
        """
        Detect TOOL: tool_name | input pattern.
        """
        for line in text.splitlines():
            match = re.match(r"^TOOL:\s*([\w\.-]+)\s*\|\s*(.*)$", line.strip())
            if match:
                return match.group(1), match.group(2)
        return None

    @staticmethod
    def _extract_exit_code(output: str) -> int:
        match = re.search(r"EXIT_CODE:\s*(-?\d+)", output)
        if not match:
            return 1
        return int(match.group(1))

    @staticmethod
    def _is_project_level_goal(user_message: str) -> bool:
        if not MANTIS_DEV_MODE:
            return False
        lowered = (user_message or "").lower()
        if not lowered.strip():
            return False

        project_keywords = {
            "implement",
            "create",
            "build",
            "refactor",
            "add",
            "feature",
            "endpoint",
            "tests",
            "codebase",
            "repository",
        }
        return len(lowered) >= 60 or any(keyword in lowered for keyword in project_keywords)
