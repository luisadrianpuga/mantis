import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from agent.plan_store import PlanStore
from agent.planner import Plan, Planner
from agent.prompt import build_system_prompt
from config import MANTIS_DEV_MODE, MANTIS_RETRY_BASE_DELAY_SEC, MANTIS_RETRY_MAX_RETRIES
from identity.manager import IdentityManager
from tools.registry import ToolRegistry

if TYPE_CHECKING:
    from agent.memory import MemoryManager


WRITE_TOOLS = {"create_file", "write_file", "patch_file", "delete_file"}
RETRY_TOOLS = WRITE_TOOLS | {"run_tests"}


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
        self.plan_store = PlanStore()
        self.metrics_path = Path(".mantis") / "metrics.jsonl"

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
            final_reply, _, _, conversation = await self._run_chat_loop(
                conversation,
                model=model,
                iteration_limit=self.max_iterations,
            )

        if latest_user_message and len(latest_user_message) > 40 and final_reply:
            summary = f"User: {latest_user_message}\nAssistant: {final_reply}"
            self.memory.store_memory(summary, {"source": "chat", "type": "exchange"})

        return final_reply or "Reached iteration limit without final answer."

    async def resume_running_plan(self, model: str | None = None) -> str | None:
        running_plan = self.plan_store.latest_running()
        if not running_plan:
            return None

        goal = running_plan.get("goal", "")
        if not goal:
            return None

        memories = self.memory.retrieve_memory(goal) if goal else []
        identity_block = self.identity.load_identity_block() if self.identity else ""
        system_prompt = build_system_prompt(self.tools.list_tools(), memories, identity_block)
        conversation = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Resume unfinished plan for goal: {goal}",
            },
        ]

        reply, _ = await self._run_project_goal(conversation, goal, model=model, existing_plan=running_plan)
        return reply

    async def _run_project_goal(
        self,
        conversation: List[Dict[str, str]],
        user_goal: str,
        model: str | None = None,
        existing_plan: Dict[str, object] | None = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        started_at = time.monotonic()
        tests_passed = True
        changed_paths: Set[str] = set()
        snapshot = await self._run_tool_with_retry("workspace.snapshot", "")
        conversation.append({"role": "user", "content": f"Repository snapshot:\n{snapshot}"})

        if existing_plan:
            plan_payload = existing_plan
            plan = Plan(raw=self._format_plan(plan_payload.get("tasks", [])), tasks=list(plan_payload.get("tasks", [])))
            conversation.append({"role": "assistant", "content": f"Resuming plan:\n{plan.raw}"})
        else:
            plan = await self.planner.create_plan(user_goal, self.tools.list_tools(), model=model)
            plan_payload = self.plan_store.create_plan(user_goal, plan.tasks)
            conversation.append({"role": "assistant", "content": plan.raw})

        plan_id = str(plan_payload.get("id"))

        git_status = await self._run_tool_with_retry("git.status", "")
        conversation.append({"role": "user", "content": f"Tool git.status result:\n{git_status}"})
        if "error" not in git_status.lower():
            branch_input = json.dumps({"goal": user_goal})
            branch_result = await self._run_tool_with_retry("git.create_branch", branch_input)
            conversation.append({"role": "user", "content": f"Tool git.create_branch result:\n{branch_result}"})

        completed_indices = set(int(i) for i in plan_payload.get("completed_tasks", []))
        last_reply = plan.raw
        tests_ran = False

        for index, task in enumerate(plan.tasks):
            if index in completed_indices:
                continue

            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"Execute plan task {index + 1}/{len(plan.tasks)}: {task}\n"
                        "Use tools as needed. Keep changes minimal and safe."
                    ),
                }
            )
            last_reply, used_tools, paths_changed, conversation = await self._run_chat_loop(
                conversation,
                model=model,
                iteration_limit=self.max_iterations,
            )
            changed_paths.update(paths_changed)

            changed_files = bool(used_tools.intersection(WRITE_TOOLS))
            if changed_files:
                tests_ran = True
                test_result = await self._run_tool_with_retry("run_tests", "")
                conversation.append({"role": "user", "content": f"Automatic test result:\n{test_result}"})

                if self._extract_exit_code(test_result) != 0:
                    tests_passed = False
                    fixed = await self._run_fix_loop(conversation, changed_paths, model=model)
                    if not fixed:
                        self.plan_store.set_status(plan_id, "failed")
                        self._log_metrics(user_goal, started_at, False, changed_paths)
                        return (
                            "Stopped after reaching iteration limit while fixing failing tests.\n"
                            f"Last test output:\n{test_result}",
                            conversation,
                        )
                    tests_passed = True

            self.plan_store.mark_task_completed(plan_id, index)

        if tests_ran:
            final_test_result = await self._run_tool_with_retry("run_tests", "")
            conversation.append({"role": "user", "content": f"Final test verification:\n{final_test_result}"})
            if self._extract_exit_code(final_test_result) != 0:
                tests_passed = False
                fixed = await self._run_fix_loop(conversation, changed_paths, model=model)
                if not fixed:
                    self.plan_store.set_status(plan_id, "failed")
                    self._log_metrics(user_goal, started_at, False, changed_paths)
                    return (
                        "Plan execution finished but tests remain failing after max fix iterations.\n"
                        f"Last test output:\n{final_test_result}",
                        conversation,
                    )
                tests_passed = True

        if tests_passed and tests_ran:
            commit_message = f"Mantis: {self._summarize_goal(user_goal)}\nTests: PASS"
            commit_result = await self._run_tool_with_retry("git.commit", json.dumps({"message": commit_message}))
            conversation.append({"role": "user", "content": f"Tool git.commit result:\n{commit_result}"})

        self.plan_store.set_status(plan_id, "completed")
        self._log_metrics(user_goal, started_at, tests_passed, changed_paths)

        conversation.append(
            {
                "role": "user",
                "content": "Provide the final user-facing summary with what changed and current test status.",
            }
        )
        last_reply, _, _, conversation = await self._run_chat_loop(
            conversation,
            model=model,
            iteration_limit=self.max_iterations,
        )
        return last_reply, conversation

    async def _run_fix_loop(
        self,
        conversation: List[Dict[str, str]],
        changed_paths: Set[str],
        model: str | None = None,
    ) -> bool:
        for _ in range(self.max_iterations):
            conversation.append(
                {
                    "role": "user",
                    "content": "Tests are failing. Fix the failure with minimal diffs, then stop for test execution.",
                }
            )
            _, used_tools, paths_changed, conversation = await self._run_chat_loop(
                conversation,
                model=model,
                iteration_limit=self.max_iterations,
            )
            changed_paths.update(paths_changed)

            if not used_tools.intersection(WRITE_TOOLS):
                return False

            test_result = await self._run_tool_with_retry("run_tests", "")
            conversation.append({"role": "user", "content": f"Test rerun result:\n{test_result}"})
            if self._extract_exit_code(test_result) == 0:
                return True

        return False

    async def _run_chat_loop(
        self,
        conversation: List[Dict[str, str]],
        model: str | None = None,
        iteration_limit: int = 5,
    ) -> Tuple[str, Set[str], Set[str], List[Dict[str, str]]]:
        used_tools: Set[str] = set()
        changed_paths: Set[str] = set()
        last_reply = ""

        for _ in range(iteration_limit):
            reply = await self._chat_with_retry(conversation, model=model)
            if reply.startswith("[LLM error"):
                return reply, used_tools, changed_paths, conversation

            tool_call = self._parse_tool_call(reply)
            if tool_call:
                tool_name, tool_input = tool_call
                used_tools.add(tool_name)
                conversation.append({"role": "assistant", "content": reply})

                if tool_name in WRITE_TOOLS:
                    tool_result, approved = await self._execute_write_tool_with_confirmation(
                        conversation,
                        tool_name,
                        tool_input,
                        model=model,
                    )
                    if approved:
                        changed_paths.update(self._extract_changed_paths(tool_name, tool_input))
                else:
                    tool_result = await self._run_tool_with_retry(tool_name, tool_input)

                conversation.append({"role": "user", "content": f"Tool {tool_name} result:\n{tool_result}"})
                continue

            conversation.append({"role": "assistant", "content": reply})
            last_reply = reply
            break

        return last_reply, used_tools, changed_paths, conversation

    async def _execute_write_tool_with_confirmation(
        self,
        conversation: List[Dict[str, str]],
        tool_name: str,
        tool_input: str,
        model: str | None = None,
    ) -> Tuple[str, bool]:
        preview_payload = self._build_preview_payload(tool_name, tool_input)
        preview_diff = await self._run_tool_with_retry("git.diff", json.dumps(preview_payload))

        prompt = (
            "Diff preview for pending write action:\n"
            f"{preview_diff}\n\n"
            "Confirm changes? yes/no\n"
            "Reply with exactly one word: yes or no."
        )
        conversation.append({"role": "user", "content": prompt})
        decision = (await self._chat_with_retry(conversation, model=model)).strip().lower()
        conversation.append({"role": "assistant", "content": decision})

        if decision != "yes":
            return "Write skipped: change not confirmed by the agent.", False

        return await self._run_tool_with_retry(tool_name, tool_input), True

    async def _chat_with_retry(self, conversation: List[Dict[str, str]], model: str | None = None) -> str:
        attempts = max(0, MANTIS_RETRY_MAX_RETRIES) + 1
        for attempt in range(attempts):
            reply = await self.llm.chat(conversation, model=model)
            if not reply.startswith("[LLM error"):
                return reply
            if attempt + 1 < attempts:
                await asyncio.sleep(MANTIS_RETRY_BASE_DELAY_SEC * (2**attempt))
        return reply

    async def _run_tool_with_retry(self, tool_name: str, tool_input: str) -> str:
        attempts = max(0, MANTIS_RETRY_MAX_RETRIES) + 1 if tool_name in RETRY_TOOLS else 1
        last_result = ""
        for attempt in range(attempts):
            result = await self.tools.run(tool_name, tool_input)
            last_result = result
            if not self._should_retry_tool(tool_name, result):
                return result
            if attempt + 1 < attempts:
                await asyncio.sleep(MANTIS_RETRY_BASE_DELAY_SEC * (2**attempt))
        return last_result

    @staticmethod
    def _should_retry_tool(tool_name: str, output: str) -> bool:
        lowered = output.lower()
        if tool_name == "run_tests":
            return "test execution failed" in lowered
        if tool_name in WRITE_TOOLS:
            return f"{tool_name} error:" in lowered
        return False

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

    @staticmethod
    def _extract_changed_paths(tool_name: str, tool_input: str) -> Set[str]:
        if tool_name not in WRITE_TOOLS:
            return set()
        try:
            payload = json.loads(tool_input)
            path = str(payload.get("path", "")).strip()
            return {path} if path else set()
        except Exception:
            return set()

    @staticmethod
    def _build_preview_payload(tool_name: str, tool_input: str) -> Dict[str, object]:
        try:
            payload = json.loads(tool_input)
        except Exception:
            return {}

        path = str(payload.get("path", "")).strip()
        if not path:
            return {}

        if tool_name in {"create_file", "write_file"}:
            return {"path": path, "proposed_content": str(payload.get("content", ""))}
        if tool_name == "delete_file":
            return {"path": path, "delete": True}
        if tool_name == "patch_file":
            diff_text = str(payload.get("diff", ""))
            if not diff_text:
                return {"path": path}
            try:
                from tools import filesystem

                target = (filesystem.WORKSPACE_ROOT / path).resolve()
                before = target.read_text(encoding="utf-8") if target.exists() else ""
                after = filesystem._apply_unified_diff(before, diff_text)
                return {"path": path, "proposed_content": after}
            except Exception:
                return {"path": path}
        return {}

    @staticmethod
    def _format_plan(tasks: object) -> str:
        if not isinstance(tasks, list):
            return "PLAN:\n1. Resume goal execution"
        lines = ["PLAN:"]
        for idx, task in enumerate(tasks, start=1):
            lines.append(f"{idx}. {task}")
        return "\n".join(lines)

    @staticmethod
    def _summarize_goal(goal: str, max_len: int = 72) -> str:
        clean = " ".join(goal.split())
        if len(clean) <= max_len:
            return clean
        return clean[: max_len - 3] + "..."

    def _log_metrics(self, goal: str, started_at: float, tests_passed: bool, changed_paths: Set[str]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        duration = max(0.0, time.monotonic() - started_at)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "goal": goal,
            "duration_sec": round(duration, 3),
            "tests_passed": tests_passed,
            "files_changed": len({path for path in changed_paths if path}),
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
