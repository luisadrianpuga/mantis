import re
from typing import Dict, List, Optional, Tuple

from agent.memory import MemoryManager
from agent.prompt import build_system_prompt
from permissions.policy import check_permission
from providers.router import ProviderRouter
from tools.registry import ToolRegistry


class AgentLoop:
    """
    Minimal agent loop:
    - build prompt with memory + tool list
    - call LLM
    - detect tool call pattern
    - execute tool and feed result back
    - return final reply
    """

    def __init__(
        self,
        llm_client: ProviderRouter,
        tools: ToolRegistry,
        memory: MemoryManager,
        max_iterations: int = 5,
    ) -> None:
        self.llm = llm_client
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations

    async def run(self, messages: List[Dict[str, str]] | str, model: str | None = None) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        latest_user_message = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        memories = self.memory.retrieve_memory(latest_user_message) if latest_user_message else []
        system_prompt = build_system_prompt(self.tools.list_tools(), memories)
        conversation: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}] + messages

        for _ in range(self.max_iterations):
            reply = await self.llm.chat(conversation, model=model)
            if reply.startswith("[LLM error"):
                return reply
            tool_call = self._parse_tool_call(reply)
            if tool_call:
                tool_name, tool_input = tool_call
                if not check_permission(tool_name):
                    return "Tool requires approval."
                conversation.append({"role": "assistant", "content": reply})
                tool_result = await self.tools.run(tool_name, tool_input)
                # Surface tool result back to the model as user content.
                conversation.append({"role": "user", "content": f"Tool {tool_name} result:\n{tool_result}"})
                continue

            conversation.append({"role": "assistant", "content": reply})
            if latest_user_message and len(latest_user_message) > 40:
                summary = f"User: {latest_user_message}\nAssistant: {reply}"
                self.memory.store_memory(summary, {"source": "chat", "type": "exchange"})
            return reply

        return "Reached iteration limit without final answer."

    @staticmethod
    def _parse_tool_call(text: str) -> Optional[Tuple[str, str]]:
        """
        Detects TOOL: tool_name | input pattern.
        """
        for line in text.splitlines():
            match = re.match(r"TOOL:\s*([\w\.-]+)\s*\|\s*(.+)", line.strip())
            if match:
                return match.group(1), match.group(2)
        return None
