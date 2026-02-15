from typing import List, Optional
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from agent.loop import AgentLoop
from agent.memory import MemoryManager
from planner.store import add_goal, load_goals
from providers.router import ProviderRouter
from storage.vectordb import VectorStore
from tools.registry import ToolRegistry

router = APIRouter()
logger = logging.getLogger("mantis")

# Instantiate core runtime components once for the app lifecycle.
vector_store = VectorStore()
memory_manager = MemoryManager(vector_store)
tool_registry = ToolRegistry()
provider_router = ProviderRouter()
agent_loop = AgentLoop(provider_router, tool_registry, memory_manager)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]


class GoalCreateRequest(BaseModel):
    title: str


@router.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": "mantis-local-agent"},
            {"id": "gpt-4o-mini"},
            {"id": "claude-3-5-sonnet-latest"},
        ]
    }


@router.get("/v1/goals")
async def list_goals():
    goals = load_goals()
    return {
        "data": [
            {
                "id": goal.id,
                "title": goal.title,
                "tasks": [
                    {
                        "id": task.id,
                        "title": task.title,
                        "description": task.description,
                        "priority": task.priority,
                        "status": task.status.value,
                        "result": task.result,
                    }
                    for task in goal.tasks
                ],
            }
            for goal in goals
        ]
    }


@router.post("/v1/goals")
async def create_goal(payload: GoalCreateRequest):
    goal = add_goal(payload.title)
    return {"id": goal.id, "title": goal.title}


@router.post("/v1/chat/completions")
async def chat_completions(payload: ChatRequest):
    logger.info("Incoming chat request with %d messages", len(payload.messages))
    reply = await agent_loop.run(
        [message.model_dump() for message in payload.messages],
        model=payload.model,
    )
    logger.info("Agent reply generated")
    return {"choices": [{"message": {"role": "assistant", "content": reply}}]}
