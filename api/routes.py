from typing import List, Optional
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from agent.loop import AgentLoop
from agent.memory import MemoryManager
from identity.bootstrap import bootstrap_identity
from providers.router import ProviderRouter
from storage.vectordb import VectorStore
from tools.registry import ToolRegistry

router = APIRouter()
logger = logging.getLogger("mantis.api")

# Instantiate core runtime components once for the app lifecycle.
vector_store = VectorStore()
memory_manager = MemoryManager(vector_store)
tool_registry = ToolRegistry()
provider_router = ProviderRouter()
identity_manager = bootstrap_identity()
agent_loop = AgentLoop(provider_router, tool_registry, memory_manager, identity=identity_manager)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]


@router.get("/v1/models")
async def list_models():
    return {"data": [{"id": "mantis-local-agent"}]}


@router.get("/v1/identity")
async def get_identity():
    return identity_manager.get_identity_sections(recent_lines=20)


@router.post("/v1/chat/completions")
async def chat_completions(payload: ChatRequest):
    logger.info("Incoming chat request with %d messages", len(payload.messages))
    reply = await agent_loop.run(
        [message.model_dump() for message in payload.messages],
        model=payload.model,
    )
    logger.info("Agent reply generated")
    return {"choices": [{"message": {"role": "assistant", "content": reply}}]}
