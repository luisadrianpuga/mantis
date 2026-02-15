from typing import List, Optional
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from agent.loop import AgentLoop
from agent.memory import MemoryManager
from llm.ollama import OllamaClient
from storage.vectordb import VectorStore
from tools.registry import ToolRegistry

router = APIRouter()
logger = logging.getLogger("mantis")

# Instantiate core runtime components once for the app lifecycle.
vector_store = VectorStore()
memory_manager = MemoryManager(vector_store)
tool_registry = ToolRegistry()
ollama_client = OllamaClient()
agent_loop = AgentLoop(ollama_client, tool_registry, memory_manager)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]


@router.get("/v1/models")
async def list_models():
    return {"data": [{"id": "mantis-local-agent"}]}


@router.post("/v1/chat/completions")
async def chat_completions(payload: ChatRequest):
    logger.info("Incoming chat request with %d messages", len(payload.messages))
    reply = await agent_loop.run(
        [message.model_dump() for message in payload.messages],
        model=payload.model,
    )
    logger.info("Agent reply generated")
    return {"choices": [{"message": {"role": "assistant", "content": reply}}]}
