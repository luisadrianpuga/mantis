import argparse
import asyncio
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["api", "worker", "chat"])
    args = parser.parse_args()

    if args.command == "api":
        subprocess.run(["uvicorn", "app:app", "--reload"], check=False)

    elif args.command == "worker":
        from worker.service import main as worker_main

        asyncio.run(worker_main())

    elif args.command == "chat":
        from identity.bootstrap import bootstrap_identity
        from providers.router import ProviderRouter
        from tools.registry import ToolRegistry
        from storage.vectordb import VectorStore
        from agent.memory import MemoryManager
        from agent.loop import AgentLoop

        identity = bootstrap_identity()
        router = ProviderRouter()
        tools = ToolRegistry()
        memory = MemoryManager(VectorStore())
        agent = AgentLoop(router, tools, memory, identity=identity)

        while True:
            try:
                msg = input("You: ")
            except EOFError:
                break
            if not msg.strip():
                continue
            if msg.strip().lower() in {"exit", "quit"}:
                break
            reply = asyncio.run(agent.run(msg))
            print("Mantis:", reply)


if __name__ == "__main__":
    main()
