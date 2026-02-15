import os
import socket
from urllib.parse import urlparse


def has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def has_anthropic() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def has_ollama() -> bool:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    try:
        connection = socket.create_connection((host, port), 1)
        connection.close()
        return True
    except OSError:
        return False
