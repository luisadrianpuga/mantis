import os
from typing import Dict, List
from uuid import uuid4

import chromadb


class VectorStore:
    """
    Lightweight wrapper for Chroma persistent collections.
    """

    def __init__(self, persist_directory: str | None = None, collection_name: str = "mantis-memory") -> None:
        self.persist_directory = persist_directory or os.getenv("MANTIS_CHROMA_DIR", ".mantis/chroma")
        os.makedirs(self.persist_directory, exist_ok=True)
        client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, text: str, embedding: List[float], metadata: Dict | None = None) -> None:
        self.collection.add(
            ids=[str(uuid4())],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )

    def query(self, embedding: List[float], k: int = 5) -> Dict:
        return self.collection.query(query_embeddings=[embedding], n_results=k)
