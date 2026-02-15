from typing import Dict, List

from sentence_transformers import SentenceTransformer

from storage.vectordb import VectorStore


class MemoryManager:
    """
    Handles embedding, storage, and retrieval of long-term memory.
    """

    def __init__(self, vectordb: VectorStore, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.vectordb = vectordb
        self.embedder = SentenceTransformer(model_name)

    def store_memory(self, text: str, metadata: Dict) -> None:
        embedding = self._embed(text)
        self.vectordb.add(text=text, embedding=embedding, metadata=metadata)

    def retrieve_memory(self, query: str, k: int = 5) -> List[Dict]:
        embedding = self._embed(query)
        results = self.vectordb.query(embedding=embedding, k=k)
        memories: List[Dict] = []
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        if documents:
            docs = documents[0] if isinstance(documents[0], list) else documents
            metas = metadatas[0] if metadatas and isinstance(metadatas[0], list) else metadatas
            for doc, meta in zip(docs, metas):
                memories.append({"text": doc, "metadata": meta})
        return memories

    def _embed(self, text: str) -> List[float]:
        vector = self.embedder.encode([text])[0]
        return vector.tolist()
