from typing import Dict, List

from storage.vectordb import VectorStore


class MemoryManager:
    """
    Edge-mode memory for low-power devices.
    Uses keyword recall instead of embeddings.
    """

    def __init__(self, vectordb: VectorStore) -> None:
        self.vectordb = vectordb

    def store_memory(self, text: str, metadata: Dict) -> None:
        self.vectordb.add(text=text, embedding=[0.0], metadata=metadata)

    def retrieve_memory(self, query: str, k: int = 5) -> List[Dict]:
        results = self.vectordb.collection.get(include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        matches: List[Dict] = []
        for doc, meta in zip(docs, metas):
            if query.lower() in doc.lower():
                matches.append({"text": doc, "metadata": meta})
                if len(matches) >= k:
                    break
        return matches
