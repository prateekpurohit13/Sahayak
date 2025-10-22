import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Read the Qdrant URL from an environment variable for deployment flexibility
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

class QdrantStorage:
    # --- CHANGE: Updated the default dimension from 1024 to 1536 ---
    def __init__(self, url=QDRANT_URL, collection="docs", dim=1536):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        # This will now create a collection with the correct vector size
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5, language: str = None):
        query_filter = None
        if language:
            query_filter = Filter(
                must=[FieldCondition(key="language", match=MatchValue(value=language))]
            )
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k,
            query_filter=query_filter
        )
        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
