import os, sys
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "research")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qclient = QdrantClient(url=QDRANT_URL)

def search(query: str, k: int = 5):
    vec = embedder.encode(query).tolist()
    r = qclient.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=k,
        with_payload=True,
    )
    return r

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "最新のTransformerの学習安定化手法"
    hits = search(q)
    for i, h in enumerate(hits, 1):
        p = h.payload
        print(f"[{i}] {p.get('title')}\nURL: {p.get('url')}\n要約: {p.get('summary')[:200]}...\n")
