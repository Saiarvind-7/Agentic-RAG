"""Hybrid retrieval: dense vectors (ChromaDB) + keywords (BM25), merged with RRF."""
import json

import numpy as np
from rank_bm25 import BM25Okapi

from app import embeddings
from app.config import CANDIDATES, CHROMA_DIR, CHUNKS_FILE, RRF_K, TOP_K
from app.fusion import reciprocal_rank_fusion, tokenize
from app.ingest import COLLECTION


class IndexNotBuilt(Exception):
    pass


class HybridRetriever:
    def __init__(self):
        import chromadb

        if not CHUNKS_FILE.exists():
            raise IndexNotBuilt("The index has not been built yet. Run `python -m app.ingest` first.")

        with CHUNKS_FILE.open(encoding="utf-8") as f:
            self.chunks = {c["id"]: c for c in map(json.loads, f)}
        self.ids = list(self.chunks)
        self.bm25 = BM25Okapi([tokenize(self.chunks[i]["text"]) for i in self.ids])

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION)

    def dense_search(self, query: str, n: int = CANDIDATES) -> list[str]:
        """Chunk ids ranked by meaning (cosine similarity of embeddings)."""
        result = self.collection.query(
            query_embeddings=embeddings.embed([query]),
            n_results=min(n, len(self.ids)),
        )
        return result["ids"][0]

    def keyword_search(self, query: str, n: int = CANDIDATES) -> list[str]:
        """Chunk ids ranked by BM25. Chunks with no matching word are dropped."""
        scores = self.bm25.get_scores(tokenize(query))
        best = np.argsort(scores)[::-1][:n]
        return [self.ids[i] for i in best if scores[i] > 0]

    def search(self, query: str, top_k: int = TOP_K, mode: str = "hybrid") -> list[dict]:
        """mode is 'hybrid', 'dense' or 'bm25'. The last two are handy for comparing."""
        dense = self.dense_search(query) if mode in ("hybrid", "dense") else []
        keyword = self.keyword_search(query) if mode in ("hybrid", "bm25") else []

        fused = reciprocal_rank_fusion([dense, keyword], k=RRF_K)[:top_k]
        results = []
        for chunk_id, score in fused:
            found_by = [name for name, ids in (("dense", dense), ("bm25", keyword)) if chunk_id in ids]
            results.append({**self.chunks[chunk_id], "score": round(score, 5), "found_by": found_by})
        return results
