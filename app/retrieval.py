from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

LOCATION_QUERY_MAP = {
    "us": "US English accent United States",
    "uk": "UK English accent United Kingdom",
    "india": "Indian English accent India",
    "australia": "Australian English accent Australia",
    "canada": "Canadian English accent Canada",
}


class CatalogRetriever:

    def __init__(self, catalog: list[dict[str, Any]]):
        self.catalog = catalog
        self._model = None
        self._index = None
        self._init_embeddings()

    def _init_embeddings(self):
        if not self.catalog:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self._model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu",
            )
            texts = [item["_search_text"] for item in self.catalog]
            embeddings = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(np.array(embeddings).astype(np.float32))
            logger.info(
                "Embedding index built with %d items (dim=%d)", len(texts), dim
            )
        except Exception as exc:
            logger.warning("Embedding init failed, falling back to keyword search: %s", exc)
            self._model = None
            self._index = None

    def search(self, query: str, k: int = 15) -> list[dict[str, Any]]:
        if not self.catalog:
            return []
        semantic_results = self._semantic_search(query, k) if self._index else []
        keyword_results = self._keyword_search(query)
        seen = set()
        merged = []
        for item in semantic_results + keyword_results:
            uid = item.get("entity_id") or item.get("name", "")
            if uid not in seen:
                seen.add(uid)
                merged.append(item)
        return merged[:k]

    def _semantic_search(
        self, query: str, k: int
    ) -> list[dict[str, Any]]:
        vec = self._model.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(
            np.array(vec).astype(np.float32), k
        )
        results = []
        for idx, score in zip(indices[0], scores[0]):
            item = dict(self.catalog[int(idx)])
            results.append(item)
        return results

    def _keyword_search(self, query: str) -> list[dict[str, Any]]:
        q = query.lower()
        terms = set(q.split())
        scored = []
        for item in self.catalog:
            name = item.get("name", "").lower()
            desc = item.get("description", "").lower()
            match_count = sum(1 for t in terms if t in name or t in desc)
            if match_count > 0:
                scored.append((match_count, item))
        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored]

    def search_by_name(self, query: str) -> list[dict[str, Any]]:
        q = query.lower()
        results = []
        for item in self.catalog:
            if q in item.get("name", "").lower():
                results.append(item)
        return results
