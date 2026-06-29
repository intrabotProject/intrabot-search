"""
Adaptateur ChromaDB — lecture des chunks indexés par intrabot-ingestion.

Contrat partagé (voir `intrabot-ingestion/app/adapters/vectorstore/chroma_store.py`) :
  - metadata `source`      → nom du fichier
  - metadata `chunk_index` → position dans le document
  - espace vectoriel       → cosine
"""

import chromadb
import numpy as np
from chromadb.errors import InternalError

from app.domain.interfaces.secondary.vector_store.vector_store import IVectorStore
from app.domain.models.retrieved_chunk import RetrievedChunk

COSINE_SIMILARITY_FROM_DISTANCE_OFFSET: float = 1.0
DEFAULT_CHUNK_CATEGORY: str = "public"


def _build_where_filter(
    source_filter: str | None,
    allowed_categories: list[str] | None,
) -> dict | None:
    """Filtre Chroma : source uniquement. Les catégories sont filtrées en Python."""
    if allowed_categories is not None and not allowed_categories:
        return {"category": "__deny_all__"}

    if source_filter:
        return {"source": source_filter}

    return None


class ChromaVectorStore(IVectorStore):
    """Implémente IVectorStore en interrogeant une collection ChromaDB existante."""

    def __init__(self, chroma_client: chromadb.ClientAPI, collection_name: str) -> None:
        try:
            self._collection = chroma_client.get_collection(name=collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"Collection '{collection_name}' not found in ChromaDB. "
                "Run POST /ingest on intrabot-ingestion before querying."
            ) from exc

    def query_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        source_filter: str | None = None,
        allowed_categories: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        where_filter = _build_where_filter(source_filter, allowed_categories)

        if where_filter == {"category": "__deny_all__"}:
            return []

        if where_filter:
            filtered = self._collection.get(where=where_filter, include=[])
            available_chunk_count = len(filtered["ids"])
        else:
            available_chunk_count = self._collection.count()

        if available_chunk_count == 0 or top_k == 0:
            return []

        safe_top_k: int = min(top_k, available_chunk_count)
        query_k = safe_top_k
        if allowed_categories is not None:
            query_k = min(available_chunk_count, max(safe_top_k * 5, safe_top_k))

        try:
            query_results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=query_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except InternalError:
            # Index Chroma corrompu ou lecture concurrente pendant l'ingestion.
            return []

        raw_distances: list[float] = query_results["distances"][0]
        similarity_scores: np.ndarray = np.clip(
            COSINE_SIMILARITY_FROM_DISTANCE_OFFSET - np.array(raw_distances),
            a_min=0.0,
            a_max=1.0,
        )

        results: list[RetrievedChunk] = []
        for chunk_id, metadata, document_text, score in zip(
            query_results["ids"][0],
            query_results["metadatas"][0],
            query_results["documents"][0],
            similarity_scores,
        ):
            meta = metadata or {}
            chunk_category = meta.get("category", DEFAULT_CHUNK_CATEGORY)

            if allowed_categories is not None and chunk_category not in allowed_categories:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_filename=meta.get("source", "unknown"),
                    content=document_text or "",
                    similarity_score=float(score),
                    chunk_index=int(meta.get("chunk_index", 0)),
                )
            )

        results.sort(key=lambda chunk: chunk.similarity_score, reverse=True)
        return results[:safe_top_k]
