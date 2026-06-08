import numpy as np
import chromadb

from app.domain.interfaces.secondary.vector_store.vector_store import IVectorStore
from app.domain.models.retrieved_chunk import RetrievedChunk

CHROMA_COLLECTION_NAME: str = "intrabot"
COSINE_SIMILARITY_FROM_DISTANCE_OFFSET: float = 1.0


class ChromaVectorStore(IVectorStore):
    """
    Secondary adapter — implements the IVectorStore driven port using ChromaDB.

    Metadata key mapping (ingestion service → search service):
        "source"      → document_filename  (DoclingChunker uses "source", not "filename")
        "chunk_index" → chunk_index
        "headings"    → ignored (display only, not used in retrieval)
    """

    def __init__(self, chroma_client: chromadb.ClientAPI) -> None:
        self._collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def query_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        available_chunk_count: int = self._collection.count()
        if available_chunk_count == 0 or top_k == 0:
            return []

        safe_top_k: int = min(top_k, available_chunk_count)

        query_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_top_k,
            include=["documents", "metadatas", "distances"],
        )

        raw_distances: list[float] = query_results["distances"][0]
        # Vectorial conversion: cosine distance → similarity score
        similarity_scores: np.ndarray = (
            COSINE_SIMILARITY_FROM_DISTANCE_OFFSET - np.array(raw_distances)
        )

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                # ingestion-service stores filename under "source" key (DoclingChunker)
                document_filename=metadata.get("source", metadata.get("filename", "unknown")),
                content=document_text,
                similarity_score=float(score),
                chunk_index=int(metadata.get("chunk_index", 0)),
            )
            for chunk_id, metadata, document_text, score in zip(
                query_results["ids"][0],
                query_results["metadatas"][0],
                query_results["documents"][0],
                similarity_scores,
            )
        ]
