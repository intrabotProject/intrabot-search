from abc import ABC, abstractmethod

from app.domain.models.retrieved_chunk import RetrievedChunk


class IVectorStore(ABC):
    """
    Contract for a semantic vector store that can retrieve the most similar
    document chunks for a given query embedding.
    Concrete implementations (ChromaVectorStore, QdrantVectorStore …) must
    honour this contract — in particular they MUST return an empty list when
    no chunk is sufficiently similar (never raise on "no results").
    """

    @abstractmethod
    def query_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        source_filter: str | None = None,
        allowed_categories: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Return up to top_k chunks whose embedding is most similar to
        query_embedding, ordered by descending similarity score.
        When source_filter is set, only chunks from that document are returned.
        When allowed_categories is set, only chunks in those categories are returned.
        An empty allowed_categories list returns no chunks (deny-all).
        Returns an empty list if the collection is empty or top_k == 0.
        """
