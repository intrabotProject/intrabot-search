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
    ) -> list[RetrievedChunk]:
        """
        Return up to top_k chunks whose embedding is most similar to
        query_embedding, ordered by descending similarity score.
        Returns an empty list if the collection is empty or top_k == 0.
        """
