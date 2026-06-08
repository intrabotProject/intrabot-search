from abc import ABC, abstractmethod


class IEmbeddingProvider(ABC):
    """
    Contract for a service that converts text into dense float vectors.
    Concrete implementations (MistralEmbeddingProvider, OpenAIEmbeddingProvider …)
    must honour all three methods.
    """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string and return its dense vector representation.
        The returned list length equals self.embedding_dimension.
        """

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in a single API call (more efficient than N calls
        to embed_text). Returns a list of vectors in the same order as texts.
        """

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Dimensionality of every vector produced by this provider."""
