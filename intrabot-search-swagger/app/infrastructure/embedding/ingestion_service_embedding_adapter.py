import httpx

from app.domain.interfaces.secondary.embedding.embedding_provider import IEmbeddingProvider

# Cohere embed-multilingual-v3.0 produces 1024-dimensional vectors
COHERE_EMBED_MULTILINGUAL_V3_DIMENSION: int = 1024


class IngestionServiceEmbeddingAdapter(IEmbeddingProvider):
    """
    Secondary adapter — delegates query embedding to the intrabot-ingestion
    microservice via its POST /embed HTTP endpoint.

    This keeps the embedding logic centralised in a single service:
    intrabot-ingestion owns both document embedding (at ingestion time) and
    query embedding (at search time), guaranteeing that both use identical
    model parameters (Cohere embed-multilingual-v3.0, input_type=search_query).

    Calling the same service for both sides of the vector space is essential
    for cosine similarity to be meaningful.
    """

    def __init__(self, ingestion_service_url: str) -> None:
        # Strip trailing slash to avoid double-slash in URL construction
        self._base_url: str = ingestion_service_url.rstrip("/")

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single query string by calling POST /embed on the ingestion service.
        Uses input_type='search_query' on the Cohere side (handled by the ingestion service).
        """
        response = httpx.post(
            url=f"{self._base_url}/embed",
            json={"text": text},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts by making one HTTP call per text.
        The ingestion service /embed endpoint is single-text only.
        """
        return [self.embed_text(text) for text in texts]

    @property
    def embedding_dimension(self) -> int:
        return COHERE_EMBED_MULTILINGUAL_V3_DIMENSION
