from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.embedding.ingestion_service_embedding_adapter import (
    COHERE_EMBED_MULTILINGUAL_V3_DIMENSION,
    IngestionServiceEmbeddingAdapter,
)
from tests.fixtures.embeddings import QUERY_EMBEDDING_TELEWORK

INGESTION_SERVICE_URL: str = "http://localhost:8000"


def _build_mock_httpx_response(embedding: list[float]) -> MagicMock:
    """Builds the shape returned by httpx.post for POST /embed."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": embedding}
    mock_response.raise_for_status = MagicMock()
    return mock_response


class TestIngestionServiceEmbeddingAdapter:
    """
    Battery of tests for IngestionServiceEmbeddingAdapter.
    httpx.post is always mocked — no real HTTP calls are made.
    """

    def setup_method(self) -> None:
        self._adapter = IngestionServiceEmbeddingAdapter(
            ingestion_service_url=INGESTION_SERVICE_URL
        )

    # ── embed_text ────────────────────────────────────────────────────────────

    def test_embed_text_calls_correct_endpoint(self) -> None:
        with patch("app.infrastructure.embedding.ingestion_service_embedding_adapter.httpx.post") as mock_post:
            mock_post.return_value = _build_mock_httpx_response(QUERY_EMBEDDING_TELEWORK)
            self._adapter.embed_text("Politique de télétravail")
            called_url = mock_post.call_args[1]["url"]
            assert called_url == f"{INGESTION_SERVICE_URL}/embed"

    def test_embed_text_sends_correct_payload(self) -> None:
        with patch("app.infrastructure.embedding.ingestion_service_embedding_adapter.httpx.post") as mock_post:
            mock_post.return_value = _build_mock_httpx_response(QUERY_EMBEDDING_TELEWORK)
            self._adapter.embed_text("Ma question")
            sent_payload = mock_post.call_args[1]["json"]
            assert sent_payload == {"text": "Ma question"}

    def test_embed_text_returns_embedding_from_response(self) -> None:
        with patch("app.infrastructure.embedding.ingestion_service_embedding_adapter.httpx.post") as mock_post:
            mock_post.return_value = _build_mock_httpx_response(QUERY_EMBEDDING_TELEWORK)
            result = self._adapter.embed_text("Question test")
            assert result == QUERY_EMBEDDING_TELEWORK

    def test_embed_text_returns_vector_of_correct_dimension(self) -> None:
        with patch("app.infrastructure.embedding.ingestion_service_embedding_adapter.httpx.post") as mock_post:
            mock_post.return_value = _build_mock_httpx_response(QUERY_EMBEDDING_TELEWORK)
            result = self._adapter.embed_text("Question test")
            assert len(result) == COHERE_EMBED_MULTILINGUAL_V3_DIMENSION

    def test_embed_text_calls_raise_for_status(self) -> None:
        """Ensures HTTP errors are propagated rather than swallowed."""
        with patch("app.infrastructure.embedding.ingestion_service_embedding_adapter.httpx.post") as mock_post:
            mock_response = _build_mock_httpx_response(QUERY_EMBEDDING_TELEWORK)
            mock_post.return_value = mock_response
            self._adapter.embed_text("Question")
            mock_response.raise_for_status.assert_called_once()

    def test_trailing_slash_in_url_is_stripped(self) -> None:
        """LSP edge case: URL with trailing slash must not produce double-slash."""
        adapter = IngestionServiceEmbeddingAdapter(
            ingestion_service_url="http://localhost:8000/"
        )
        with patch("app.infrastructure.embedding.ingestion_service_embedding_adapter.httpx.post") as mock_post:
            mock_post.return_value = _build_mock_httpx_response(QUERY_EMBEDDING_TELEWORK)
            adapter.embed_text("Question")
            called_url = mock_post.call_args[1]["url"]
            assert "//" not in called_url.replace("http://", "")

    # ── embed_batch ───────────────────────────────────────────────────────────

    def test_embed_batch_returns_one_vector_per_text(self) -> None:
        with patch("app.infrastructure.embedding.ingestion_service_embedding_adapter.httpx.post") as mock_post:
            mock_post.return_value = _build_mock_httpx_response(QUERY_EMBEDDING_TELEWORK)
            results = self._adapter.embed_batch(["t1", "t2", "t3"])
            assert len(results) == 3

    def test_embed_batch_calls_endpoint_once_per_text(self) -> None:
        with patch("app.infrastructure.embedding.ingestion_service_embedding_adapter.httpx.post") as mock_post:
            mock_post.return_value = _build_mock_httpx_response(QUERY_EMBEDDING_TELEWORK)
            self._adapter.embed_batch(["t1", "t2", "t3"])
            assert mock_post.call_count == 3

    # ── embedding_dimension property ──────────────────────────────────────────

    def test_embedding_dimension_returns_1024(self) -> None:
        assert self._adapter.embedding_dimension == COHERE_EMBED_MULTILINGUAL_V3_DIMENSION
