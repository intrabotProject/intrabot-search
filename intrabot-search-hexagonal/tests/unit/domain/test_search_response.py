import pytest
from pydantic import ValidationError

from app.domain.models.search_response import SearchResponse, SourceChunk


class TestSourceChunk:
    """Tests for the SourceChunk API response fragment."""

    def test_valid_source_chunk_is_accepted(self) -> None:
        chunk = SourceChunk(
            chunk_id="chunk-001",
            filename="rh_politique_teletravail_2025.pdf",
            excerpt="Le télétravail est autorisé jusqu'à 3 jours...",
            similarity_score=0.91,
        )
        assert chunk.similarity_score == 0.91

    def test_similarity_score_of_exactly_zero_is_accepted(self) -> None:
        chunk = SourceChunk(
            chunk_id="c", filename="f.pdf", excerpt="e", similarity_score=0.0
        )
        assert chunk.similarity_score == 0.0

    def test_similarity_score_of_exactly_one_is_accepted(self) -> None:
        chunk = SourceChunk(
            chunk_id="c", filename="f.pdf", excerpt="e", similarity_score=1.0
        )
        assert chunk.similarity_score == 1.0

    def test_similarity_score_above_one_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            SourceChunk(chunk_id="c", filename="f.pdf", excerpt="e", similarity_score=1.01)

    def test_negative_similarity_score_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            SourceChunk(chunk_id="c", filename="f.pdf", excerpt="e", similarity_score=-0.01)


class TestSearchResponse:
    """Tests for the full SearchResponse envelope."""

    def test_valid_response_with_sources_is_accepted(self) -> None:
        response = SearchResponse(
            answer="Le télétravail est autorisé 3 jours par semaine.",
            sources=[
                SourceChunk(
                    chunk_id="chunk-001",
                    filename="rh_politique_teletravail_2025.pdf",
                    excerpt="...",
                    similarity_score=0.91,
                )
            ],
            latency_ms=320,
        )
        assert len(response.sources) == 1
        assert response.latency_ms == 320

    def test_response_with_empty_sources_is_accepted(self) -> None:
        response = SearchResponse(answer="No answer found.", sources=[], latency_ms=50)
        assert response.sources == []

    def test_latency_of_zero_is_accepted(self) -> None:
        response = SearchResponse(answer="A", sources=[], latency_ms=0)
        assert response.latency_ms == 0

    def test_negative_latency_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            SearchResponse(answer="A", sources=[], latency_ms=-1)
