from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.domain.interfaces.primary.search.search_service import ISearchService
from app.domain.models.search_response import SearchResponse, SourceChunk
from app.factories.provider_factory import get_search_service
from app.main import app

_MOCK_RESPONSE_WITH_SOURCE = SearchResponse(
    answer="Le télétravail est autorisé jusqu'à 3 jours par semaine.",
    sources=[
        SourceChunk(
            chunk_id="chunk-001",
            filename="rh_politique_teletravail_2025.pdf",
            excerpt="Le télétravail est autorisé...",
            similarity_score=0.91,
        )
    ],
    latency_ms=320,
)

_MOCK_RESPONSE_NO_SOURCE = SearchResponse(
    answer="I cannot find the answer to this question in the available documents.",
    sources=[],
    latency_ms=80,
)


@pytest.fixture
def client_with_matching_answer() -> TestClient:
    mock_service = MagicMock(spec=ISearchService)
    mock_service.search.return_value = _MOCK_RESPONSE_WITH_SOURCE
    app.dependency_overrides[get_search_service] = lambda: mock_service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def client_with_no_source_answer() -> TestClient:
    mock_service = MagicMock(spec=ISearchService)
    mock_service.search.return_value = _MOCK_RESPONSE_NO_SOURCE
    app.dependency_overrides[get_search_service] = lambda: mock_service
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestSearchEndpoint:

    def test_valid_request_returns_200(self, client_with_matching_answer) -> None:
        response = client_with_matching_answer.post(
            "/api/v1/search", json={"question": "Politique télétravail ?"}
        )
        assert response.status_code == 200

    def test_response_body_contains_answer_sources_latency(
        self, client_with_matching_answer
    ) -> None:
        response = client_with_matching_answer.post(
            "/api/v1/search", json={"question": "Politique télétravail ?"}
        )
        body = response.json()
        assert "answer" in body and "sources" in body and "latency_ms" in body

    def test_custom_top_k_is_accepted(self, client_with_matching_answer) -> None:
        response = client_with_matching_answer.post(
            "/api/v1/search", json={"question": "Question", "top_k": 3}
        )
        assert response.status_code == 200

    def test_no_source_response_has_empty_sources_list(
        self, client_with_no_source_answer
    ) -> None:
        response = client_with_no_source_answer.post(
            "/api/v1/search", json={"question": "Fournitures de bureau ?"}
        )
        assert response.json()["sources"] == []

    def test_mock_is_instance_of_primary_port(self) -> None:
        mock_service = MagicMock(spec=ISearchService)
        assert isinstance(mock_service, ISearchService)

    def test_empty_question_returns_422(self) -> None:
        assert TestClient(app).post("/api/v1/search", json={"question": ""}).status_code == 422

    def test_missing_question_returns_422(self) -> None:
        assert TestClient(app).post("/api/v1/search", json={}).status_code == 422

    def test_health_returns_200(self) -> None:
        assert TestClient(app).get("/health").status_code == 200
