from unittest.mock import MagicMock

import pytest

from app.application.search.rag_search_service import RAGSearchService
from app.domain.interfaces.primary.search.search_service import ISearchService
from app.domain.interfaces.secondary.embedding.embedding_provider import IEmbeddingProvider
from app.domain.interfaces.secondary.llm.llm_provider import ILLMProvider
from app.domain.interfaces.secondary.prompt.prompt_builder import IPromptBuilder
from app.domain.interfaces.secondary.vector_store.vector_store import IVectorStore
from app.domain.models.retrieved_chunk import RetrievedChunk
from app.domain.models.search_request import SearchRequest
from app.domain.models.search_response import SearchResponse
from tests.fixtures.documents import (
    CHUNK_CI_CD_PIPELINE,
    CHUNK_TELEWORK_POLICY,
    QUESTION_WITH_NO_MATCHING_SOURCE,
)
from tests.fixtures.embeddings import QUERY_EMBEDDING_TELEWORK, QUERY_EMBEDDING_UNKNOWN_TOPIC

NO_ANSWER_SENTINEL: str = (
    "I cannot find the answer to this question in the available documents."
)


def _make_retrieved_chunk(fixture, score: float = 0.90) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=fixture.chunk_id,
        document_filename=fixture.source,    # .source (not .document_filename)
        content=fixture.text,                # .text (not .content)
        similarity_score=score,
        chunk_index=fixture.chunk_index,
    )


class TestRAGSearchService:
    """
    Battery of tests for RAGSearchService.

    LSP note: RAGSearchService implements ISearchService (primary port).
    Tests verify that the concrete class honours the primary port contract,
    and that it correctly delegates to secondary ports via their interfaces.
    """

    def setup_method(self) -> None:
        self._mock_embedding_provider = MagicMock(spec=IEmbeddingProvider)
        self._mock_vector_store = MagicMock(spec=IVectorStore)
        self._mock_llm_provider = MagicMock(spec=ILLMProvider)
        self._mock_prompt_builder = MagicMock(spec=IPromptBuilder)

        self._service = RAGSearchService(
            embedding_provider=self._mock_embedding_provider,
            vector_store=self._mock_vector_store,
            llm_provider=self._mock_llm_provider,
            prompt_builder=self._mock_prompt_builder,
        )

    def test_rag_search_service_is_instance_of_primary_port(self) -> None:
        """
        LSP: RAGSearchService must be substitutable for ISearchService.
        Any code depending on ISearchService must work with this concrete class.
        """
        assert isinstance(self._service, ISearchService)

    def test_search_returns_a_search_response_instance(self) -> None:
        self._mock_embedding_provider.embed_text.return_value = QUERY_EMBEDDING_TELEWORK
        self._mock_vector_store.query_similar_chunks.return_value = [
            _make_retrieved_chunk(CHUNK_TELEWORK_POLICY)
        ]
        self._mock_prompt_builder.build_rag_prompt.return_value = "augmented prompt"
        self._mock_llm_provider.generate_answer.return_value = "3 jours par semaine."

        response = self._service.search(SearchRequest(question="Politique télétravail ?"))
        assert isinstance(response, SearchResponse)

    def test_search_embeds_the_exact_user_question(self) -> None:
        self._mock_embedding_provider.embed_text.return_value = QUERY_EMBEDDING_TELEWORK
        self._mock_vector_store.query_similar_chunks.return_value = []
        self._mock_prompt_builder.build_rag_prompt.return_value = "p"
        self._mock_llm_provider.generate_answer.return_value = NO_ANSWER_SENTINEL

        self._service.search(SearchRequest(question="Ma question exacte"))
        self._mock_embedding_provider.embed_text.assert_called_once_with("Ma question exacte")

    def test_search_queries_vector_store_with_correct_top_k(self) -> None:
        self._mock_embedding_provider.embed_text.return_value = QUERY_EMBEDDING_TELEWORK
        self._mock_vector_store.query_similar_chunks.return_value = []
        self._mock_prompt_builder.build_rag_prompt.return_value = "p"
        self._mock_llm_provider.generate_answer.return_value = NO_ANSWER_SENTINEL

        self._service.search(SearchRequest(question="Question", top_k=3))
        self._mock_vector_store.query_similar_chunks.assert_called_once_with(
            query_embedding=QUERY_EMBEDDING_TELEWORK,
            top_k=3,
            source_filter=None,
            allowed_categories=None,
        )

    def test_sources_count_matches_retrieved_chunks_count(self) -> None:
        chunks = [
            _make_retrieved_chunk(CHUNK_TELEWORK_POLICY, score=0.92),
            _make_retrieved_chunk(CHUNK_CI_CD_PIPELINE, score=0.78),
        ]
        self._mock_embedding_provider.embed_text.return_value = QUERY_EMBEDDING_TELEWORK
        self._mock_vector_store.query_similar_chunks.return_value = chunks
        self._mock_prompt_builder.build_rag_prompt.return_value = "p"
        self._mock_llm_provider.generate_answer.return_value = "Réponse."

        response = self._service.search(SearchRequest(question="Question"))
        assert len(response.sources) == 2

    def test_source_similarity_score_is_preserved(self) -> None:
        expected_score: float = 0.87
        self._mock_embedding_provider.embed_text.return_value = QUERY_EMBEDDING_TELEWORK
        self._mock_vector_store.query_similar_chunks.return_value = [
            _make_retrieved_chunk(CHUNK_TELEWORK_POLICY, score=expected_score)
        ]
        self._mock_prompt_builder.build_rag_prompt.return_value = "p"
        self._mock_llm_provider.generate_answer.return_value = "Réponse."

        response = self._service.search(SearchRequest(question="Question"))
        assert response.sources[0].similarity_score == expected_score

    def test_response_latency_is_non_negative(self) -> None:
        self._mock_embedding_provider.embed_text.return_value = QUERY_EMBEDDING_TELEWORK
        self._mock_vector_store.query_similar_chunks.return_value = []
        self._mock_prompt_builder.build_rag_prompt.return_value = "p"
        self._mock_llm_provider.generate_answer.return_value = NO_ANSWER_SENTINEL

        response = self._service.search(SearchRequest(question="Question"))
        assert response.latency_ms >= 0

    # ── LSP / hallucination-prevention edge cases ─────────────────────────────

    def test_prompt_builder_receives_empty_list_when_no_chunks_retrieved(self) -> None:
        """
        LSP contract on IVectorStore: returning [] is valid.
        The service must forward the empty list to IPromptBuilder
        to activate the hallucination guardrail.
        """
        self._mock_embedding_provider.embed_text.return_value = QUERY_EMBEDDING_UNKNOWN_TOPIC
        self._mock_vector_store.query_similar_chunks.return_value = []
        self._mock_prompt_builder.build_rag_prompt.return_value = "no-context prompt"
        self._mock_llm_provider.generate_answer.return_value = NO_ANSWER_SENTINEL

        self._service.search(SearchRequest(question=QUESTION_WITH_NO_MATCHING_SOURCE))

        self._mock_prompt_builder.build_rag_prompt.assert_called_once_with(
            user_question=QUESTION_WITH_NO_MATCHING_SOURCE,
            retrieved_chunks=[],
        )

    def test_response_has_no_sources_when_no_chunks_retrieved(self) -> None:
        self._mock_embedding_provider.embed_text.return_value = QUERY_EMBEDDING_UNKNOWN_TOPIC
        self._mock_vector_store.query_similar_chunks.return_value = []
        self._mock_prompt_builder.build_rag_prompt.return_value = "p"
        self._mock_llm_provider.generate_answer.return_value = NO_ANSWER_SENTINEL

        response = self._service.search(
            SearchRequest(question=QUESTION_WITH_NO_MATCHING_SOURCE)
        )
        assert response.sources == []

    def test_llm_is_called_even_when_no_chunks_retrieved(self) -> None:
        self._mock_embedding_provider.embed_text.return_value = QUERY_EMBEDDING_UNKNOWN_TOPIC
        self._mock_vector_store.query_similar_chunks.return_value = []
        self._mock_prompt_builder.build_rag_prompt.return_value = "no-context prompt"
        self._mock_llm_provider.generate_answer.return_value = NO_ANSWER_SENTINEL

        self._service.search(SearchRequest(question=QUESTION_WITH_NO_MATCHING_SOURCE))
        self._mock_llm_provider.generate_answer.assert_called_once()
