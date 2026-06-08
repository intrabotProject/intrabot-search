import numpy as np
import pytest
import chromadb

from app.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore
from tests.fixtures.documents import (
    ALL_CHUNK_FIXTURES, CHUNK_CI_CD_PIPELINE, CHUNK_TELEWORK_POLICY,
)
from tests.fixtures.embeddings import (
    ALL_CHUNK_EMBEDDINGS, QUERY_EMBEDDING_CI_CD,
    QUERY_EMBEDDING_TELEWORK, QUERY_EMBEDDING_UNKNOWN_TOPIC,
)


@pytest.fixture
def populated_vector_store() -> ChromaVectorStore:
    """
    In-memory ChromaDB pre-populated using the ingestion-service metadata schema:
        { "source": filename, "chunk_index": int, "headings": str }
    This mirrors exactly what DoclingChunker produces.
    """
    in_memory_client = chromadb.Client()
    store = ChromaVectorStore(chroma_client=in_memory_client)
    store._collection.add(
        ids=[f.chunk_id for f in ALL_CHUNK_FIXTURES],
        embeddings=ALL_CHUNK_EMBEDDINGS,
        documents=[f.content for f in ALL_CHUNK_FIXTURES],
        metadatas=[
            # "source" key — matches DoclingChunker output from intrabot-ingestion
            {"source": f.document_filename, "chunk_index": f.chunk_index, "headings": ""}
            for f in ALL_CHUNK_FIXTURES
        ],
    )
    return store


@pytest.fixture
def empty_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(chroma_client=chromadb.Client())


class TestChromaVectorStore:

    def test_query_returns_most_similar_chunk_for_telework_query(
        self, populated_vector_store
    ) -> None:
        results = populated_vector_store.query_similar_chunks(QUERY_EMBEDDING_TELEWORK, 1)
        assert results[0].chunk_id == CHUNK_TELEWORK_POLICY.chunk_id

    def test_query_returns_most_similar_chunk_for_ci_cd_query(
        self, populated_vector_store
    ) -> None:
        results = populated_vector_store.query_similar_chunks(QUERY_EMBEDDING_CI_CD, 1)
        assert results[0].chunk_id == CHUNK_CI_CD_PIPELINE.chunk_id

    def test_document_filename_resolved_from_source_metadata_key(
        self, populated_vector_store
    ) -> None:
        """
        Verifies the metadata key mapping from intrabot-ingestion:
        ChromaDB stores 'source' (DoclingChunker) → resolved as document_filename.
        """
        results = populated_vector_store.query_similar_chunks(QUERY_EMBEDDING_TELEWORK, 1)
        assert results[0].document_filename == CHUNK_TELEWORK_POLICY.document_filename

    def test_query_returns_exactly_top_k_results(self, populated_vector_store) -> None:
        results = populated_vector_store.query_similar_chunks(QUERY_EMBEDDING_TELEWORK, 3)
        assert len(results) == 3

    def test_similarity_scores_are_in_valid_range(self, populated_vector_store) -> None:
        results = populated_vector_store.query_similar_chunks(QUERY_EMBEDDING_TELEWORK, 5)
        scores = np.array([c.similarity_score for c in results])
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_similarity_scores_are_ordered_descending(self, populated_vector_store) -> None:
        results = populated_vector_store.query_similar_chunks(QUERY_EMBEDDING_TELEWORK, 5)
        scores = [c.similarity_score for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_collection_returns_empty_list(self, empty_vector_store) -> None:
        """LSP: IVectorStore must return [] on empty collection, never raise."""
        assert empty_vector_store.query_similar_chunks(QUERY_EMBEDDING_TELEWORK, 5) == []

    def test_top_k_exceeding_collection_size_returns_all_available(
        self, populated_vector_store
    ) -> None:
        results = populated_vector_store.query_similar_chunks(
            QUERY_EMBEDDING_TELEWORK, len(ALL_CHUNK_FIXTURES) + 10
        )
        assert len(results) == len(ALL_CHUNK_FIXTURES)

    def test_unknown_topic_yields_lower_scores_than_known_topic(
        self, populated_vector_store
    ) -> None:
        """LSP / hallucination-prevention: score = 1 - cosine_distance is correct."""
        known = populated_vector_store.query_similar_chunks(QUERY_EMBEDDING_TELEWORK, 1)
        unknown = populated_vector_store.query_similar_chunks(QUERY_EMBEDDING_UNKNOWN_TOPIC, 1)
        assert unknown[0].similarity_score < known[0].similarity_score
