"""Composition root — câblage des adaptateurs du service search."""

import chromadb

from app.application.prompt.rag_prompt_builder import RAGPromptBuilder
from app.application.search.rag_search_service import RAGSearchService
from app.core.config import get_settings
from app.domain.interfaces.primary.search.search_service import ISearchService
from app.infrastructure.embedding.ingestion_service_embedding_adapter import (
    IngestionServiceEmbeddingAdapter,
)
from app.infrastructure.llm.cohere_llm_provider import CohereLLMProvider
from app.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore


def _get_chroma_client() -> chromadb.ClientAPI:
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_path)


def get_search_service() -> ISearchService:
    """
    Assemble le pipeline RAG :
      embed (via ingestion) → ChromaDB → prompt → LLM Cohere
    """
    settings = get_settings()
    return RAGSearchService(
        embedding_provider=IngestionServiceEmbeddingAdapter(
            ingestion_service_url=settings.ingestion_service_url,
        ),
        vector_store=ChromaVectorStore(
            chroma_client=_get_chroma_client(),
            collection_name=settings.chroma_collection_name,
        ),
        llm_provider=CohereLLMProvider(api_key=settings.cohere_api_key),
        prompt_builder=RAGPromptBuilder(),
    )
