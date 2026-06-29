import time

from app.domain.interfaces.primary.search.search_service import ISearchService
from app.domain.interfaces.secondary.embedding.embedding_provider import IEmbeddingProvider
from app.domain.interfaces.secondary.llm.llm_provider import ILLMProvider
from app.domain.interfaces.secondary.prompt.prompt_builder import IPromptBuilder
from app.domain.interfaces.secondary.vector_store.vector_store import IVectorStore
from app.domain.models.retrieved_chunk import RetrievedChunk
from app.domain.models.search_request import SearchRequest
from app.domain.models.search_response import SearchResponse, SourceChunk


class RAGSearchService(ISearchService):
    """
    Concrete implementation of the ISearchService primary port.

    Orchestrates the four steps of a RAG query:
      1. Embed the user question via IEmbeddingProvider (secondary port).
      2. Retrieve similar chunks via IVectorStore (secondary port).
      3. Build an augmented prompt via IPromptBuilder (secondary port).
      4. Generate the answer via ILLMProvider (secondary port).

    All secondary-port dependencies are injected as interfaces — swapping
    any provider requires zero modification to this class (OCP / Strategy).
    """

    def __init__(
        self,
        embedding_provider: IEmbeddingProvider,
        vector_store: IVectorStore,
        llm_provider: ILLMProvider,
        prompt_builder: IPromptBuilder,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._llm_provider = llm_provider
        self._prompt_builder = prompt_builder

    def search(self, request: SearchRequest) -> SearchResponse:
        start_timestamp_ms: float = time.time() * 1000

        query_embedding: list[float] = self._embedding_provider.embed_text(request.question)

        if request.allowed_categories is not None and not request.allowed_categories:
            all_chunks: list[RetrievedChunk] = []
        else:
            all_chunks = self._vector_store.query_similar_chunks(
                query_embedding=query_embedding,
                top_k=request.top_k,
                source_filter=request.source_filter,
                allowed_categories=request.allowed_categories,
            )

        retrieved_chunks = [
            chunk
            for chunk in all_chunks
            if chunk.similarity_score >= request.min_score
        ]
        excluded_chunks = [
            chunk
            for chunk in all_chunks
            if chunk.similarity_score < request.min_score
        ]

        augmented_prompt: str = self._prompt_builder.build_rag_prompt(
            user_question=request.question,
            retrieved_chunks=retrieved_chunks,
        )

        generated_answer: str = self._llm_provider.generate_answer(augmented_prompt)

        elapsed_ms: int = int(time.time() * 1000 - start_timestamp_ms)

        return SearchResponse(
            answer=generated_answer,
            sources=self._map_retrieved_chunks_to_source_chunks(retrieved_chunks),
            excluded_by_threshold=self._map_retrieved_chunks_to_source_chunks(
                excluded_chunks
            ),
            latency_ms=elapsed_ms,
        )

    def _map_retrieved_chunks_to_source_chunks(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[SourceChunk]:
        return [
            SourceChunk(
                chunk_id=chunk.chunk_id,
                filename=chunk.document_filename,
                excerpt=chunk.content,
                similarity_score=chunk.similarity_score,
            )
            for chunk in retrieved_chunks
        ]
