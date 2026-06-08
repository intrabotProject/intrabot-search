import pytest

from app.application.prompt.rag_prompt_builder import (
    NO_CONTEXT_DOCUMENTS_PLACEHOLDER,
    SYSTEM_INSTRUCTION,
    RAGPromptBuilder,
)
from app.domain.models.retrieved_chunk import RetrievedChunk
from tests.fixtures.documents import CHUNK_TELEWORK_POLICY, CHUNK_CI_CD_PIPELINE


def _make_retrieved_chunk(fixture, score: float = 0.90) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=fixture.chunk_id,
        document_filename=fixture.document_filename,
        content=fixture.content,
        similarity_score=score,
        chunk_index=fixture.chunk_index,
    )


class TestRAGPromptBuilder:

    def setup_method(self) -> None:
        self._builder = RAGPromptBuilder()

    def test_prompt_contains_system_instruction(self) -> None:
        chunk = _make_retrieved_chunk(CHUNK_TELEWORK_POLICY)
        prompt = self._builder.build_rag_prompt("Question ?", [chunk])
        assert SYSTEM_INSTRUCTION.strip() in prompt

    def test_prompt_contains_user_question(self) -> None:
        chunk = _make_retrieved_chunk(CHUNK_TELEWORK_POLICY)
        user_question = "Combien de jours de télétravail sont autorisés ?"
        prompt = self._builder.build_rag_prompt(user_question, [chunk])
        assert user_question in prompt

    def test_prompt_contains_chunk_source_filename(self) -> None:
        chunk = _make_retrieved_chunk(CHUNK_TELEWORK_POLICY)
        prompt = self._builder.build_rag_prompt("Question ?", [chunk])
        assert CHUNK_TELEWORK_POLICY.document_filename in prompt

    def test_prompt_contains_chunk_content(self) -> None:
        chunk = _make_retrieved_chunk(CHUNK_TELEWORK_POLICY)
        prompt = self._builder.build_rag_prompt("Question ?", [chunk])
        assert CHUNK_TELEWORK_POLICY.content in prompt

    def test_prompt_with_multiple_chunks_contains_all_filenames(self) -> None:
        chunks = [
            _make_retrieved_chunk(CHUNK_TELEWORK_POLICY, score=0.92),
            _make_retrieved_chunk(CHUNK_CI_CD_PIPELINE, score=0.75),
        ]
        prompt = self._builder.build_rag_prompt("Question ?", chunks)
        assert CHUNK_TELEWORK_POLICY.document_filename in prompt
        assert CHUNK_CI_CD_PIPELINE.document_filename in prompt

    def test_prompt_preserves_chunk_order(self) -> None:
        chunks = [
            _make_retrieved_chunk(CHUNK_TELEWORK_POLICY),
            _make_retrieved_chunk(CHUNK_CI_CD_PIPELINE),
        ]
        prompt = self._builder.build_rag_prompt("Question ?", chunks)
        assert prompt.index(CHUNK_TELEWORK_POLICY.document_filename) < \
               prompt.index(CHUNK_CI_CD_PIPELINE.document_filename)

    def test_prompt_with_empty_chunks_still_contains_system_instruction(self) -> None:
        """
        LSP edge case: IPromptBuilder contract requires the system instruction
        to be present even when retrieved_chunks is empty.
        """
        prompt = self._builder.build_rag_prompt("Question sans source ?", [])
        assert SYSTEM_INSTRUCTION.strip() in prompt

    def test_prompt_with_empty_chunks_contains_no_context_placeholder(self) -> None:
        prompt = self._builder.build_rag_prompt("Question ?", [])
        assert NO_CONTEXT_DOCUMENTS_PLACEHOLDER in prompt

    def test_prompt_with_empty_chunks_does_not_contain_source_block(self) -> None:
        prompt = self._builder.build_rag_prompt("Question ?", [])
        assert "DOCUMENTS:\n[Source:" not in prompt
