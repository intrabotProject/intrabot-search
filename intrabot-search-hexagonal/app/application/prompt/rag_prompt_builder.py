from app.domain.interfaces.secondary.prompt.prompt_builder import IPromptBuilder
from app.domain.models.retrieved_chunk import RetrievedChunk

SYSTEM_INSTRUCTION: str = (
    "You are an intranet assistant.\n"
    "Answer the question ONLY based on the documents provided below.\n"
    "If the answer cannot be found in the documents, reply exactly:\n"
    '  "I cannot find the answer to this question in the available documents."\n'
    "Answer in the same language as the question.\n"
    "Be concise and cite the source filename for every claim.\n"
)

NO_CONTEXT_DOCUMENTS_PLACEHOLDER: str = "DOCUMENTS: (none — no relevant document was retrieved)"


class RAGPromptBuilder(IPromptBuilder):
    """
    Secondary adapter (application-side) that implements the IPromptBuilder
    driven port. Embeds the hallucination guardrail: when no chunks are
    available the placeholder instructs the model it has no grounding material.
    """

    def build_rag_prompt(
        self,
        user_question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        documents_block = (
            self._format_chunks_as_context_block(retrieved_chunks)
            if retrieved_chunks
            else NO_CONTEXT_DOCUMENTS_PLACEHOLDER
        )
        return f"{SYSTEM_INSTRUCTION}\n{documents_block}\n\nQUESTION: {user_question}"

    def _format_chunks_as_context_block(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        formatted_chunks: list[str] = [
            f"[Source: {chunk.document_filename} — chunk {chunk.chunk_index}]\n{chunk.content}"
            for chunk in retrieved_chunks
        ]
        return "DOCUMENTS:\n" + "\n\n".join(formatted_chunks)
