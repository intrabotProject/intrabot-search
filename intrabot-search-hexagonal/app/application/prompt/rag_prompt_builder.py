from app.domain.interfaces.secondary.prompt.prompt_builder import IPromptBuilder
from app.domain.models.retrieved_chunk import RetrievedChunk

SYSTEM_INSTRUCTION: str = (
    "Tu es un assistant intranet d'entreprise.\n"
    "Réponds UNIQUEMENT à partir des documents fournis ci-dessous.\n"
    "Si la réponse ne figure pas dans les documents, réponds exactement :\n"
    '  "Je ne trouve pas la réponse à cette question dans les documents disponibles."\n'
    "Réponds toujours en français.\n"
    "Sois concis et cite le nom du fichier source pour chaque affirmation.\n"
)

NO_CONTEXT_DOCUMENTS_PLACEHOLDER: str = (
    "DOCUMENTS : (aucun — aucun passage pertinent n'a été retrouvé)"
)


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
        return "DOCUMENTS :\n" + "\n\n".join(formatted_chunks)
