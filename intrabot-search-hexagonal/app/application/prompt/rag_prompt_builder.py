from app.domain.interfaces.secondary.prompt.prompt_builder import IPromptBuilder
from app.domain.models.retrieved_chunk import RetrievedChunk

SYSTEM_INSTRUCTION: str = (
    "Tu es un assistant intranet d'entreprise spécialisé dans la recherche documentaire.\n"
    "Ta mission : répondre aux questions des utilisateurs en exploitant le contenu des documents fournis.\n\n"
    "Comment répondre :\n"
    "- Lis attentivement tous les documents fournis et construis ta réponse à partir de leur contenu.\n"
    "- Pour les questions définitionnelles ('c'est quoi X ?', 'qu'est-ce que Y ?') : "
    "décris X ou Y à partir de ce que les documents révèlent sur le sujet "
    "(fonctionnalités, exemples, usages, caractéristiques mentionnées).\n"
    "- Tu peux synthétiser, regrouper et reformuler — tu n'as pas besoin d'une définition "
    "explicite dans le texte pour répondre.\n"
    "- Ne pas inventer de faits absents des documents.\n"
    "- Cite le fichier source entre parenthèses pour chaque point clé.\n"
    "- Réponds toujours en français.\n"
    "- Style : réponse courte et conversationnelle. Évite les listes à puces et le gras excessifs. "
    "Préfère 2-4 phrases directes plutôt qu'une énumération exhaustive. "
    "N'utilise une liste que si la question appelle explicitement une comparaison ou une énumération.\n\n"
    "Uniquement si les documents ne contiennent aucune information en rapport avec la question, "
    'réponds : "Je ne trouve pas la réponse à cette question dans les documents disponibles."\n'
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
        return f"{documents_block}\n\nQUESTION: {user_question}"

    def _format_chunks_as_context_block(
        self,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        formatted_chunks: list[str] = [
            f"[Source: {chunk.document_filename} — chunk {chunk.chunk_index}]\n{chunk.content}"
            for chunk in retrieved_chunks
        ]
        return "DOCUMENTS :\n" + "\n\n".join(formatted_chunks)
