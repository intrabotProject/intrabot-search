from abc import ABC, abstractmethod

from app.domain.models.retrieved_chunk import RetrievedChunk


class IPromptBuilder(ABC):
    """
    Contract for assembling the augmented prompt fed to the LLM.
    The implementation is responsible for the hallucination guardrail:
    when retrieved_chunks is empty it MUST instruct the model not to
    answer from parametric knowledge.
    """

    @abstractmethod
    def build_rag_prompt(
        self,
        user_question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        """
        Build and return the complete prompt string.

        Args:
            user_question:    The original question posed by the user.
            retrieved_chunks: Ordered list of chunks from the vector store.
                              An empty list means no relevant document was found.
        Returns:
            A complete prompt string ready to be sent to an ILLMProvider.
        """
