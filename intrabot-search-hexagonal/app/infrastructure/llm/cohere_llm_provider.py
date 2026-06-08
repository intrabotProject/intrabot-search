import cohere

from app.domain.interfaces.secondary.llm.llm_provider import ILLMProvider

# command-r-plus is Cohere's RAG-optimised generation model
COHERE_GENERATION_MODEL: str = "command-r-plus-08-2024"


class CohereLLMProvider(ILLMProvider):
    """
    Secondary adapter — implements the ILLMProvider driven port
    using the Cohere chat completion API.

    command-r-plus is specifically trained for RAG use cases:
    it follows grounding instructions reliably and tends to refuse
    answering outside the provided context, reinforcing the
    hallucination guardrail embedded in RAGPromptBuilder.
    """

    def __init__(self, api_key: str) -> None:
        self._client = cohere.ClientV2(api_key=api_key)

    def generate_answer(self, augmented_prompt: str) -> str:
        response = self._client.chat(
            model=COHERE_GENERATION_MODEL,
            messages=[{"role": "user", "content": augmented_prompt}],
        )
        generated_text: str = response.message.content[0].text
        return generated_text if generated_text is not None else ""
