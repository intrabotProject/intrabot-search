from abc import ABC, abstractmethod


class ILLMProvider(ABC):
    """
    Contract for a language model that generates a text answer from an
    augmented prompt.  Concrete implementations (MistralLLMProvider,
    OpenAILLMProvider, GeminiLLMProvider …) must honour this contract.

    NOTE — The hallucination guardrail is the responsibility of IPromptBuilder,
    NOT of ILLMProvider. The provider always forwards the prompt as-is to the
    underlying model and returns the raw text response.
    """

    @abstractmethod
    def generate_answer(self, augmented_prompt: str) -> str:
        """
        Send augmented_prompt to the language model and return the raw
        generated text.  Must never return None; return an empty string
        if the model produces no output.
        """
