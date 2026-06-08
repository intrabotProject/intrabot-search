from unittest.mock import MagicMock, patch

from app.infrastructure.llm.cohere_llm_provider import (
    COHERE_GENERATION_MODEL,
    CohereLLMProvider,
)


def _build_mock_cohere_response(answer_text: str) -> MagicMock:
    """Builds the shape returned by cohere.ClientV2.chat()."""
    mock_content_item = MagicMock()
    mock_content_item.text = answer_text
    mock_message = MagicMock()
    mock_message.content = [mock_content_item]
    mock_response = MagicMock()
    mock_response.message = mock_message
    return mock_response


class TestCohereLLMProvider:
    """
    Battery of tests for CohereLLMProvider.
    cohere.ClientV2 is always mocked — no real API calls are made.
    """

    def setup_method(self) -> None:
        with patch("app.infrastructure.llm.cohere_llm_provider.cohere.ClientV2") as mock_cls:
            self._mock_client = MagicMock()
            mock_cls.return_value = self._mock_client
            self._provider = CohereLLMProvider(api_key="test-api-key")

    def test_generate_answer_calls_api_with_correct_model(self) -> None:
        self._mock_client.chat.return_value = _build_mock_cohere_response("Réponse.")
        self._provider.generate_answer("Prompt test")
        call_kwargs = self._mock_client.chat.call_args[1]
        assert call_kwargs["model"] == COHERE_GENERATION_MODEL

    def test_generate_answer_sends_prompt_as_user_message(self) -> None:
        self._mock_client.chat.return_value = _build_mock_cohere_response("Réponse.")
        self._provider.generate_answer("Mon prompt augmenté")
        call_kwargs = self._mock_client.chat.call_args[1]
        assert call_kwargs["messages"] == [
            {"role": "user", "content": "Mon prompt augmenté"}
        ]

    def test_generate_answer_returns_llm_text_content(self) -> None:
        expected = "Le télétravail est autorisé 3 jours par semaine."
        self._mock_client.chat.return_value = _build_mock_cohere_response(expected)
        assert self._provider.generate_answer("Question") == expected

    def test_generate_answer_returns_empty_string_when_content_is_none(self) -> None:
        """LSP: provider must return str, never None."""
        self._mock_client.chat.return_value = _build_mock_cohere_response(None)
        result = self._provider.generate_answer("Question")
        assert result == ""
        assert isinstance(result, str)

    def test_generate_answer_with_no_context_prompt_still_calls_api(self) -> None:
        """
        LSP — hallucination guardrail responsibility:
        The hallucination guardrail lives in RAGPromptBuilder, not here.
        The LLM provider must NEVER short-circuit on an empty-context prompt.
        """
        self._mock_client.chat.return_value = _build_mock_cohere_response(
            "I cannot find the answer to this question in the available documents."
        )
        result = self._provider.generate_answer("")
        self._mock_client.chat.assert_called_once()
        assert isinstance(result, str)
