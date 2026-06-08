import pytest
from pydantic import ValidationError

from app.domain.models.search_request import SearchRequest


class TestSearchRequest:
    """Battery of tests for the SearchRequest value object."""

    def test_valid_request_with_defaults_is_accepted(self) -> None:
        request = SearchRequest(question="Quelle est la politique de télétravail ?")
        assert request.question == "Quelle est la politique de télétravail ?"
        assert request.top_k == 5

    def test_custom_top_k_is_stored(self) -> None:
        request = SearchRequest(question="Ma question", top_k=3)
        assert request.top_k == 3

    def test_question_surrounding_whitespace_is_stripped(self) -> None:
        request = SearchRequest(question="  Ma question  ")
        assert request.question == "Ma question"

    # ── Rejection cases ───────────────────────────────────────────────────────

    def test_empty_question_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(question="")

    def test_whitespace_only_question_raises_validation_error(self) -> None:
        """
        LSP edge case: a string of spaces satisfies min_length=1 at the raw
        level but must be rejected after stripping.
        """
        with pytest.raises(ValidationError):
            SearchRequest(question="     ")

    def test_question_exceeding_max_length_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(question="a" * 2001)

    def test_top_k_of_zero_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(question="Valid question", top_k=0)

    def test_top_k_above_maximum_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(question="Valid question", top_k=21)

    def test_top_k_at_minimum_boundary_is_accepted(self) -> None:
        request = SearchRequest(question="Question", top_k=1)
        assert request.top_k == 1

    def test_top_k_at_maximum_boundary_is_accepted(self) -> None:
        request = SearchRequest(question="Question", top_k=20)
        assert request.top_k == 20
