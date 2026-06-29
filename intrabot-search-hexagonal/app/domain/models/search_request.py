from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """
    Validated input for the POST /search endpoint.
    Pydantic enforces types and constraints at deserialisation time.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question asked by the user.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve from the vector store.",
    )
    source_filter: str | None = Field(
        default=None,
        description="If set, restrict retrieval to chunks from this document filename.",
    )
    min_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for a chunk to be used.",
    )
    allowed_categories: list[str] | None = Field(
        default=None,
        description="If set, restrict retrieval to chunks in these access categories.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, raw_question: str) -> str:
        """Reject questions that are whitespace-only after stripping."""
        stripped_question = raw_question.strip()
        if not stripped_question:
            raise ValueError("question must not be blank or whitespace-only")
        return stripped_question
