from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """
    A document chunk exposed in the API response, carrying provenance
    information so the client can display citations.
    """

    chunk_id: str
    filename: str
    excerpt: str
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine similarity between the query and this chunk (∈ [0, 1]).",
    )


class SearchResponse(BaseModel):
    """
    Full API response returned by POST /search, containing the generated
    answer, the list of source chunks used, and the end-to-end latency.
    """

    answer: str
    sources: list[SourceChunk]
    latency_ms: int = Field(..., ge=0, description="End-to-end processing time in milliseconds.")
