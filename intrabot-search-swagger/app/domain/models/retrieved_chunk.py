from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """
    Immutable value object representing a document chunk retrieved from the
    vector store, along with its cosine similarity score to the query.
    frozen=True enforces immutability: once created, no field can be reassigned.
    """

    chunk_id: str
    document_filename: str
    content: str
    similarity_score: float  # cosine similarity ∈ [0.0, 1.0]; 1.0 = identical
    chunk_index: int          # position of this chunk inside its parent document
