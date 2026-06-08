"""
Synthetic embedding fixtures for unit and integration tests.

All vectors are 1024-dimensional (matching mistral-embed) and L2-normalised
so that cosine similarity equals dot product.

Construction strategy
---------------------
* Five document embeddings are drawn independently from a seeded RNG and
  normalised — they are near-orthogonal by construction.
* Query embeddings for known topics are built by adding small Gaussian noise
  to the target document embedding and re-normalising, guaranteeing high cosine
  similarity with their target and lower similarity with all others.
* The unknown-topic query is drawn independently and will have low similarity
  with every document embedding — used to test hallucination-prevention paths.
"""

import numpy as np

EMBEDDING_DIMENSION: int = 1024
# With d=1024 dimensions, noise_scale=0.05 would give noise_magnitude≈1.6,
# collapsing cosine similarity to ~0.53.  0.005 gives magnitude≈0.16 → sim≈0.987.
_NOISE_SCALE: float = 0.005       # controls how close a query is to its target
_SEED: int = 42

_seeded_rng: np.random.Generator = np.random.default_rng(seed=_SEED)


def _normalised_random_vector() -> list[float]:
    """Draw a unit-norm random vector from the seeded generator."""
    raw_vector: np.ndarray = _seeded_rng.standard_normal(EMBEDDING_DIMENSION)
    return (raw_vector / np.linalg.norm(raw_vector)).tolist()


def _perturbed_normalised_vector(
    base_vector: list[float],
    noise_scale: float = _NOISE_SCALE,
) -> list[float]:
    """
    Return a new unit-norm vector close to base_vector.
    Used to produce query embeddings that are semantically near their target
    document without being identical.
    """
    base_array: np.ndarray = np.array(base_vector)
    noise: np.ndarray = _seeded_rng.standard_normal(EMBEDDING_DIMENSION) * noise_scale
    perturbed: np.ndarray = base_array + noise
    return (perturbed / np.linalg.norm(perturbed)).tolist()


# ── Document chunk embeddings (one per fixture document) ──────────────────────

CHUNK_EMBEDDING_TELEWORK_POLICY: list[float] = _normalised_random_vector()
CHUNK_EMBEDDING_CI_CD_PIPELINE: list[float] = _normalised_random_vector()
CHUNK_EMBEDDING_EXPENSE_REPORT: list[float] = _normalised_random_vector()
CHUNK_EMBEDDING_ONBOARDING_GUIDE: list[float] = _normalised_random_vector()
CHUNK_EMBEDDING_SECURITY_POLICY: list[float] = _normalised_random_vector()

ALL_CHUNK_EMBEDDINGS: list[list[float]] = [
    CHUNK_EMBEDDING_TELEWORK_POLICY,
    CHUNK_EMBEDDING_CI_CD_PIPELINE,
    CHUNK_EMBEDDING_EXPENSE_REPORT,
    CHUNK_EMBEDDING_ONBOARDING_GUIDE,
    CHUNK_EMBEDDING_SECURITY_POLICY,
]

# ── Query embeddings ──────────────────────────────────────────────────────────

# Close to CHUNK_EMBEDDING_TELEWORK_POLICY — should retrieve that chunk first
QUERY_EMBEDDING_TELEWORK: list[float] = _perturbed_normalised_vector(
    CHUNK_EMBEDDING_TELEWORK_POLICY
)

# Close to CHUNK_EMBEDDING_CI_CD_PIPELINE — should retrieve that chunk first
QUERY_EMBEDDING_CI_CD: list[float] = _perturbed_normalised_vector(
    CHUNK_EMBEDDING_CI_CD_PIPELINE
)

# Independent from all documents — simulates a question with no matching source
# Used to test that the service does NOT hallucinate an answer
QUERY_EMBEDDING_UNKNOWN_TOPIC: list[float] = _normalised_random_vector()
