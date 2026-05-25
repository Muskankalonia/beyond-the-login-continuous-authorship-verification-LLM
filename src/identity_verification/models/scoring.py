"""Scoring helpers shared by benchmark runners."""

from __future__ import annotations

from typing import Any, Sequence

from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.metrics.pairwise import paired_cosine_distances

from identity_verification.models.encoders import TextEncoder


def cosine_similarity(left: Any, right: Any) -> float:
    """Return cosine similarity for dense or sparse vector rows."""
    return float(sklearn_cosine_similarity(left, right)[0, 0])


def score_text_pairs(
    encoder: TextEncoder,
    reference_texts: Sequence[str],
    query_texts: Sequence[str],
) -> list[float]:
    """Encode and cosine-score aligned reference/query text pairs."""
    if len(reference_texts) != len(query_texts):
        raise ValueError(
            "reference_texts and query_texts must have the same length: "
            f"{len(reference_texts)} != {len(query_texts)}"
        )
    if not reference_texts:
        return []

    vectors = encoder.encode([*reference_texts, *query_texts])
    reference_vectors = vectors[: len(reference_texts)]
    query_vectors = vectors[len(reference_texts) :]
    distances = paired_cosine_distances(reference_vectors, query_vectors)
    return [float(1.0 - distance) for distance in distances]
