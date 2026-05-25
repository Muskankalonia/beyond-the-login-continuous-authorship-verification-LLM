"""Dataset-agnostic encoders and scoring helpers."""

from identity_verification.models.encoders import TextEncoder, TfidfEncoder
from identity_verification.models.scoring import cosine_similarity, score_text_pairs

__all__ = [
    "TextEncoder",
    "TfidfEncoder",
    "cosine_similarity",
    "score_text_pairs",
]
