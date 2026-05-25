"""Shared schemas for verification datasets and experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SplitName(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class HardnessLabel(str, Enum):
    POSITIVE = "positive"
    RANDOM = "random"
    SEMI_HARD = "semi_hard"
    HARD = "hard"


class ProtocolName(str, Enum):
    PAN_PROFILE = "pan_profile"
    WILDCHAT_PROFILE = "wildchat_profile"


@dataclass(frozen=True)
class TextRecord:
    record_id: str
    author_id: str
    text: str
    dataset: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class PairExample:
    example_id: str
    reference_text: str
    query_text: str
    label: int
    split: SplitName
    protocol: ProtocolName
    hardness: HardnessLabel
    reference_author_id: str | None = None
    query_author_id: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProfileQueryExample:
    example_id: str
    profile_text: str
    query_texts: tuple[str, ...]
    label: int
    split: SplitName
    protocol: ProtocolName
    hardness: HardnessLabel
    profile_author_id: str
    query_author_id: str
    metadata: dict[str, Any] | None = None

    @property
    def query_text(self) -> str:
        """Return the full query stream as a single text block."""
        return "\n".join(self.query_texts)


@dataclass(frozen=True)
class PredictionRecord:
    example_id: str
    label: int
    score: float
    split: SplitName
    protocol: ProtocolName
    hardness: HardnessLabel | None = None
    threshold: float | None = None
    prediction: int | None = None
    metadata: dict[str, Any] | None = None
