"""Triplet sampling utilities for contrastive profile verification."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
import random
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch.utils.data import Dataset

from identity_verification.datasets.profile_examples import (
    MIXED_NEGATIVE_WEIGHTS,
    MIXED_REGIME,
    RANDOM_ONLY_REGIME,
    AuthorHardnessCandidates,
    HardnessCandidate,
    ProfileRecord,
    build_hardness_index,
    load_profile_records,
)
from identity_verification.schemas import HardnessLabel
from identity_verification.training.profile_dataset import truncate_text


TEXT_SOURCE_CHOICES = ("all", "profile", "query")


class TextTokenizer(Protocol):
    """Minimal tokenizer protocol used by the triplet collator."""

    def __call__(
        self,
        text: Sequence[str],
        *,
        padding: str | bool,
        truncation: bool | str,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        """Tokenize raw texts into tensors."""


@dataclass(frozen=True)
class ProfileTextPool:
    """Author-level text pool for contrastive sampling and evaluation."""

    author_id: str
    profile_texts: tuple[str, ...]
    query_texts: tuple[str, ...]

    @property
    def all_texts(self) -> tuple[str, ...]:
        return (*self.profile_texts, *self.query_texts)


@dataclass(frozen=True)
class ContrastiveTriplet:
    """One anchor/positive/negative training example."""

    anchor_text: str
    positive_text: str
    negative_text: str
    anchor_author_id: str
    positive_author_id: str
    negative_author_id: str
    negative_hardness: str
    negative_similarity_score: float | None = None


@dataclass(frozen=True)
class TripletBatch:
    """Tokenized triplet batch plus metadata."""

    anchor: dict[str, torch.Tensor]
    positive: dict[str, torch.Tensor]
    negative: dict[str, torch.Tensor]
    negative_hardness: list[str]
    metadata: list[dict[str, Any]]


class ContrastiveTripletDataset(Dataset[ContrastiveTriplet]):
    """Small in-memory dataset for sampled contrastive triplets."""

    def __init__(self, triplets: Sequence[ContrastiveTriplet]) -> None:
        if not triplets:
            raise ValueError("ContrastiveTripletDataset requires at least one triplet")
        self.triplets = list(triplets)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> ContrastiveTriplet:
        return self.triplets[index]


@dataclass(frozen=True)
class TripletCollator:
    """Tokenize anchor/positive/negative texts independently."""

    tokenizer: TextTokenizer
    max_length: int = 256
    max_chars_per_text: int = 2000
    padding: str | bool = "longest"
    truncation: bool | str = True

    def __call__(self, triplets: Sequence[ContrastiveTriplet]) -> TripletBatch:
        if not triplets:
            raise ValueError("Cannot collate an empty triplet batch")

        anchors = [self._prepare_text(triplet.anchor_text) for triplet in triplets]
        positives = [self._prepare_text(triplet.positive_text) for triplet in triplets]
        negatives = [self._prepare_text(triplet.negative_text) for triplet in triplets]

        return TripletBatch(
            anchor=self._tokenize(anchors),
            positive=self._tokenize(positives),
            negative=self._tokenize(negatives),
            negative_hardness=[triplet.negative_hardness for triplet in triplets],
            metadata=[
                {
                    "anchor_author_id": triplet.anchor_author_id,
                    "positive_author_id": triplet.positive_author_id,
                    "negative_author_id": triplet.negative_author_id,
                    "negative_similarity_score": triplet.negative_similarity_score,
                }
                for triplet in triplets
            ],
        )

    def _prepare_text(self, text: str) -> str:
        clean = " ".join(text.split())
        return truncate_text(clean, self.max_chars_per_text, keep="head")

    def _tokenize(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        tokenized = self.tokenizer(
            list(texts),
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return dict(tokenized)


def load_profile_text_pools(path: str | Path) -> dict[str, ProfileTextPool]:
    """Load author text pools from a generated profile split JSONL."""
    records = load_profile_records(path)
    pools = {
        record.author_id: ProfileTextPool(
            author_id=record.author_id,
            profile_texts=record.profile_texts,
            query_texts=record.query_texts,
        )
        for record in records
    }
    if len(pools) != len(records):
        raise ValueError(f"Duplicate author IDs found in profile split: {path}")
    return pools


def sample_contrastive_triplets(
    profiles: Sequence[ProfileRecord],
    *,
    regime: str,
    triplets_per_author: int,
    random_seed: int,
    text_source: str = "all",
    hardness_index: Mapping[str, AuthorHardnessCandidates] | None = None,
) -> list[ContrastiveTriplet]:
    """Sample anchor/positive/negative triplets from profile records."""
    if triplets_per_author <= 0:
        raise ValueError("triplets_per_author must be positive")
    _validate_text_source(text_source)

    sorted_profiles = sorted(profiles, key=lambda profile: profile.author_id)
    if len(sorted_profiles) < 2:
        raise ValueError("Need at least two authors to sample contrastive triplets")

    pools = {
        profile.author_id: _texts_for_source(profile, text_source)
        for profile in sorted_profiles
    }
    valid_author_ids = [
        author_id
        for author_id, texts in pools.items()
        if len(texts) >= 2
    ]
    if len(valid_author_ids) < 2:
        raise ValueError("Need at least two authors with two or more texts")

    rng = random.Random(random_seed)
    negative_hardnesses = negative_hardness_sequence(
        total=len(valid_author_ids) * triplets_per_author,
        regime=regime,
        rng=rng,
    )
    if hardness_index is None and any(
        hardness != HardnessLabel.RANDOM for hardness in negative_hardnesses
    ):
        hardness_index = build_hardness_index(sorted_profiles)

    triplets: list[ContrastiveTriplet] = []
    hardness_iterator = iter(negative_hardnesses)
    for author_id in valid_author_ids:
        own_texts = pools[author_id]
        for _ in range(triplets_per_author):
            anchor_text, positive_text = rng.sample(own_texts, 2)
            negative_hardness = next(hardness_iterator)
            negative_candidate = sample_negative_candidate(
                author_id,
                negative_hardness,
                valid_author_ids=valid_author_ids,
                hardness_index=hardness_index,
                rng=rng,
            )
            negative_text = rng.choice(pools[negative_candidate.author_id])
            triplets.append(
                ContrastiveTriplet(
                    anchor_text=anchor_text,
                    positive_text=positive_text,
                    negative_text=negative_text,
                    anchor_author_id=author_id,
                    positive_author_id=author_id,
                    negative_author_id=negative_candidate.author_id,
                    negative_hardness=negative_hardness.value,
                    negative_similarity_score=negative_candidate.similarity_score,
                )
            )

    rng.shuffle(triplets)
    return triplets


def negative_hardness_sequence(
    *,
    total: int,
    regime: str,
    rng: random.Random,
) -> list[HardnessLabel]:
    """Return an exact per-epoch hardness schedule for triplet negatives."""
    if total <= 0:
        return []
    if regime == RANDOM_ONLY_REGIME or regime == HardnessLabel.RANDOM.value:
        return [HardnessLabel.RANDOM] * total
    if regime == HardnessLabel.SEMI_HARD.value:
        return [HardnessLabel.SEMI_HARD] * total
    if regime == HardnessLabel.HARD.value:
        return [HardnessLabel.HARD] * total
    if regime != MIXED_REGIME:
        raise ValueError(f"Unsupported contrastive regime: {regime}")

    counts = allocate_weighted_counts(total, MIXED_NEGATIVE_WEIGHTS)
    hardnesses: list[HardnessLabel] = []
    for hardness in (HardnessLabel.RANDOM, HardnessLabel.SEMI_HARD, HardnessLabel.HARD):
        hardnesses.extend([hardness] * counts[hardness])
    rng.shuffle(hardnesses)
    return hardnesses


def allocate_weighted_counts(
    total: int,
    weights: Mapping[HardnessLabel, float],
) -> dict[HardnessLabel, int]:
    """Allocate integer counts while preserving the requested total."""
    raw_counts = {hardness: total * weight for hardness, weight in weights.items()}
    counts = {hardness: floor(raw_count) for hardness, raw_count in raw_counts.items()}
    remainder = total - sum(counts.values())
    ranked_remainders = sorted(
        weights,
        key=lambda hardness: (raw_counts[hardness] - counts[hardness], hardness.value),
        reverse=True,
    )
    for hardness in ranked_remainders[:remainder]:
        counts[hardness] += 1
    return counts


def sample_negative_hardness(regime: str, *, rng: random.Random) -> HardnessLabel:
    """Sample the desired negative hardness for a training triplet."""
    if regime == RANDOM_ONLY_REGIME or regime == HardnessLabel.RANDOM.value:
        return HardnessLabel.RANDOM
    if regime == HardnessLabel.SEMI_HARD.value:
        return HardnessLabel.SEMI_HARD
    if regime == HardnessLabel.HARD.value:
        return HardnessLabel.HARD
    if regime != MIXED_REGIME:
        raise ValueError(f"Unsupported contrastive regime: {regime}")

    draw = rng.random()
    cumulative = 0.0
    for hardness in (HardnessLabel.RANDOM, HardnessLabel.SEMI_HARD, HardnessLabel.HARD):
        cumulative += MIXED_NEGATIVE_WEIGHTS[hardness]
        if draw <= cumulative:
            return hardness
    return HardnessLabel.HARD


def sample_negative_candidate(
    author_id: str,
    hardness: HardnessLabel,
    *,
    valid_author_ids: Sequence[str],
    hardness_index: Mapping[str, AuthorHardnessCandidates] | None,
    rng: random.Random,
) -> HardnessCandidate:
    """Sample a negative author, falling back to random when needed."""
    valid_author_set = set(valid_author_ids)
    if hardness == HardnessLabel.RANDOM or hardness_index is None:
        other_authors = [other for other in valid_author_ids if other != author_id]
        if not other_authors:
            raise ValueError(f"No negative authors available for author_id={author_id}")
        return HardnessCandidate(
            author_id=rng.choice(other_authors),
            similarity_score=None,
        )

    buckets = hardness_index.get(author_id)
    candidates = [
        candidate
        for candidate in (buckets.get(hardness) if buckets is not None else ())
        if candidate.author_id in valid_author_set and candidate.author_id != author_id
    ]
    if not candidates:
        return sample_negative_candidate(
            author_id,
            HardnessLabel.RANDOM,
            valid_author_ids=valid_author_ids,
            hardness_index=None,
            rng=rng,
        )
    return rng.choice(candidates)


def _texts_for_source(profile: ProfileRecord, text_source: str) -> tuple[str, ...]:
    _validate_text_source(text_source)
    if text_source == "profile":
        texts = profile.profile_texts
    elif text_source == "query":
        texts = profile.query_texts
    else:
        texts = (*profile.profile_texts, *profile.query_texts)
    return tuple(text for text in texts if text and text.strip())


def _validate_text_source(text_source: str) -> None:
    if text_source not in TEXT_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported text_source={text_source!r}. "
            f"Valid choices: {TEXT_SOURCE_CHOICES}"
        )
