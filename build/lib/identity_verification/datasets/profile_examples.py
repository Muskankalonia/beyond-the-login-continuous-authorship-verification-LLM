"""Build profile/query examples with random and hard negative sampling."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
from math import ceil, floor
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer

from identity_verification.models.encoders import TfidfEncoder
from identity_verification.schemas import HardnessLabel, SplitName


RANDOM_ONLY_REGIME = "random_only"
MIXED_REGIME = "mixed"
PURE_NEGATIVE_REGIMES = (
    HardnessLabel.RANDOM.value,
    HardnessLabel.SEMI_HARD.value,
    HardnessLabel.HARD.value,
)
TRAIN_VAL_REGIMES = (RANDOM_ONLY_REGIME, MIXED_REGIME)
TEST_REGIMES = (
    HardnessLabel.RANDOM.value,
    HardnessLabel.SEMI_HARD.value,
    HardnessLabel.HARD.value,
    MIXED_REGIME,
)
MIXED_NEGATIVE_WEIGHTS = {
    HardnessLabel.RANDOM: 0.50,
    HardnessLabel.SEMI_HARD: 0.30,
    HardnessLabel.HARD: 0.20,
}
HARDNESS_INDEX_CACHE_FORMAT = "identity_verification.hardness_index.jsonl.v1"
TFIDF_HARDNESS_MINER = "tfidf"
SBERT_HARDNESS_MINER = "sbert"
DEFAULT_SBERT_HARDNESS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HARDNESS_MINERS = (TFIDF_HARDNESS_MINER, SBERT_HARDNESS_MINER)


@dataclass(frozen=True)
class ProfileRecord:
    """Canonical profile/query record used by profile-based benchmarks."""

    author_id: str
    dataset: str
    protocol: str
    split: SplitName
    seed: int
    profile_texts: tuple[str, ...]
    query_texts: tuple[str, ...]
    metadata: dict[str, Any] | None = None

    @property
    def profile_text(self) -> str:
        return "\n".join(self.profile_texts)

    @property
    def query_text(self) -> str:
        return "\n".join(self.query_texts)


@dataclass(frozen=True)
class HardnessCandidate:
    """A candidate imposter query author for one profile author."""

    author_id: str
    similarity_score: float


@dataclass(frozen=True)
class AuthorHardnessCandidates:
    """Ranked imposter buckets for one profile author."""

    random: tuple[HardnessCandidate, ...]
    semi_hard: tuple[HardnessCandidate, ...]
    hard: tuple[HardnessCandidate, ...]

    def get(self, hardness: HardnessLabel) -> tuple[HardnessCandidate, ...]:
        if hardness == HardnessLabel.RANDOM:
            return self.random
        if hardness == HardnessLabel.SEMI_HARD:
            return self.semi_hard
        if hardness == HardnessLabel.HARD:
            return self.hard
        raise ValueError(f"Unsupported negative hardness: {hardness}")


def load_profile_records(path: str | Path) -> list[ProfileRecord]:
    """Load profile records from a JSONL split file."""
    path = Path(path)
    records: list[ProfileRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(_profile_record_from_row(row, path=path, line_number=line_number))
    return records


def write_examples_jsonl(path: str | Path, examples: Iterable[dict[str, Any]]) -> None:
    """Write generated example rows as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")


def build_profile_example_sets(
    profiles: Sequence[ProfileRecord],
    *,
    dataset: str,
    seed: int,
    split: SplitName | str,
    prefix_lengths: Sequence[int] = (1, 3),
    miner: str = "tfidf",
) -> dict[str, list[dict[str, Any]]]:
    """Build all standard example sets for one dataset seed/split."""
    split = _coerce_split(split)
    regimes = regimes_for_split(split)
    hardness_index = build_hardness_index(
        profiles,
        prefix_lengths=prefix_lengths,
        miner=miner,
    )
    return {
        f"{split.value}_{regime}": build_profile_examples_for_regime(
            profiles,
            dataset=dataset,
            seed=seed,
            split=split,
            regime=regime,
            prefix_lengths=prefix_lengths,
            hardness_index=hardness_index,
        )
        for regime in regimes
    }


def regimes_for_split(split: SplitName | str) -> tuple[str, ...]:
    """Return the standard example regimes for a split."""
    split = _coerce_split(split)
    return TEST_REGIMES if split == SplitName.TEST else TRAIN_VAL_REGIMES


def build_profile_examples_for_regime(
    profiles: Sequence[ProfileRecord],
    *,
    dataset: str,
    seed: int,
    split: SplitName | str,
    regime: str,
    prefix_lengths: Sequence[int] = (1, 3),
    hardness_index: dict[str, AuthorHardnessCandidates] | None = None,
) -> list[dict[str, Any]]:
    """Build raw-text positive and negative rows for one regime."""
    return list(
        iter_profile_examples_for_regime(
            profiles,
            dataset=dataset,
            seed=seed,
            split=split,
            regime=regime,
            prefix_lengths=prefix_lengths,
            hardness_index=hardness_index,
        )
    )


def iter_profile_examples_for_regime(
    profiles: Sequence[ProfileRecord],
    *,
    dataset: str,
    seed: int,
    split: SplitName | str,
    regime: str,
    prefix_lengths: Sequence[int] = (1, 3),
    hardness_index: dict[str, AuthorHardnessCandidates] | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield raw-text positive and negative rows for one regime."""
    split = _coerce_split(split)
    sorted_profiles = _sorted_profiles(profiles)
    _validate_profiles(sorted_profiles, prefix_lengths=prefix_lengths)
    _validate_regime(regime)

    if hardness_index is None:
        hardness_index = build_hardness_index(sorted_profiles, prefix_lengths=prefix_lengths)

    profile_by_author = {profile.author_id: profile for profile in sorted_profiles}
    rng = random.Random(_derive_seed(seed=seed, split=split.value, regime=regime))
    negative_hardnesses = _negative_hardness_sequence(
        total=len(sorted_profiles),
        regime=regime,
        rng=rng,
    )

    for profile_index, (profile, negative_hardness) in enumerate(
        zip(sorted_profiles, negative_hardnesses, strict=True)
    ):
        candidate = _sample_candidate(
            profile.author_id,
            negative_hardness,
            hardness_index,
            rng=rng,
        )
        negative_profile = profile_by_author[candidate.author_id]

        for prefix_len in prefix_lengths:
            yield _make_example_row(
                dataset=dataset,
                seed=seed,
                split=split,
                regime=regime,
                prefix_len=prefix_len,
                label=1,
                hardness=HardnessLabel.POSITIVE,
                profile=profile,
                query_profile=profile,
                similarity_score=None,
                profile_index=profile_index,
            )
            yield _make_example_row(
                dataset=dataset,
                seed=seed,
                split=split,
                regime=regime,
                prefix_len=prefix_len,
                label=0,
                hardness=negative_hardness,
                profile=profile,
                query_profile=negative_profile,
                similarity_score=candidate.similarity_score,
                profile_index=profile_index,
            )


def build_hardness_index(
    profiles: Sequence[ProfileRecord],
    *,
    prefix_lengths: Sequence[int] = (1, 3),
    miner: str = "tfidf",
    miner_model: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    max_length: int = 256,
    max_chars_per_text: int = 2000,
    batch_size: int = 256,
) -> dict[str, AuthorHardnessCandidates]:
    """Mine random, semi-hard, and hard candidate pools inside one split."""
    if miner not in HARDNESS_MINERS:
        raise ValueError(
            f"Unsupported miner: {miner}. Currently supported: {', '.join(HARDNESS_MINERS)}"
        )

    sorted_profiles = _sorted_profiles(profiles)
    _validate_profiles(sorted_profiles, prefix_lengths=prefix_lengths)
    if len(sorted_profiles) < 2:
        raise ValueError("Need at least two profiles to mine imposter candidates")

    if miner == TFIDF_HARDNESS_MINER:
        profile_vectors, query_vectors = _encode_tfidf_hardness_vectors(sorted_profiles)
    else:
        profile_vectors, query_vectors = _encode_sbert_hardness_vectors(
            sorted_profiles,
            model_name=miner_model or DEFAULT_SBERT_HARDNESS_MODEL,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            max_length=max_length,
            max_chars_per_text=max_chars_per_text,
            batch_size=batch_size,
        )

    index: dict[str, AuthorHardnessCandidates] = {}
    for batch_start in range(0, len(sorted_profiles), batch_size):
        batch_end = min(batch_start + batch_size, len(sorted_profiles))
        similarities = cosine_similarity(profile_vectors[batch_start:batch_end], query_vectors)
        for row_offset, scores in enumerate(similarities):
            profile_index = batch_start + row_offset
            profile = sorted_profiles[profile_index]
            ranked_candidates = _rank_candidates(
                anchor_index=profile_index,
                profiles=sorted_profiles,
                scores=scores,
            )
            index[profile.author_id] = _bucket_candidates(ranked_candidates)

    return index


def _encode_tfidf_hardness_vectors(
    profiles: Sequence[ProfileRecord],
) -> tuple[Any, Any]:
    """Return profile/query vectors from the existing TF-IDF hardness miner."""
    profile_texts = [profile.profile_text for profile in profiles]
    query_texts = [profile.query_text for profile in profiles]
    encoder = TfidfEncoder(min_df=1)
    encoder.fit([*profile_texts, *query_texts])
    return encoder.encode(profile_texts), encoder.encode(query_texts)


def _encode_sbert_hardness_vectors(
    profiles: Sequence[ProfileRecord],
    *,
    model_name: str,
    cache_dir: str | Path | None,
    local_files_only: bool,
    max_length: int,
    max_chars_per_text: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return profile/query vectors from a sentence-transformer-style encoder."""
    if max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")
    if max_chars_per_text <= 0:
        raise ValueError(f"max_chars_per_text must be positive, got {max_chars_per_text}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir_str,
        local_files_only=local_files_only,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir_str,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    profile_vectors = _encode_profile_text_groups(
        [profile.profile_texts for profile in profiles],
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        max_chars_per_text=max_chars_per_text,
        batch_size=batch_size,
    )
    query_vectors = _encode_profile_text_groups(
        [profile.query_texts for profile in profiles],
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        max_chars_per_text=max_chars_per_text,
        batch_size=batch_size,
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return profile_vectors, query_vectors


def _encode_profile_text_groups(
    text_groups: Sequence[Sequence[str]],
    *,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
    max_chars_per_text: int,
    batch_size: int,
) -> np.ndarray:
    """Encode groups of texts as normalized mean-pooled group embeddings."""
    flat_texts: list[str] = []
    group_bounds: list[tuple[int, int]] = []
    for text_group in text_groups:
        start = len(flat_texts)
        flat_texts.extend(
            _truncate_for_sbert(text, max_chars_per_text=max_chars_per_text)
            for text in text_group
            if text.strip()
        )
        end = len(flat_texts)
        if start == end:
            raise ValueError("Cannot build SBERT hardness vectors for an empty text group")
        group_bounds.append((start, end))

    text_vectors = _encode_sbert_texts(
        flat_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    group_vectors = [
        _l2_normalize_vector(text_vectors[start:end].mean(axis=0))
        for start, end in group_bounds
    ]
    return np.vstack(group_vectors).astype(np.float32, copy=False)


def _encode_sbert_texts(
    texts: Sequence[str],
    *,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Encode individual texts with attention-mask mean pooling."""
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = list(texts[batch_start : batch_start + batch_size])
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = _mean_pool_last_hidden_state(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors).astype(np.float32, copy=False)


def _mean_pool_last_hidden_state(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _truncate_for_sbert(text: str, max_chars_per_text: int) -> str:
    if len(text) <= max_chars_per_text:
        return text
    half = max_chars_per_text // 2
    tail_chars = max_chars_per_text - half
    return text[:half] + text[-tail_chars:]


def save_hardness_index(
    path: str | Path,
    hardness_index: dict[str, AuthorHardnessCandidates],
) -> None:
    """Save mined hardness candidates as stable JSONL for reuse across runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as cache_file:
        cache_file.write(json.dumps({"format": HARDNESS_INDEX_CACHE_FORMAT}) + "\n")
        for author_id, candidates in sorted(hardness_index.items()):
            row = {
                "author_id": author_id,
                "candidates": {
                    HardnessLabel.RANDOM.value: _hardness_candidates_to_rows(
                        candidates.random
                    ),
                    HardnessLabel.SEMI_HARD.value: _hardness_candidates_to_rows(
                        candidates.semi_hard
                    ),
                    HardnessLabel.HARD.value: _hardness_candidates_to_rows(candidates.hard),
                },
            }
            cache_file.write(
                json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            )
    tmp_path.replace(path)


def load_hardness_index(
    path: str | Path,
    *,
    include_random: bool = True,
) -> dict[str, AuthorHardnessCandidates]:
    """Load a hardness index saved by ``save_hardness_index``."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as cache_file:
        first_line = cache_file.readline()
        try:
            header = json.loads(first_line)
        except json.JSONDecodeError:
            return _load_legacy_hardness_index(path, include_random=include_random)
        if isinstance(header, dict) and header.get("format") == HARDNESS_INDEX_CACHE_FORMAT:
            return _load_jsonl_hardness_index(
                path,
                cache_file,
                include_random=include_random,
            )
    return _load_legacy_hardness_index(path, include_random=include_random)


def _load_jsonl_hardness_index(
    path: Path,
    rows: Iterable[str],
    *,
    include_random: bool,
) -> dict[str, AuthorHardnessCandidates]:
    index: dict[str, AuthorHardnessCandidates] = {}
    for line_number, line in enumerate(rows, start=2):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Invalid hardness cache row at {path}:{line_number}")
        try:
            author_id = str(row["author_id"])
            candidates_row = row["candidates"]
        except KeyError as exc:
            raise ValueError(
                f"Missing required field {exc!s} in hardness cache at {path}:{line_number}"
            ) from exc
        if not isinstance(candidates_row, dict):
            raise ValueError(
                f"Invalid hardness candidates for author {author_id!r} at {path}:{line_number}"
            )
        index[author_id] = _author_hardness_candidates_from_row(
            candidates_row,
            path=path,
            author_id=author_id,
            include_random=include_random,
        )
    return index


def _load_legacy_hardness_index(
    path: Path,
    *,
    include_random: bool,
) -> dict[str, AuthorHardnessCandidates]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Hardness index must be a JSON object: {path}")

    index: dict[str, AuthorHardnessCandidates] = {}
    for author_id, row in payload.items():
        if not isinstance(row, dict):
            raise ValueError(f"Invalid hardness row for author {author_id!r} in {path}")
        author_id_str = str(author_id)
        index[author_id_str] = _author_hardness_candidates_from_row(
            row,
            path=path,
            author_id=author_id_str,
            include_random=include_random,
        )
    return index


def _author_hardness_candidates_from_row(
    row: dict[str, Any],
    *,
    path: Path,
    author_id: str,
    include_random: bool,
) -> AuthorHardnessCandidates:
    return AuthorHardnessCandidates(
        random=(
            _hardness_candidates_from_rows(
                row.get(HardnessLabel.RANDOM.value, ()),
                path=path,
                author_id=author_id,
                hardness=HardnessLabel.RANDOM,
            )
            if include_random
            else ()
        ),
        semi_hard=_hardness_candidates_from_rows(
            row.get(HardnessLabel.SEMI_HARD.value, ()),
            path=path,
            author_id=author_id,
            hardness=HardnessLabel.SEMI_HARD,
        ),
        hard=_hardness_candidates_from_rows(
            row.get(HardnessLabel.HARD.value, ()),
            path=path,
            author_id=author_id,
            hardness=HardnessLabel.HARD,
        ),
    )


def _profile_record_from_row(
    row: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> ProfileRecord:
    try:
        profile_texts = tuple(str(text) for text in row["profile_texts"])
        query_texts = tuple(str(text) for text in row["query_texts"])
        return ProfileRecord(
            author_id=str(row["author_id"]),
            dataset=str(row.get("dataset", "")),
            protocol=str(row.get("protocol", "")),
            split=_coerce_split(row["split"]),
            seed=int(row["seed"]),
            profile_texts=profile_texts,
            query_texts=query_texts,
            metadata={
                key: value
                for key, value in row.items()
                if key
                not in {
                    "author_id",
                    "dataset",
                    "protocol",
                    "split",
                    "seed",
                    "profile_texts",
                    "query_texts",
                }
            },
        )
    except KeyError as exc:
        raise ValueError(f"Missing required field {exc!s} at {path}:{line_number}") from exc


def _make_example_row(
    *,
    dataset: str,
    seed: int,
    split: SplitName,
    regime: str,
    prefix_len: int,
    label: int,
    hardness: HardnessLabel,
    profile: ProfileRecord,
    query_profile: ProfileRecord,
    similarity_score: float | None,
    profile_index: int,
) -> dict[str, Any]:
    query_texts = tuple(query_profile.query_texts[:prefix_len])
    return {
        "example_id": _make_example_id(
            dataset=dataset,
            seed=seed,
            split=split.value,
            regime=regime,
            prefix_len=prefix_len,
            label=label,
            profile_author_id=profile.author_id,
            query_author_id=query_profile.author_id,
        ),
        "dataset": dataset,
        "seed": seed,
        "split": split.value,
        "regime": regime,
        "prefix_len": prefix_len,
        "label": label,
        "hardness": hardness.value,
        "profile_author_id": profile.author_id,
        "query_author_id": query_profile.author_id,
        "profile_text": profile.profile_text,
        "query_texts": list(query_texts),
        "query_text": "\n".join(query_texts),
        "similarity_score": similarity_score,
        "metadata": {
            "profile_index": profile_index,
            "profile_text_count": len(profile.profile_texts),
            "full_query_text_count": len(query_profile.query_texts),
        },
    }


def _make_example_id(
    *,
    dataset: str,
    seed: int,
    split: str,
    regime: str,
    prefix_len: int,
    label: int,
    profile_author_id: str,
    query_author_id: str,
) -> str:
    label_name = "positive" if label == 1 else "negative"
    raw_id = "|".join(
        [
            dataset,
            str(seed),
            split,
            regime,
            str(prefix_len),
            str(label),
            profile_author_id,
            query_author_id,
        ]
    )
    digest = sha1(raw_id.encode("utf-8")).hexdigest()[:12]
    return f"{dataset}_{seed}_{split}_{regime}_p{prefix_len}_{label_name}_{digest}"


def _rank_candidates(
    *,
    anchor_index: int,
    profiles: Sequence[ProfileRecord],
    scores: Sequence[float],
) -> tuple[HardnessCandidate, ...]:
    candidates = [
        HardnessCandidate(
            author_id=profile.author_id,
            similarity_score=float(scores[candidate_index]),
        )
        for candidate_index, profile in enumerate(profiles)
        if candidate_index != anchor_index
    ]
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: (-candidate.similarity_score, candidate.author_id),
        )
    )


def _bucket_candidates(
    ranked_candidates: Sequence[HardnessCandidate],
) -> AuthorHardnessCandidates:
    if not ranked_candidates:
        raise ValueError("Cannot bucket an empty candidate list")

    candidate_count = len(ranked_candidates)
    hard_end = max(1, ceil(candidate_count * 0.05))

    # Semi-hard is the 70th-95th similarity percentile: after descending sort,
    # this is the slice below the hardest 5% and above the lower 70%.
    semi_end = max(hard_end + 1, ceil(candidate_count * 0.30))
    semi_end = min(semi_end, candidate_count)

    hard = tuple(ranked_candidates[:hard_end])
    semi_hard = tuple(ranked_candidates[hard_end:semi_end])
    if not semi_hard:
        semi_hard = tuple(ranked_candidates[hard_end:]) or hard

    return AuthorHardnessCandidates(
        random=tuple(ranked_candidates),
        semi_hard=semi_hard,
        hard=hard,
    )


def _hardness_candidates_to_rows(
    candidates: Sequence[HardnessCandidate],
) -> list[dict[str, Any]]:
    return [
        {
            "author_id": candidate.author_id,
            "similarity_score": candidate.similarity_score,
        }
        for candidate in candidates
    ]


def _hardness_candidates_from_rows(
    rows: Any,
    *,
    path: Path,
    author_id: str,
    hardness: HardnessLabel,
) -> tuple[HardnessCandidate, ...]:
    if not isinstance(rows, list):
        raise ValueError(
            f"Expected list for {author_id!r}/{hardness.value} candidates in {path}"
        )
    candidates: list[HardnessCandidate] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                f"Invalid candidate row {row_index} for {author_id!r}/{hardness.value} in {path}"
            )
        try:
            candidates.append(
                HardnessCandidate(
                    author_id=str(row["author_id"]),
                    similarity_score=float(row["similarity_score"]),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"Missing candidate field {exc!s} for {author_id!r}/{hardness.value} in {path}"
            ) from exc
    return tuple(candidates)


def _negative_hardness_sequence(
    *,
    total: int,
    regime: str,
    rng: random.Random,
) -> list[HardnessLabel]:
    if total <= 0:
        return []
    if regime == RANDOM_ONLY_REGIME or regime == HardnessLabel.RANDOM.value:
        return [HardnessLabel.RANDOM] * total
    if regime == HardnessLabel.SEMI_HARD.value:
        return [HardnessLabel.SEMI_HARD] * total
    if regime == HardnessLabel.HARD.value:
        return [HardnessLabel.HARD] * total
    if regime != MIXED_REGIME:
        raise ValueError(f"Unsupported regime: {regime}")

    counts = _allocate_weighted_counts(total, MIXED_NEGATIVE_WEIGHTS)
    hardnesses: list[HardnessLabel] = []
    for hardness in (HardnessLabel.RANDOM, HardnessLabel.SEMI_HARD, HardnessLabel.HARD):
        hardnesses.extend([hardness] * counts[hardness])
    rng.shuffle(hardnesses)
    return hardnesses


def _allocate_weighted_counts(
    total: int,
    weights: dict[HardnessLabel, float],
) -> dict[HardnessLabel, int]:
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


def _sample_candidate(
    author_id: str,
    hardness: HardnessLabel,
    hardness_index: dict[str, AuthorHardnessCandidates],
    *,
    rng: random.Random,
) -> HardnessCandidate:
    buckets = hardness_index.get(author_id)
    if buckets is None:
        raise ValueError(f"No hardness candidates found for author_id={author_id}")

    candidates = buckets.get(hardness)
    if not candidates:
        raise ValueError(f"No {hardness.value} candidates found for author_id={author_id}")
    return rng.choice(candidates)


def _validate_profiles(
    profiles: Sequence[ProfileRecord],
    *,
    prefix_lengths: Sequence[int],
) -> None:
    if not profiles:
        raise ValueError("No profiles provided")
    if len({profile.author_id for profile in profiles}) != len(profiles):
        raise ValueError("Profile author IDs must be unique within a split")
    if not prefix_lengths:
        raise ValueError("At least one prefix length is required")
    if any(prefix_len <= 0 for prefix_len in prefix_lengths):
        raise ValueError(f"Prefix lengths must be positive: {prefix_lengths}")

    max_prefix_len = max(prefix_lengths)
    short_authors = [
        profile.author_id
        for profile in profiles
        if len(profile.query_texts) < max_prefix_len
    ]
    if short_authors:
        preview = ", ".join(short_authors[:5])
        raise ValueError(
            f"{len(short_authors)} profiles have fewer than {max_prefix_len} query texts. "
            f"Examples: {preview}"
        )


def _validate_regime(regime: str) -> None:
    valid_regimes = {RANDOM_ONLY_REGIME, MIXED_REGIME, *PURE_NEGATIVE_REGIMES}
    if regime not in valid_regimes:
        raise ValueError(f"Unsupported regime: {regime}. Valid regimes: {sorted(valid_regimes)}")


def _sorted_profiles(profiles: Sequence[ProfileRecord]) -> list[ProfileRecord]:
    return sorted(profiles, key=lambda profile: profile.author_id)


def _coerce_split(split: SplitName | str) -> SplitName:
    if isinstance(split, SplitName):
        return split
    return SplitName(str(split))


def _derive_seed(*, seed: int, split: str, regime: str) -> int:
    digest = sha1(f"{seed}|{split}|{regime}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)
