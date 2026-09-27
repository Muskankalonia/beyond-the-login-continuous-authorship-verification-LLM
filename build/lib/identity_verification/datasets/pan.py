"""Audit helpers for PAN authorship verification data."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha1
import json
from math import ceil
from pathlib import Path
import random
from statistics import mean, median
from typing import Any

from identity_verification.schemas import SplitName


@dataclass(frozen=True)
class PanPairRecord:
    pair_id: str
    text_a: str
    text_b: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PanTruthRecord:
    pair_id: str
    same: bool
    author_a: str | None
    author_b: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PanAuthorProfile:
    author_id: str
    split: SplitName
    profile_texts: tuple[str, ...]
    query_texts: tuple[str, ...]

    @property
    def profile_text(self) -> str:
        return "\n".join(self.profile_texts)

    @property
    def query_text(self) -> str:
        return "\n".join(self.query_texts)


def load_pan_author_texts(
    pairs_path: str | Path,
    truth_path: str | Path,
    *,
    dedupe_pair_ids: bool = True,
) -> dict[str, tuple[str, ...]]:
    """Reconstruct author -> unique texts from PAN pair/truth files.

    Author IDs are used only for grouping. They must not be used as model inputs.
    Texts are deduplicated per author by normalized text hash while preserving the
    first observed text for that author/hash.
    """
    pairs_path = Path(pairs_path)
    truth_path = Path(truth_path)
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Pairs file does not exist: {pairs_path}")
    if not truth_path.is_file():
        raise FileNotFoundError(f"Truth file does not exist: {truth_path}")

    truth_by_id = _load_truth_by_id(truth_path)
    author_hash_to_text: dict[str, dict[str, str]] = defaultdict(dict)
    seen_pair_ids: set[str] = set()

    for row in _iter_jsonl_dicts(pairs_path):
        if row is None:
            continue

        pair = _parse_pair_row(row)
        if pair is None:
            continue
        if dedupe_pair_ids and pair.pair_id in seen_pair_ids:
            continue
        seen_pair_ids.add(pair.pair_id)

        truth = truth_by_id.get(pair.pair_id)
        if truth is None:
            continue

        if truth.author_a is not None:
            _add_author_text(author_hash_to_text, truth.author_a, pair.text_a)
        if truth.author_b is not None:
            _add_author_text(author_hash_to_text, truth.author_b, pair.text_b)

    return {
        author_id: tuple(hash_to_text.values())
        for author_id, hash_to_text in author_hash_to_text.items()
    }


def build_pan_author_profiles(
    pairs_path: str | Path,
    truth_path: str | Path,
    *,
    seed: int,
    min_texts_per_author: int = 10,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    profile_fraction: float = 0.7,
) -> dict[SplitName, list[PanAuthorProfile]]:
    """Build PAN profile/query units with author-disjoint stratified splits.

    PAN has no timestamps, so each eligible author's texts are randomly split into
    profile/query text sets using the provided seed.
    """
    _validate_profile_builder_args(
        min_texts_per_author=min_texts_per_author,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        profile_fraction=profile_fraction,
    )

    author_texts = load_pan_author_texts(pairs_path, truth_path)
    eligible_author_ids = sorted(
        author_id
        for author_id, texts in author_texts.items()
        if len(texts) >= min_texts_per_author
    )
    if len(eligible_author_ids) < 3:
        raise ValueError(
            "Need at least 3 eligible authors to build train/val/test splits, "
            f"got {len(eligible_author_ids)}"
        )

    split_author_ids = _stratified_author_split(
        eligible_author_ids=eligible_author_ids,
        author_texts=author_texts,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    profiles: dict[SplitName, list[PanAuthorProfile]] = {
        SplitName.TRAIN: [],
        SplitName.VAL: [],
        SplitName.TEST: [],
    }
    for split, author_ids in split_author_ids.items():
        for author_id in author_ids:
            profile_texts, query_texts = _split_author_texts(
                author_id=author_id,
                texts=author_texts[author_id],
                seed=seed,
                profile_fraction=profile_fraction,
            )
            profiles[split].append(
                PanAuthorProfile(
                    author_id=author_id,
                    split=split,
                    profile_texts=profile_texts,
                    query_texts=query_texts,
                )
            )

    return profiles


def audit_pan_dataset(pairs_path: str | Path, truth_path: str | Path) -> dict[str, Any]:
    """Audit PAN pair and truth files.

    The audit intentionally stores only aggregate counts and text hashes. Raw text is not
    written into the report.
    """
    pairs_path = Path(pairs_path)
    truth_path = Path(truth_path)
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Pairs file does not exist: {pairs_path}")
    if not truth_path.is_file():
        raise FileNotFoundError(f"Truth file does not exist: {truth_path}")

    truth_records, truth_row_count, truth_json_errors, malformed_truth = _load_truth_records(
        truth_path
    )
    truth_by_id: dict[str, PanTruthRecord] = {}
    duplicate_truth_ids = 0
    for truth in truth_records:
        if truth.pair_id in truth_by_id:
            duplicate_truth_ids += 1
        truth_by_id[truth.pair_id] = truth

    pair_row_count = 0
    pair_json_errors = 0
    valid_pair_rows = 0
    malformed_pairs = 0
    pair_ids: set[str] = set()
    processed_pair_ids: set[str] = set()
    duplicate_pair_ids = 0

    label_counts: Counter[str] = Counter()
    author_text_hashes: dict[str, set[str]] = defaultdict(set)
    unique_text_hashes: set[str] = set()
    text_reference_count = 0
    joined_pair_count = 0

    for row in _iter_jsonl_dicts(pairs_path):
        if row is None:
            pair_json_errors += 1
            continue

        pair_row_count += 1
        pair = _parse_pair_row(row)
        if pair is None:
            malformed_pairs += 1
            continue

        valid_pair_rows += 1
        pair_ids.add(pair.pair_id)
        if pair.pair_id in processed_pair_ids:
            duplicate_pair_ids += 1
            continue
        processed_pair_ids.add(pair.pair_id)

        truth = truth_by_id.get(pair.pair_id)
        if truth is None:
            continue

        joined_pair_count += 1
        label_counts["positive" if truth.same else "negative"] += 1

        text_a_hash = _hash_text(pair.text_a)
        text_b_hash = _hash_text(pair.text_b)
        unique_text_hashes.update([text_a_hash, text_b_hash])
        text_reference_count += 2

        if truth.author_a is not None:
            author_text_hashes[truth.author_a].add(text_a_hash)
        if truth.author_b is not None:
            author_text_hashes[truth.author_b].add(text_b_hash)

    texts_per_author = [len(texts) for texts in author_text_hashes.values()]

    return {
        "pairs_path": str(pairs_path),
        "truth_path": str(truth_path),
        "pair_rows": pair_row_count,
        "truth_rows": truth_row_count,
        "pair_json_errors": pair_json_errors,
        "truth_json_errors": truth_json_errors,
        "valid_pair_rows": valid_pair_rows,
        "valid_truth_rows": len(truth_records),
        "unique_pair_ids": len(pair_ids),
        "unique_truth_ids": len(truth_by_id),
        "malformed_pair_rows": malformed_pairs,
        "malformed_truth_rows": malformed_truth,
        "joined_pair_rows": joined_pair_count,
        "missing_truth_rows": len(pair_ids - set(truth_by_id)),
        "extra_truth_rows": len(set(truth_by_id) - pair_ids),
        "duplicate_pair_ids": duplicate_pair_ids,
        "duplicate_truth_ids": duplicate_truth_ids,
        "positive_pairs": label_counts["positive"],
        "negative_pairs": label_counts["negative"],
        "unique_authors": len(author_text_hashes),
        "unique_texts": len(unique_text_hashes),
        "text_references": text_reference_count,
        "duplicate_text_references": text_reference_count - len(unique_text_hashes),
        "texts_per_author": _numeric_summary(texts_per_author),
        "authors_with_min_texts": {
            "ge_2": _count_at_least(texts_per_author, 2),
            "ge_3": _count_at_least(texts_per_author, 3),
            "ge_5": _count_at_least(texts_per_author, 5),
            "ge_10": _count_at_least(texts_per_author, 10),
        },
        "can_build_pair_benchmark": label_counts["positive"] > 0 and label_counts["negative"] > 0,
        "can_build_profile_benchmark_ge_10": _count_at_least(texts_per_author, 10) > 0,
        "model_input_policy": {
            "allowed_model_inputs": ["text_a", "text_b"],
            "excluded_from_model_inputs": ["author_ids", "pair_id", "fandoms", "metadata"],
        },
    }


def format_pan_audit(audit: dict[str, Any]) -> str:
    """Format a PAN audit report for console output."""
    lines = [
        "PAN Audit",
        "=========",
        f"Pairs file: {audit['pairs_path']}",
        f"Truth file: {audit['truth_path']}",
        "",
        f"Pair rows: {audit['pair_rows']}",
        f"Truth rows: {audit['truth_rows']}",
        f"Valid pair rows: {audit['valid_pair_rows']}",
        f"Valid truth rows: {audit['valid_truth_rows']}",
        f"Unique pair IDs: {audit['unique_pair_ids']}",
        f"Unique truth IDs: {audit['unique_truth_ids']}",
        f"Duplicate pair IDs: {audit['duplicate_pair_ids']}",
        f"Duplicate truth IDs: {audit['duplicate_truth_ids']}",
        f"Joined pair rows: {audit['joined_pair_rows']}",
        f"Missing truth rows: {audit['missing_truth_rows']}",
        f"Extra truth rows: {audit['extra_truth_rows']}",
        "",
        f"Positive pairs: {audit['positive_pairs']}",
        f"Negative pairs: {audit['negative_pairs']}",
        f"Unique authors: {audit['unique_authors']}",
        f"Unique texts: {audit['unique_texts']}",
        f"Duplicate text references: {audit['duplicate_text_references']}",
        "",
        "Authors with at least N unique texts:",
        f"  >= 2: {audit['authors_with_min_texts']['ge_2']}",
        f"  >= 3: {audit['authors_with_min_texts']['ge_3']}",
        f"  >= 5: {audit['authors_with_min_texts']['ge_5']}",
        f"  >= 10: {audit['authors_with_min_texts']['ge_10']}",
        "",
        "Texts per author:",
        _format_numeric_summary(audit["texts_per_author"]),
        "",
        "Model input policy: use text only; do not feed author IDs, pair IDs, fandoms, or metadata.",
    ]
    return "\n".join(lines)


def _iter_jsonl_dicts(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                yield None
                continue
            if isinstance(row, dict):
                yield row
            else:
                yield None


def _load_truth_records(path: Path) -> tuple[list[PanTruthRecord], int, int, int]:
    records: list[PanTruthRecord] = []
    row_count = 0
    json_errors = 0
    malformed = 0

    for row in _iter_jsonl_dicts(path):
        if row is None:
            json_errors += 1
            continue

        row_count += 1
        record = _parse_truth_row(row)
        if record is None:
            malformed += 1
            continue
        records.append(record)

    return records, row_count, json_errors, malformed


def _load_truth_by_id(path: Path) -> dict[str, PanTruthRecord]:
    truth_by_id: dict[str, PanTruthRecord] = {}
    for truth in _load_truth_records(path)[0]:
        if truth.pair_id not in truth_by_id:
            truth_by_id[truth.pair_id] = truth
    return truth_by_id


def _parse_pair_rows(rows: list[dict[str, Any]]) -> tuple[list[PanPairRecord], int]:
    records: list[PanPairRecord] = []
    malformed = 0
    for row in rows:
        record = _parse_pair_row(row)
        if record is None:
            malformed += 1
            continue
        records.append(record)
    return records, malformed


def _parse_truth_rows(rows: list[dict[str, Any]]) -> tuple[list[PanTruthRecord], int]:
    records: list[PanTruthRecord] = []
    malformed = 0
    for row in rows:
        record = _parse_truth_row(row)
        if record is None:
            malformed += 1
            continue
        records.append(record)
    return records, malformed


def _parse_pair_row(row: dict[str, Any]) -> PanPairRecord | None:
    pair_id = _extract_id(row)
    texts = _extract_pair_texts(row)
    if pair_id is None or texts is None:
        return None
    return PanPairRecord(pair_id=pair_id, text_a=texts[0], text_b=texts[1], metadata=row)


def _parse_truth_row(row: dict[str, Any]) -> PanTruthRecord | None:
    pair_id = _extract_id(row)
    same = row.get("same")
    authors = row.get("authors")
    if pair_id is None or not isinstance(same, bool):
        return None

    author_a: str | None = None
    author_b: str | None = None
    if isinstance(authors, list) and len(authors) >= 2:
        author_a = str(authors[0])
        author_b = str(authors[1])

    return PanTruthRecord(
        pair_id=pair_id,
        same=same,
        author_a=author_a,
        author_b=author_b,
        metadata=row,
    )


def _extract_id(row: dict[str, Any]) -> str | None:
    for key in ("id", "pair_id", "problem_id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return None


def _extract_pair_texts(row: dict[str, Any]) -> tuple[str, str] | None:
    for key in ("pair", "texts", "documents"):
        value = row.get(key)
        if isinstance(value, list) and len(value) >= 2:
            if isinstance(value[0], str) and isinstance(value[1], str):
                return value[0], value[1]

    direct_key_pairs = [
        ("text_a", "text_b"),
        ("text1", "text2"),
        ("document_a", "document_b"),
        ("document1", "document2"),
    ]
    for key_a, key_b in direct_key_pairs:
        value_a = row.get(key_a)
        value_b = row.get(key_b)
        if isinstance(value_a, str) and isinstance(value_b, str):
            return value_a, value_b

    return None


def _hash_text(text: str) -> str:
    normalized = " ".join(text.split())
    return sha1(normalized.encode("utf-8")).hexdigest()


def _add_author_text(
    author_hash_to_text: dict[str, dict[str, str]],
    author_id: str,
    text: str,
) -> None:
    text_hash = _hash_text(text)
    author_hash_to_text[author_id].setdefault(text_hash, text)


def _validate_profile_builder_args(
    *,
    min_texts_per_author: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    profile_fraction: float,
) -> None:
    if min_texts_per_author < 2:
        raise ValueError("min_texts_per_author must be at least 2")
    if not 0.0 < profile_fraction < 1.0:
        raise ValueError(f"profile_fraction must be in (0, 1), got {profile_fraction}")

    fractions = (train_fraction, val_fraction, test_fraction)
    if any(fraction <= 0.0 for fraction in fractions):
        raise ValueError("train/val/test fractions must all be positive")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(
            "train/val/test fractions must sum to 1.0, got "
            f"{train_fraction + val_fraction + test_fraction}"
        )


def _stratified_author_split(
    *,
    eligible_author_ids: list[str],
    author_texts: dict[str, tuple[str, ...]],
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[SplitName, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for author_id in eligible_author_ids:
        bucket = _text_count_bucket(len(author_texts[author_id]))
        buckets[bucket].append(author_id)

    split_author_ids: dict[SplitName, list[str]] = {
        SplitName.TRAIN: [],
        SplitName.VAL: [],
        SplitName.TEST: [],
    }

    for bucket_name in sorted(buckets):
        bucket_author_ids = sorted(buckets[bucket_name])
        bucket_rng = random.Random(_stable_author_seed(seed, bucket_name))
        bucket_rng.shuffle(bucket_author_ids)

        bucket_splits = _split_ids_by_fraction(
            bucket_author_ids,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

        for split, author_ids in bucket_splits.items():
            split_author_ids[split].extend(author_ids)

    for split in split_author_ids:
        split_author_ids[split].sort()

    _ensure_non_empty_splits(split_author_ids)
    return split_author_ids


def _text_count_bucket(text_count: int) -> str:
    if text_count <= 12:
        return "10_12"
    if text_count <= 16:
        return "13_16"
    if text_count <= 20:
        return "17_20"
    return "21_plus"


def _split_ids_by_fraction(
    ids: list[str],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[SplitName, list[str]]:
    counts = _fraction_counts(
        total=len(ids),
        fractions=(train_fraction, val_fraction, test_fraction),
    )
    train_count, val_count, _test_count = counts
    val_end = train_count + val_count
    return {
        SplitName.TRAIN: ids[:train_count],
        SplitName.VAL: ids[train_count:val_end],
        SplitName.TEST: ids[val_end:],
    }


def _ensure_non_empty_splits(split_author_ids: dict[SplitName, list[str]]) -> None:
    total_authors = sum(len(author_ids) for author_ids in split_author_ids.values())
    if total_authors < len(split_author_ids):
        return

    for split, author_ids in split_author_ids.items():
        if author_ids:
            continue

        donor_split = max(split_author_ids, key=lambda candidate: len(split_author_ids[candidate]))
        if len(split_author_ids[donor_split]) <= 1:
            return
        moved_author_id = split_author_ids[donor_split].pop()
        split_author_ids[split].append(moved_author_id)

    for author_ids in split_author_ids.values():
        author_ids.sort()


def _fraction_counts(total: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    raw_counts = [total * fraction for fraction in fractions]
    counts = [int(raw_count) for raw_count in raw_counts]
    remaining = total - sum(counts)

    remainders = sorted(
        range(len(fractions)),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in remainders[:remaining]:
        counts[index] += 1

    if total >= len(counts):
        for index, count in enumerate(counts):
            if count != 0:
                continue
            donor_index = max(range(len(counts)), key=lambda candidate: counts[candidate])
            if counts[donor_index] <= 1:
                break
            counts[donor_index] -= 1
            counts[index] += 1

    return counts[0], counts[1], counts[2]


def _split_author_texts(
    *,
    author_id: str,
    texts: tuple[str, ...],
    seed: int,
    profile_fraction: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    shuffled_texts = list(texts)
    author_rng = random.Random(_stable_author_seed(seed, author_id))
    author_rng.shuffle(shuffled_texts)

    profile_count = int(len(shuffled_texts) * profile_fraction)
    profile_count = min(max(1, profile_count), len(shuffled_texts) - 1)
    return tuple(shuffled_texts[:profile_count]), tuple(shuffled_texts[profile_count:])


def _stable_author_seed(seed: int, author_id: str) -> int:
    digest = sha1(f"{seed}:{author_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _count_at_least(values: list[int], threshold: int) -> int:
    return sum(value >= threshold for value in values)


def _numeric_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p90": None, "max": None}

    sorted_values = sorted(values)
    p90_index = min(len(sorted_values) - 1, max(0, ceil(0.9 * len(sorted_values)) - 1))
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "median": float(median(sorted_values)),
        "mean": float(mean(sorted_values)),
        "p90": sorted_values[p90_index],
        "max": sorted_values[-1],
    }


def _format_numeric_summary(summary: dict[str, float | int | None]) -> str:
    if summary["count"] == 0:
        return "  no values"
    return (
        f"  count={summary['count']}, min={summary['min']}, median={summary['median']}, "
        f"mean={summary['mean']:.2f}, p90={summary['p90']}, max={summary['max']}"
    )
