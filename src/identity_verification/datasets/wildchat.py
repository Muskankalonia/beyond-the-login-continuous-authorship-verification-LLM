"""Audit helpers for WildChat-style conversation data."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha1
import json
from math import ceil
from pathlib import Path
import random
from statistics import mean, median
from typing import Any, Iterable

from identity_verification.schemas import SplitName


DEFAULT_USER_FIELD = "hashed_ip"

USER_FIELD_CANDIDATES = ("hashed_ip", "user_id", "uid", "user_hash")
TIMESTAMP_FIELD_CANDIDATES = ("timestamp", "created_at", "time")
LANGUAGE_FIELD_CANDIDATES = ("language", "lang", "detected_language")

MODEL_EXCLUDED_FIELDS = [
    "hashed_ip",
    "user_id",
    "uid",
    "user_hash",
    "timestamp",
    "created_at",
    "time",
    "country",
    "state",
    "ip",
    "header",
    "request_header",
    "language",
    "lang",
    "detected_language",
]


@dataclass(frozen=True)
class WildChatMessage:
    """One user-authored WildChat message with ordering metadata."""

    text: str
    timestamp_key: str
    row_index: int
    message_index: int


@dataclass(frozen=True)
class WildChatUserProfile:
    """Profile/query unit for WildChat identity verification."""

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


def audit_wildchat_dataset(
    input_path: str | Path,
    user_field: str = DEFAULT_USER_FIELD,
) -> dict[str, Any]:
    """Audit WildChat conversations without exposing identifying fields to model inputs."""
    input_path = Path(input_path)
    files = _discover_input_files(input_path)

    row_count = 0
    json_errors = 0
    top_level_fields: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    missing_user_id_rows = 0
    rows_with_user_messages = 0
    user_message_counts: dict[str, int] = defaultdict(int)
    user_conversation_counts: dict[str, int] = defaultdict(int)
    english_user_message_counts: dict[str, int] = defaultdict(int)
    english_user_conversation_counts: dict[str, int] = defaultdict(int)

    detected_user_fields: Counter[str] = Counter()
    detected_timestamp_fields: Counter[str] = Counter()
    detected_language_fields: Counter[str] = Counter()

    for file_path in files:
        for row in _iter_rows(file_path):
            if row is None:
                json_errors += 1
                continue

            row_count += 1
            top_level_fields.update(row.keys())
            detected_user_fields.update(_present_fields(row, USER_FIELD_CANDIDATES))
            detected_timestamp_fields.update(_present_fields(row, TIMESTAMP_FIELD_CANDIDATES))
            detected_language_fields.update(_present_fields(row, LANGUAGE_FIELD_CANDIDATES))

            language = _extract_language(row)
            if language is not None:
                language_counts[language] += 1

            user_id = _extract_user_id(row, user_field)
            if user_id is None:
                missing_user_id_rows += 1
                continue

            user_texts = _extract_user_texts(row)
            if not user_texts:
                continue

            rows_with_user_messages += 1
            message_count = len(user_texts)
            user_message_counts[user_id] += message_count
            user_conversation_counts[user_id] += 1

            if _is_english_language(language):
                english_user_message_counts[user_id] += message_count
                english_user_conversation_counts[user_id] += 1

    message_counts = list(user_message_counts.values())
    english_message_counts = list(english_user_message_counts.values())

    return {
        "input_path": str(input_path),
        "files": [str(path) for path in files],
        "file_count": len(files),
        "row_count": row_count,
        "json_errors": json_errors,
        "default_user_field": user_field,
        "missing_user_id_rows": missing_user_id_rows,
        "rows_with_user_messages": rows_with_user_messages,
        "detected_user_fields": dict(detected_user_fields),
        "detected_timestamp_fields": dict(detected_timestamp_fields),
        "detected_language_fields": dict(detected_language_fields),
        "top_level_fields": dict(top_level_fields.most_common(30)),
        "language_counts": dict(language_counts.most_common(30)),
        "all_users": {
            "unique_users": len(user_message_counts),
            "message_counts_per_user": _numeric_summary(message_counts),
            "users_with_min_messages": _threshold_counts(message_counts),
        },
        "english_users": {
            "unique_users": len(english_user_message_counts),
            "message_counts_per_user": _numeric_summary(english_message_counts),
            "users_with_min_messages": _threshold_counts(english_message_counts),
        },
        "model_input_policy": {
            "grouping_user_field": user_field,
            "allowed_model_inputs": ["user_message_text"],
            "excluded_from_model_inputs": MODEL_EXCLUDED_FIELDS,
            "language_use": "allowed for filtering/splitting analysis only; never feed language as a feature",
            "timestamp_use": "allowed for chronological ordering only; never feed timestamp as a feature",
        },
    }


def format_wildchat_audit(audit: dict[str, Any]) -> str:
    """Format a WildChat audit report for console output."""
    all_users = audit["all_users"]
    english_users = audit["english_users"]
    lines = [
        "WildChat Audit",
        "==============",
        f"Input path: {audit['input_path']}",
        f"Files: {audit['file_count']}",
        f"Rows/conversations: {audit['row_count']}",
        f"Rows with user messages: {audit['rows_with_user_messages']}",
        f"Default grouping field: {audit['default_user_field']}",
        f"Rows missing grouping field: {audit['missing_user_id_rows']}",
        "",
        "Detected fields:",
        f"  user fields: {audit['detected_user_fields']}",
        f"  timestamp fields: {audit['detected_timestamp_fields']}",
        f"  language fields: {audit['detected_language_fields']}",
        "",
        "All users:",
        f"  unique users: {all_users['unique_users']}",
        _format_numeric_summary("  messages per user", all_users["message_counts_per_user"]),
        _format_threshold_counts("  users with at least N messages", all_users["users_with_min_messages"]),
        "",
        "English users, if language labels are available:",
        f"  unique users: {english_users['unique_users']}",
        _format_numeric_summary("  messages per user", english_users["message_counts_per_user"]),
        _format_threshold_counts("  users with at least N messages", english_users["users_with_min_messages"]),
        "",
        f"Top languages: {audit['language_counts']}",
        "",
        "Model input policy: use user message text only. Use hashed_ip only for grouping.",
        "Do not feed timestamps, IP/user IDs, countries, headers, or language labels into the model.",
    ]
    return "\n".join(lines)


def load_wildchat_user_messages(
    input_path: str | Path,
    *,
    user_field: str = DEFAULT_USER_FIELD,
    language: str = "English",
) -> dict[str, tuple[WildChatMessage, ...]]:
    """Load English WildChat user messages grouped by hashed user ID.

    User IDs and timestamps are used only for grouping and ordering. They must
    never be included in model inputs.
    """
    input_path = Path(input_path)
    files = _discover_input_files(input_path)
    allowed_language = language.strip().lower()
    user_messages: dict[str, list[WildChatMessage]] = defaultdict(list)
    row_index = 0

    for file_path in files:
        for row in _iter_rows(file_path):
            if row is None:
                row_index += 1
                continue

            row_index += 1
            row_language = _extract_language(row)
            if row_language is None or row_language.strip().lower() != allowed_language:
                continue

            user_id = _extract_user_id(row, user_field)
            if user_id is None:
                continue

            timestamp_key = _timestamp_sort_key(row.get("timestamp"))
            for message_index, text in enumerate(_extract_user_texts(row)):
                user_messages[user_id].append(
                    WildChatMessage(
                        text=text,
                        timestamp_key=timestamp_key,
                        row_index=row_index,
                        message_index=message_index,
                    )
                )

    return {
        user_id: tuple(sorted(messages, key=_message_sort_key))
        for user_id, messages in user_messages.items()
    }


def build_wildchat_user_profiles(
    input_path: str | Path,
    *,
    seed: int,
    user_field: str = DEFAULT_USER_FIELD,
    language: str = "English",
    min_messages_per_user: int = 10,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    profile_fraction: float = 0.7,
) -> dict[SplitName, list[WildChatUserProfile]]:
    """Build WildChat profile/query units with user-disjoint stratified splits."""
    _validate_profile_builder_args(
        min_messages_per_user=min_messages_per_user,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        profile_fraction=profile_fraction,
    )

    user_messages = load_wildchat_user_messages(
        input_path,
        user_field=user_field,
        language=language,
    )
    return build_wildchat_user_profiles_from_messages(
        user_messages,
        seed=seed,
        min_messages_per_user=min_messages_per_user,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        profile_fraction=profile_fraction,
    )


def build_wildchat_user_profiles_from_messages(
    user_messages: dict[str, tuple[WildChatMessage, ...]],
    *,
    seed: int,
    min_messages_per_user: int = 10,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    profile_fraction: float = 0.7,
) -> dict[SplitName, list[WildChatUserProfile]]:
    """Build WildChat profile/query units from preloaded user messages."""
    _validate_profile_builder_args(
        min_messages_per_user=min_messages_per_user,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        profile_fraction=profile_fraction,
    )

    eligible_user_ids = sorted(
        user_id
        for user_id, messages in user_messages.items()
        if len(messages) >= min_messages_per_user
    )
    if len(eligible_user_ids) < 3:
        raise ValueError(
            "Need at least 3 eligible users to build train/val/test splits, "
            f"got {len(eligible_user_ids)}"
        )

    split_user_ids = _stratified_user_split(
        eligible_user_ids=eligible_user_ids,
        user_messages=user_messages,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    profiles: dict[SplitName, list[WildChatUserProfile]] = {
        SplitName.TRAIN: [],
        SplitName.VAL: [],
        SplitName.TEST: [],
    }
    for split, user_ids in split_user_ids.items():
        for user_id in user_ids:
            profile_texts, query_texts = _split_chronological_messages(
                user_messages[user_id],
                profile_fraction=profile_fraction,
            )
            profiles[split].append(
                WildChatUserProfile(
                    author_id=user_id,
                    split=split,
                    profile_texts=profile_texts,
                    query_texts=query_texts,
                )
            )

    return profiles


def _discover_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    supported_suffixes = {".jsonl", ".json", ".parquet"}
    files = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )
    if not files:
        suffixes = ", ".join(sorted(supported_suffixes))
        raise ValueError(f"No supported input files found in {input_path} ({suffixes})")
    return files


def _iter_rows(file_path: Path) -> Iterable[dict[str, Any] | None]:
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl_rows(file_path)
    elif suffix == ".json":
        yield from _iter_json_rows(file_path)
    elif suffix == ".parquet":
        yield from _iter_parquet_rows(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def _iter_jsonl_rows(file_path: Path) -> Iterable[dict[str, Any] | None]:
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                yield None
                continue
            yield row if isinstance(row, dict) else None


def _iter_json_rows(file_path: Path) -> Iterable[dict[str, Any] | None]:
    with file_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if isinstance(loaded, list):
        for row in loaded:
            yield row if isinstance(row, dict) else None
        return

    if isinstance(loaded, dict):
        for key in ("data", "rows", "conversations"):
            value = loaded.get(key)
            if isinstance(value, list):
                for row in value:
                    yield row if isinstance(row, dict) else None
                return
        yield loaded
        return

    yield None


def _iter_parquet_rows(file_path: Path) -> Iterable[dict[str, Any] | None]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Reading parquet files requires pyarrow.") from exc

    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            yield row if isinstance(row, dict) else None


def _extract_user_id(row: dict[str, Any], user_field: str) -> str | None:
    value = _clean_string(row.get(user_field))
    if value is not None:
        return value

    for message in _iter_conversation_messages(row):
        value = _clean_string(message.get(user_field))
        if value is not None:
            return value

    return None


def _present_fields(row: dict[str, Any], candidates: tuple[str, ...]) -> list[str]:
    return [field for field in candidates if field in row]


def _extract_language(row: dict[str, Any]) -> str | None:
    for field in LANGUAGE_FIELD_CANDIDATES:
        value = _clean_string(row.get(field))
        if value is not None:
            return value
    return None


def _is_english_language(language: str | None) -> bool:
    if language is None:
        return False
    normalized = language.strip().lower()
    return normalized in {"en", "eng", "english", "en-us", "en_us", "en-gb", "en_gb"}


def _extract_user_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for message in _iter_conversation_messages(row):
        role = str(message.get("role") or message.get("from") or message.get("speaker") or "")
        content = message.get("content")
        if content is None:
            content = message.get("text")
        if content is None:
            content = message.get("value")
        if _is_user_role(role) and isinstance(content, str) and content.strip():
            texts.append(content.strip())

    for field in ("prompt", "user_message", "user_text"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())

    return texts


def _iter_conversation_messages(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    conversation = row.get("conversation")
    if conversation is None:
        conversation = row.get("messages")

    if not isinstance(conversation, (list, tuple)):
        return

    for message in conversation:
        if isinstance(message, dict):
            yield message


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _is_user_role(role: str) -> bool:
    return role.strip().lower() in {"user", "human", "prompter"}


def _message_sort_key(message: WildChatMessage) -> tuple[str, int, int]:
    return message.timestamp_key, message.row_index, message.message_index


def _timestamp_sort_key(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _validate_profile_builder_args(
    *,
    min_messages_per_user: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    profile_fraction: float,
) -> None:
    if min_messages_per_user < 10:
        raise ValueError("min_messages_per_user must be at least 10")
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


def _stratified_user_split(
    *,
    eligible_user_ids: list[str],
    user_messages: dict[str, tuple[WildChatMessage, ...]],
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[SplitName, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for user_id in eligible_user_ids:
        bucket = _message_count_bucket(len(user_messages[user_id]))
        buckets[bucket].append(user_id)

    split_user_ids: dict[SplitName, list[str]] = {
        SplitName.TRAIN: [],
        SplitName.VAL: [],
        SplitName.TEST: [],
    }
    for bucket_name in sorted(buckets):
        bucket_user_ids = sorted(buckets[bucket_name])
        bucket_rng = random.Random(_stable_user_seed(seed, bucket_name))
        bucket_rng.shuffle(bucket_user_ids)
        bucket_splits = _split_ids_by_fraction(
            bucket_user_ids,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        for split, user_ids in bucket_splits.items():
            split_user_ids[split].extend(user_ids)

    for split in split_user_ids:
        split_user_ids[split].sort()

    _ensure_non_empty_splits(split_user_ids)
    return split_user_ids


def _message_count_bucket(message_count: int) -> str:
    if message_count <= 12:
        return "10_12"
    if message_count <= 16:
        return "13_16"
    if message_count <= 20:
        return "17_20"
    if message_count <= 49:
        return "21_49"
    if message_count <= 99:
        return "50_99"
    if message_count <= 499:
        return "100_499"
    return "500_plus"


def _split_ids_by_fraction(
    ids: list[str],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[SplitName, list[str]]:
    train_count, val_count, _test_count = _fraction_counts(
        total=len(ids),
        fractions=(train_fraction, val_fraction, test_fraction),
    )
    val_end = train_count + val_count
    return {
        SplitName.TRAIN: ids[:train_count],
        SplitName.VAL: ids[train_count:val_end],
        SplitName.TEST: ids[val_end:],
    }


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


def _ensure_non_empty_splits(split_user_ids: dict[SplitName, list[str]]) -> None:
    total_users = sum(len(user_ids) for user_ids in split_user_ids.values())
    if total_users < len(split_user_ids):
        return

    for split, user_ids in split_user_ids.items():
        if user_ids:
            continue
        donor_split = max(split_user_ids, key=lambda candidate: len(split_user_ids[candidate]))
        if len(split_user_ids[donor_split]) <= 1:
            return
        moved_user_id = split_user_ids[donor_split].pop()
        split_user_ids[split].append(moved_user_id)

    for user_ids in split_user_ids.values():
        user_ids.sort()


def _split_chronological_messages(
    messages: tuple[WildChatMessage, ...],
    *,
    profile_fraction: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    profile_count = int(len(messages) * profile_fraction)
    profile_count = min(max(1, profile_count), len(messages) - 1)
    profile_texts = tuple(message.text for message in messages[:profile_count])
    query_texts = tuple(message.text for message in messages[profile_count:])
    return profile_texts, query_texts


def _stable_user_seed(seed: int, user_id: str) -> int:
    digest = sha1(f"{seed}:{user_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _threshold_counts(values: list[int]) -> dict[str, int]:
    return {
        "ge_5": _count_at_least(values, 5),
        "ge_10": _count_at_least(values, 10),
        "ge_20": _count_at_least(values, 20),
        "ge_50": _count_at_least(values, 50),
    }


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


def _format_numeric_summary(name: str, summary: dict[str, float | int | None]) -> str:
    if summary["count"] == 0:
        return f"{name}: no values"
    return (
        f"{name}: count={summary['count']}, min={summary['min']}, "
        f"median={summary['median']}, mean={summary['mean']:.2f}, "
        f"p90={summary['p90']}, max={summary['max']}"
    )


def _format_threshold_counts(name: str, counts: dict[str, int]) -> str:
    return (
        f"{name}: >=5={counts['ge_5']}, >=10={counts['ge_10']}, "
        f">=20={counts['ge_20']}, >=50={counts['ge_50']}"
    )
