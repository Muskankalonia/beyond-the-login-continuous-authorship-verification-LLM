#!/usr/bin/env python
"""Build WildChat user-level profile/query files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from identity_verification.datasets.wildchat import (  # noqa: E402
    MODEL_EXCLUDED_FIELDS,
    WildChatUserProfile,
    build_wildchat_user_profiles_from_messages,
    load_wildchat_user_messages,
)
from identity_verification.paths import ensure_dir  # noqa: E402
from identity_verification.schemas import SplitName  # noqa: E402


DEFAULT_INPUT_PATH = Path("data/raw/wildchat")
DEFAULT_SEEDS = (13, 21, 42)
MIN_PROFILE_MESSAGES = 7
MIN_QUERY_MESSAGES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/wildchat_profiles"))
    parser.add_argument("--summary-dir", type=Path, default=Path("outputs/profiles/wildchat"))
    parser.add_argument("--user-field", default="hashed_ip")
    parser.add_argument("--language", default="English")
    parser.add_argument("--min-messages-per-user", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--profile-fraction", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.summary_dir)

    print("Loading WildChat user messages once...", flush=True)
    user_messages = load_wildchat_user_messages(
        args.input,
        user_field=args.user_field,
        language=args.language,
    )
    eligible_user_count = sum(
        len(messages) >= args.min_messages_per_user
        for messages in user_messages.values()
    )
    print(
        "Loaded users: "
        f"total={len(user_messages)}, "
        f"eligible>={args.min_messages_per_user}={eligible_user_count}",
        flush=True,
    )

    for seed in args.seeds:
        print(f"Building WildChat profiles for seed={seed}...", flush=True)
        profiles = build_wildchat_user_profiles_from_messages(
            user_messages,
            seed=seed,
            min_messages_per_user=args.min_messages_per_user,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            profile_fraction=args.profile_fraction,
        )
        validate_profiles(profiles, args=args)

        seed_output_dir = ensure_dir(args.output_dir / f"seed_{seed}")
        profiles_path = seed_output_dir / "profiles.jsonl"
        write_profiles_jsonl(profiles_path, profiles, seed)
        split_profile_paths = write_split_profiles_jsonl(seed_output_dir, profiles, seed)

        summary = summarize_profiles(
            profiles=profiles,
            seed=seed,
            profiles_path=profiles_path,
            split_profile_paths=split_profile_paths,
            args=args,
            loaded_user_count=len(user_messages),
            eligible_user_count=eligible_user_count,
        )
        summary_path = args.summary_dir / f"seed_{seed}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        print_seed_summary(summary)
        print(f"Saved profiles: {profiles_path}", flush=True)
        for split in (SplitName.TRAIN, SplitName.VAL, SplitName.TEST):
            print(f"Saved {split.value} profiles: {split_profile_paths[split]}", flush=True)
        print(f"Saved summary: {summary_path}", flush=True)


def write_profiles_jsonl(
    path: Path,
    profiles: dict[SplitName, list[WildChatUserProfile]],
    seed: int,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for split in (SplitName.TRAIN, SplitName.VAL, SplitName.TEST):
            for profile in profiles[split]:
                row = profile_to_row(profile, seed)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_split_profiles_jsonl(
    output_dir: Path,
    profiles: dict[SplitName, list[WildChatUserProfile]],
    seed: int,
) -> dict[SplitName, Path]:
    split_paths: dict[SplitName, Path] = {}
    for split in (SplitName.TRAIN, SplitName.VAL, SplitName.TEST):
        split_path = output_dir / f"{split.value}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for profile in profiles[split]:
                row = profile_to_row(profile, seed)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        split_paths[split] = split_path
    return split_paths


def profile_to_row(profile: WildChatUserProfile, seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "dataset": "wildchat",
        "protocol": "wildchat_profile",
        "author_id": profile.author_id,
        "split": profile.split.value,
        "profile_texts": list(profile.profile_texts),
        "query_texts": list(profile.query_texts),
        "profile_message_count": len(profile.profile_texts),
        "query_message_count": len(profile.query_texts),
        "total_message_count": len(profile.profile_texts) + len(profile.query_texts),
    }


def validate_profiles(
    profiles: dict[SplitName, list[WildChatUserProfile]],
    *,
    args: argparse.Namespace,
) -> None:
    split_author_sets = {
        split: {profile.author_id for profile in split_profiles}
        for split, split_profiles in profiles.items()
    }
    overlaps = overlap_counts(split_author_sets)
    if any(overlaps.values()):
        raise ValueError(f"WildChat user splits overlap: {overlaps}")

    forbidden_row_keys = set(MODEL_EXCLUDED_FIELDS)
    for split, split_profiles in profiles.items():
        for profile in split_profiles:
            total_messages = len(profile.profile_texts) + len(profile.query_texts)
            if total_messages < args.min_messages_per_user:
                raise ValueError(
                    f"{split.value} profile {profile.author_id} has only "
                    f"{total_messages} messages"
                )
            if len(profile.profile_texts) < MIN_PROFILE_MESSAGES:
                raise ValueError(
                    f"{split.value} profile {profile.author_id} has only "
                    f"{len(profile.profile_texts)} profile messages"
                )
            if len(profile.query_texts) < MIN_QUERY_MESSAGES:
                raise ValueError(
                    f"{split.value} profile {profile.author_id} has only "
                    f"{len(profile.query_texts)} query messages"
                )

            row_keys = set(profile_to_row(profile, seed=0))
            forbidden_written_keys = row_keys & forbidden_row_keys
            if forbidden_written_keys:
                raise ValueError(
                    "WildChat profile rows contain forbidden model-input metadata keys: "
                    f"{sorted(forbidden_written_keys)}"
                )


def summarize_profiles(
    *,
    profiles: dict[SplitName, list[WildChatUserProfile]],
    seed: int,
    profiles_path: Path,
    split_profile_paths: dict[SplitName, Path],
    args: argparse.Namespace,
    loaded_user_count: int,
    eligible_user_count: int,
) -> dict[str, Any]:
    split_counts = {split.value: len(profiles[split]) for split in profiles}
    split_author_sets = {
        split: {profile.author_id for profile in split_profiles}
        for split, split_profiles in profiles.items()
    }
    overlaps = overlap_counts(split_author_sets)
    if any(overlaps.values()):
        raise ValueError(f"WildChat user splits overlap: {overlaps}")

    return {
        "seed": seed,
        "input_path": str(args.input),
        "profiles_path": str(profiles_path),
        "split_profile_paths": {
            split.value: str(path) for split, path in split_profile_paths.items()
        },
        "parameters": {
            "user_field": args.user_field,
            "language": args.language,
            "min_messages_per_user": args.min_messages_per_user,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "profile_fraction": args.profile_fraction,
        },
        "total_profiles": sum(split_counts.values()),
        "loaded_user_count": loaded_user_count,
        "eligible_user_count": eligible_user_count,
        "split_counts": split_counts,
        "user_overlap_counts": overlaps,
        "profile_message_counts": {
            split.value: numeric_summary(
                [len(profile.profile_texts) for profile in split_profiles]
            )
            for split, split_profiles in profiles.items()
        },
        "query_message_counts": {
            split.value: numeric_summary(
                [len(profile.query_texts) for profile in split_profiles]
            )
            for split, split_profiles in profiles.items()
        },
        "total_message_counts": {
            split.value: numeric_summary(
                [
                    len(profile.profile_texts) + len(profile.query_texts)
                    for profile in split_profiles
                ]
            )
            for split, split_profiles in profiles.items()
        },
        "model_input_policy": {
            "allowed_model_inputs": ["profile_texts", "query_texts"],
            "excluded_from_model_inputs": [
                "author_id",
                "split",
                "seed",
                *MODEL_EXCLUDED_FIELDS,
            ],
            "language_use": "filtering only; never fed to model",
            "timestamp_use": "chronological ordering only; never fed to model",
        },
    }


def overlap_counts(split_author_sets: dict[SplitName, set[str]]) -> dict[str, int]:
    return {
        "train_val": len(split_author_sets[SplitName.TRAIN] & split_author_sets[SplitName.VAL]),
        "train_test": len(split_author_sets[SplitName.TRAIN] & split_author_sets[SplitName.TEST]),
        "val_test": len(split_author_sets[SplitName.VAL] & split_author_sets[SplitName.TEST]),
    }


def numeric_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}

    sorted_values = sorted(values)
    count = len(sorted_values)
    midpoint = count // 2
    if count % 2:
        median = float(sorted_values[midpoint])
    else:
        median = (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2

    return {
        "count": count,
        "min": sorted_values[0],
        "mean": sum(sorted_values) / count,
        "median": median,
        "max": sorted_values[-1],
    }


def print_seed_summary(summary: dict[str, Any]) -> None:
    split_counts = summary["split_counts"]
    print(
        "Split counts: "
        f"train={split_counts['train']}, "
        f"val={split_counts['val']}, "
        f"test={split_counts['test']}, "
        f"total={summary['total_profiles']}",
        flush=True,
    )
    print(f"User overlap counts: {summary['user_overlap_counts']}", flush=True)


if __name__ == "__main__":
    main()
