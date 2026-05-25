#!/usr/bin/env python
"""Build PAN author-level profile/query files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from identity_verification.datasets.pan import PanAuthorProfile, build_pan_author_profiles
from identity_verification.paths import ensure_dir
from identity_verification.schemas import SplitName


DEFAULT_PAN_DIR = Path("data/raw/pan/pan20-authorship-verification-training-large")
DEFAULT_PAIRS_PATH = DEFAULT_PAN_DIR / "pan20-authorship-verification-training-large.jsonl"
DEFAULT_TRUTH_PATH = DEFAULT_PAN_DIR / "pan20-authorship-verification-training-large-truth.jsonl"
DEFAULT_SEEDS = (13, 21, 42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pan_profiles"))
    parser.add_argument("--summary-dir", type=Path, default=Path("outputs/profiles/pan"))
    parser.add_argument("--min-texts-per-author", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--profile-fraction", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.summary_dir)

    for seed in args.seeds:
        print(f"Building PAN profiles for seed={seed}...", flush=True)
        profiles = build_pan_author_profiles(
            args.pairs,
            args.truth,
            seed=seed,
            min_texts_per_author=args.min_texts_per_author,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            profile_fraction=args.profile_fraction,
        )

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
    profiles: dict[SplitName, list[PanAuthorProfile]],
    seed: int,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for split in (SplitName.TRAIN, SplitName.VAL, SplitName.TEST):
            for profile in profiles[split]:
                row = profile_to_row(profile, seed)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_split_profiles_jsonl(
    output_dir: Path,
    profiles: dict[SplitName, list[PanAuthorProfile]],
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


def profile_to_row(profile: PanAuthorProfile, seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "dataset": "pan",
        "protocol": "pan_profile",
        "author_id": profile.author_id,
        "split": profile.split.value,
        "profile_texts": list(profile.profile_texts),
        "query_texts": list(profile.query_texts),
        "profile_text_count": len(profile.profile_texts),
        "query_text_count": len(profile.query_texts),
        "total_text_count": len(profile.profile_texts) + len(profile.query_texts),
    }


def summarize_profiles(
    *,
    profiles: dict[SplitName, list[PanAuthorProfile]],
    seed: int,
    profiles_path: Path,
    split_profile_paths: dict[SplitName, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_counts = {split.value: len(profiles[split]) for split in profiles}
    split_author_sets = {
        split: {profile.author_id for profile in split_profiles}
        for split, split_profiles in profiles.items()
    }
    overlaps = {
        "train_val": len(split_author_sets[SplitName.TRAIN] & split_author_sets[SplitName.VAL]),
        "train_test": len(split_author_sets[SplitName.TRAIN] & split_author_sets[SplitName.TEST]),
        "val_test": len(split_author_sets[SplitName.VAL] & split_author_sets[SplitName.TEST]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Author splits overlap: {overlaps}")

    return {
        "seed": seed,
        "pairs_path": str(args.pairs),
        "truth_path": str(args.truth),
        "profiles_path": str(profiles_path),
        "split_profile_paths": {
            split.value: str(path) for split, path in split_profile_paths.items()
        },
        "parameters": {
            "min_texts_per_author": args.min_texts_per_author,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "profile_fraction": args.profile_fraction,
        },
        "total_profiles": sum(split_counts.values()),
        "split_counts": split_counts,
        "author_overlap_counts": overlaps,
        "profile_text_counts": {
            split.value: numeric_summary(
                [len(profile.profile_texts) for profile in split_profiles]
            )
            for split, split_profiles in profiles.items()
        },
        "query_text_counts": {
            split.value: numeric_summary([len(profile.query_texts) for profile in split_profiles])
            for split, split_profiles in profiles.items()
        },
        "total_text_counts": {
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
            "excluded_from_model_inputs": ["author_id", "split", "seed"],
        },
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
    print(f"Author overlap counts: {summary['author_overlap_counts']}", flush=True)


if __name__ == "__main__":
    main()
