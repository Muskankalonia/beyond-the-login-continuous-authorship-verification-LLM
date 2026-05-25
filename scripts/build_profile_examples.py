#!/usr/bin/env python
"""Build raw-text profile/query examples from profile split files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from identity_verification.datasets.profile_examples import (  # noqa: E402
    build_hardness_index,
    iter_profile_examples_for_regime,
    load_profile_records,
    regimes_for_split,
)
from identity_verification.paths import ensure_dir  # noqa: E402
from identity_verification.schemas import SplitName  # noqa: E402


DEFAULT_SEEDS = (13, 21, 42)
DEFAULT_SPLITS = (SplitName.TRAIN, SplitName.VAL, SplitName.TEST)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pan")
    parser.add_argument("--profiles-root", type=Path, default=Path("data/processed/pan_profiles"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/profile_examples/pan"),
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("outputs/profile_examples/pan"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--splits",
        choices=[split.value for split in DEFAULT_SPLITS],
        nargs="+",
        default=[split.value for split in DEFAULT_SPLITS],
    )
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--miner", choices=["tfidf"], default="tfidf")
    parser.add_argument(
        "--max-profiles-per-split",
        type=int,
        default=None,
        help="Optional smoke-test limit. Leave unset for full generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    ensure_dir(args.summary_dir)

    for seed in args.seeds:
        print(f"Building profile examples for dataset={args.dataset}, seed={seed}", flush=True)
        seed_summary = build_seed_examples(args=args, seed=seed)
        summary_path = args.summary_dir / f"seed_{seed}_summary.json"
        summary_path.write_text(
            json.dumps(seed_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Saved summary: {summary_path}", flush=True)


def build_seed_examples(*, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    seed_input_dir = args.profiles_root / f"seed_{seed}"
    seed_output_dir = ensure_dir(args.output_root / f"seed_{seed}")

    split_summaries: dict[str, Any] = {}
    for split_value in args.splits:
        split = SplitName(split_value)
        profiles_path = seed_input_dir / f"{split.value}.jsonl"
        profiles = load_profile_records(profiles_path)
        if args.max_profiles_per_split is not None:
            profiles = profiles[: args.max_profiles_per_split]

        print(
            f"  split={split.value}: loaded {len(profiles)} profiles from {profiles_path}",
            flush=True,
        )
        hardness_index = build_hardness_index(
            profiles,
            prefix_lengths=tuple(args.prefix_lengths),
            miner=args.miner,
        )

        split_summaries[split.value] = {}
        for regime in regimes_for_split(split):
            set_name = f"{split.value}_{regime}"
            output_path = seed_output_dir / f"{set_name}.jsonl"
            examples = iter_profile_examples_for_regime(
                profiles,
                dataset=args.dataset,
                seed=seed,
                split=split,
                regime=regime,
                prefix_lengths=tuple(args.prefix_lengths),
                hardness_index=hardness_index,
            )
            summary = write_and_summarize_examples(output_path, examples)
            split_summaries[split.value][set_name] = summary
            print(
                "    saved "
                f"{set_name}: rows={summary['row_count']}, "
                f"labels={summary['label_counts']}, "
                f"negative_hardness={summary['negative_hardness_counts']}",
                flush=True,
            )

    return {
        "dataset": args.dataset,
        "seed": seed,
        "profiles_root": str(args.profiles_root),
        "output_root": str(args.output_root),
        "prefix_lengths": list(args.prefix_lengths),
        "miner": args.miner,
        "max_profiles_per_split": args.max_profiles_per_split,
        "splits": split_summaries,
    }


def write_and_summarize_examples(
    output_path: Path,
    examples: Any,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = ExampleSummary()
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            summary.update(example)
            handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    return summary.to_dict(output_path=output_path)


class ExampleSummary:
    def __init__(self) -> None:
        self.row_count = 0
        self.label_counts: Counter[str] = Counter()
        self.hardness_counts: Counter[str] = Counter()
        self.negative_hardness_counts: Counter[str] = Counter()
        self.prefix_counts: Counter[str] = Counter()
        self.negative_self_matches = 0
        self.missing_raw_text_rows = 0

    def update(self, example: dict[str, Any]) -> None:
        self.row_count += 1
        self.label_counts[str(example["label"])] += 1
        self.hardness_counts[str(example["hardness"])] += 1
        self.prefix_counts[str(example["prefix_len"])] += 1

        if example["label"] == 0:
            self.negative_hardness_counts[str(example["hardness"])] += 1
            if example["profile_author_id"] == example["query_author_id"]:
                self.negative_self_matches += 1

        if not example["profile_text"] or not example["query_text"] or not example["query_texts"]:
            self.missing_raw_text_rows += 1

    def to_dict(
        self,
        *,
        output_path: Path,
    ) -> dict[str, Any]:
        if self.negative_self_matches:
            raise ValueError(f"{output_path} contains negative self matches")
        if self.missing_raw_text_rows:
            raise ValueError(f"{output_path} contains rows with missing raw text")

        return {
            "output_path": str(output_path),
            "row_count": self.row_count,
            "label_counts": dict(sorted(self.label_counts.items())),
            "hardness_counts": dict(sorted(self.hardness_counts.items())),
            "negative_hardness_counts": dict(sorted(self.negative_hardness_counts.items())),
            "prefix_counts": dict(sorted(self.prefix_counts.items())),
            "negative_self_matches": self.negative_self_matches,
            "missing_raw_text_rows": self.missing_raw_text_rows,
        }


def summarize_examples(
    examples: list[dict[str, Any]],
    *,
    output_path: Path,
) -> dict[str, Any]:
    summary = ExampleSummary()
    for example in examples:
        summary.update(example)
    return summary.to_dict(output_path=output_path)


if __name__ == "__main__":
    main()
