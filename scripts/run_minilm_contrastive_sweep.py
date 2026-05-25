#!/usr/bin/env python
"""Run the MiniLM contrastive profile sweep for PAN and WildChat."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DATASETS = ("pan", "wildchat")
DEFAULT_SEEDS = (13, 21, 42)
DEFAULT_REGIMES = ("random_only", "mixed")
TFIDF_HARDNESS_MINER = "tfidf"
HARDNESS_MINERS = (TFIDF_HARDNESS_MINER, "sbert")
DEFAULT_HARDNESS_MINER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROFILE_ROOTS = {
    "pan": Path("data/processed/pan_profiles"),
    "wildchat": Path("data/processed/wildchat_profiles"),
}
HARDNESS_CACHE_DIRS = {
    "pan": Path("data/processed/hardness_indexes/pan"),
    "wildchat": Path("data/processed/hardness_indexes/wildchat"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--regimes", nargs="+", choices=DEFAULT_REGIMES, default=list(DEFAULT_REGIMES))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/experiments/contrastive_profile_transformers"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/model_cache"))
    parser.add_argument(
        "--train-hardness-miner",
        choices=HARDNESS_MINERS,
        default=TFIDF_HARDNESS_MINER,
    )
    parser.add_argument(
        "--eval-hardness-miner",
        choices=HARDNESS_MINERS,
        default=TFIDF_HARDNESS_MINER,
    )
    parser.add_argument(
        "--hardness-miner-model",
        default=DEFAULT_HARDNESS_MINER_MODEL,
    )
    parser.add_argument("--hardness-mining-batch-size", type=int, default=64)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("outputs/summaries/contrastive_profile_transformers"),
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--triplets-per-author", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-chars-per-text", type=int, default=2000)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun experiments even when result.json already exists.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not refresh the contrastive summary after the sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_profile_roots(args.datasets, args.seeds)

    for dataset in args.datasets:
        for seed in args.seeds:
            for regime in args.regimes:
                result_path = experiment_result_path(
                    output_root=args.output_root,
                    dataset=dataset,
                    model_name=args.model_name,
                    seed=seed,
                    regime=output_regime_name(args, regime),
                )
                if result_path.is_file() and not args.overwrite:
                    print(f"[sweep] skipping existing result: {result_path}")
                    continue

                command = build_experiment_command(
                    args,
                    dataset=dataset,
                    seed=seed,
                    regime=regime,
                )
                run_command(command, dry_run=args.dry_run)

    if not args.no_summary:
        summary_command = [
            sys.executable,
            "scripts/summarize_contrastive_results.py",
            "--results-root",
            str(args.output_root),
            "--output-dir",
            str(args.summary_dir),
        ]
        run_command(summary_command, dry_run=args.dry_run)


def validate_profile_roots(datasets: list[str], seeds: list[int]) -> None:
    missing_paths: list[Path] = []
    for dataset in datasets:
        profiles_root = PROFILE_ROOTS[dataset]
        for seed in seeds:
            seed_dir = profiles_root / f"seed_{seed}"
            for split_name in ("train", "val", "test"):
                path = seed_dir / f"{split_name}.jsonl"
                if not path.is_file():
                    missing_paths.append(path)
    if missing_paths:
        formatted = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required profile split files:\n{formatted}")


def build_experiment_command(
    args: argparse.Namespace,
    *,
    dataset: str,
    seed: int,
    regime: str,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_contrastive_profile_experiment.py",
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--model-name",
        args.model_name,
        "--train-regime",
        regime,
        "--profiles-root",
        str(PROFILE_ROOTS[dataset]),
        "--output-root",
        str(args.output_root),
        "--cache-dir",
        str(args.cache_dir),
        "--hardness-cache-dir",
        str(HARDNESS_CACHE_DIRS[dataset]),
        "--train-hardness-miner",
        args.train_hardness_miner,
        "--eval-hardness-miner",
        args.eval_hardness_miner,
        "--hardness-miner-model",
        args.hardness_miner_model,
        "--hardness-mining-batch-size",
        str(args.hardness_mining_batch_size),
        "--num-epochs",
        str(args.num_epochs),
        "--triplets-per-author",
        str(args.triplets_per_author),
        "--train-batch-size",
        str(args.train_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--max-length",
        str(args.max_length),
        "--max-chars-per-text",
        str(args.max_chars_per_text),
    ]
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def output_regime_name(args: argparse.Namespace, regime: str) -> str:
    """Match the output directory naming used by the single-run CLI."""
    suffixes: list[str] = []
    if args.train_hardness_miner != TFIDF_HARDNESS_MINER:
        suffixes.append(
            f"train-{miner_suffix(args.train_hardness_miner, args.hardness_miner_model)}"
        )
    if args.eval_hardness_miner != TFIDF_HARDNESS_MINER:
        suffixes.append(
            f"eval-{miner_suffix(args.eval_hardness_miner, args.hardness_miner_model)}"
        )
    if not suffixes:
        return regime
    return "__".join((regime, *suffixes))


def miner_suffix(miner: str, miner_model: str) -> str:
    """Return a compact miner label for output directories."""
    if miner == TFIDF_HARDNESS_MINER:
        return miner
    return f"{miner}-{sanitize_model_name(miner_model)}"


def experiment_result_path(
    *,
    output_root: Path,
    dataset: str,
    model_name: str,
    seed: int,
    regime: str,
) -> Path:
    return (
        output_root
        / dataset
        / sanitize_model_name(model_name)
        / f"seed_{seed}"
        / regime
        / "result.json"
    )


def sanitize_model_name(model_name: str) -> str:
    """Match the output directory naming used by the single-run CLI."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("_")


def run_command(command: list[str], *, dry_run: bool) -> None:
    printable = " ".join(command)
    print(f"[sweep] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
