#!/usr/bin/env python
"""Run one contrastive bi-encoder profile verification experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from identity_verification.datasets.profile_examples import (  # noqa: E402
    DEFAULT_SBERT_HARDNESS_MODEL,
    HARDNESS_MINERS,
    MIXED_REGIME,
    RANDOM_ONLY_REGIME,
    TFIDF_HARDNESS_MINER,
)
from identity_verification.schemas import HardnessLabel  # noqa: E402
from identity_verification.training.contrastive_profile_dataset import (  # noqa: E402
    TEXT_SOURCE_CHOICES,
)
from identity_verification.training.contrastive_profile_runner import (  # noqa: E402
    TEST_REGIMES,
    ContrastiveProfileConfig,
    run_contrastive_profile_experiment,
)
from identity_verification.training.transformer_runner import TEST_SET_ORDER  # noqa: E402


TRAIN_REGIMES = (RANDOM_ONLY_REGIME, MIXED_REGIME)
VAL_REGIMES = (
    HardnessLabel.RANDOM.value,
    HardnessLabel.SEMI_HARD.value,
    HardnessLabel.HARD.value,
    MIXED_REGIME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pan")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--train-regime",
        choices=TRAIN_REGIMES,
        required=True,
        help="random_only baseline or mixed hardness-aware contrastive training.",
    )
    parser.add_argument(
        "--profiles-root",
        type=Path,
        default=Path("data/processed/pan_profiles"),
        help="Directory containing seed_<N>/train.jsonl, val.jsonl, and test.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/experiments/contrastive_profile_transformers"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--hardness-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--train-hardness-miner",
        choices=HARDNESS_MINERS,
        default=TFIDF_HARDNESS_MINER,
        help="Hardness miner used for training negatives.",
    )
    parser.add_argument(
        "--eval-hardness-miner",
        choices=HARDNESS_MINERS,
        default=TFIDF_HARDNESS_MINER,
        help="Hardness miner used for validation and test negatives.",
    )
    parser.add_argument(
        "--hardness-miner-model",
        default=DEFAULT_SBERT_HARDNESS_MODEL,
        help="Transformer model used when a hardness miner is set to sbert.",
    )
    parser.add_argument(
        "--hardness-mining-batch-size",
        type=int,
        default=64,
        help="Transformer embedding batch size for SBERT hardness mining.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--val-regime", choices=VAL_REGIMES, default=None)
    parser.add_argument(
        "--test-regimes",
        nargs="+",
        choices=tuple(TEST_REGIMES),
        default=tuple(TEST_REGIMES),
        help="Subset of test regimes to run. Defaults to all four.",
    )
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--triplets-per-author", type=int, default=16)
    parser.add_argument("--text-source", choices=TEXT_SOURCE_CHOICES, default="all")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-chars-per-text", type=int, default=2000)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument(
        "--hardness-curriculum",
        action="store_true",
        help="For mixed training, warm up with random_only before mixed negatives.",
    )
    parser.add_argument(
        "--curriculum-random-epochs",
        type=int,
        default=1,
        help="Number of initial random_only epochs when curriculum is enabled.",
    )
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-authors", type=int, default=None)
    parser.add_argument("--max-val-authors", type=int, default=None)
    parser.add_argument("--max-test-authors", type=int, default=None)
    parser.add_argument("--no-save-best-model", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_dir = args.profiles_root / f"seed_{args.seed}"
    train_profiles_path = seed_dir / "train.jsonl"
    val_profiles_path = seed_dir / "val.jsonl"
    test_profiles_path = seed_dir / "test.jsonl"
    validate_paths(
        train_profiles_path=train_profiles_path,
        val_profiles_path=val_profiles_path,
        test_profiles_path=test_profiles_path,
    )

    output_dir = (
        args.output_root
        / args.dataset
        / sanitize_model_name(args.model_name)
        / f"seed_{args.seed}"
        / output_regime_name(args)
    )
    config = ContrastiveProfileConfig(
        model_name=args.model_name,
        output_dir=output_dir,
        train_profiles_path=train_profiles_path,
        val_profiles_path=val_profiles_path,
        test_profiles_path=test_profiles_path,
        train_regime=args.train_regime,
        seed=args.seed,
        val_regime=args.val_regime,
        curriculum_regimes=resolve_curriculum_regimes(args),
        triplets_per_author=args.triplets_per_author,
        text_source=args.text_source,
        max_length=args.max_length,
        max_chars_per_text=args.max_chars_per_text,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        triplet_margin=args.triplet_margin,
        fp16=not args.no_fp16,
        num_workers=args.num_workers,
        max_train_authors=args.max_train_authors,
        max_val_authors=args.max_val_authors,
        max_test_authors=args.max_test_authors,
        cache_dir=args.cache_dir,
        hardness_cache_dir=args.hardness_cache_dir,
        train_hardness_miner=args.train_hardness_miner,
        eval_hardness_miner=args.eval_hardness_miner,
        hardness_miner_model=args.hardness_miner_model,
        hardness_mining_batch_size=args.hardness_mining_batch_size,
        local_files_only=args.local_files_only,
        save_best_model=not args.no_save_best_model,
        show_progress=not args.no_progress,
        prefix_lengths=tuple(dict.fromkeys(args.prefix_lengths)),
        test_regimes={
            test_name: TEST_REGIMES[test_name]
            for test_name in args.test_regimes
        },
    )
    result = run_contrastive_profile_experiment(config)
    print_result_summary(result)


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI settings that are easier to catch before loading data."""
    if any(prefix_len <= 0 for prefix_len in args.prefix_lengths):
        raise ValueError("--prefix-lengths must contain only positive integers")
    if args.hardness_curriculum and args.curriculum_random_epochs >= args.num_epochs:
        raise ValueError(
            "--curriculum-random-epochs must be smaller than --num-epochs "
            "so curriculum training reaches mixed negatives"
        )
    if args.hardness_mining_batch_size <= 0:
        raise ValueError("--hardness-mining-batch-size must be positive")


def validate_paths(
    *,
    train_profiles_path: Path,
    val_profiles_path: Path,
    test_profiles_path: Path,
) -> None:
    missing_paths = [
        path
        for path in (train_profiles_path, val_profiles_path, test_profiles_path)
        if not path.is_file()
    ]
    if missing_paths:
        formatted = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required profile split files:\n{formatted}")


def resolve_curriculum_regimes(args: argparse.Namespace) -> tuple[str, ...] | None:
    """Build the epoch-level negative-regime schedule for curriculum training."""
    if not args.hardness_curriculum:
        return None
    if args.train_regime != MIXED_REGIME:
        raise ValueError("--hardness-curriculum is only valid with --train-regime mixed")
    if args.curriculum_random_epochs <= 0:
        raise ValueError("--curriculum-random-epochs must be positive")

    schedule = [
        RANDOM_ONLY_REGIME if epoch <= args.curriculum_random_epochs else MIXED_REGIME
        for epoch in range(1, args.num_epochs + 1)
    ]
    return tuple(schedule)


def output_regime_name(args: argparse.Namespace) -> str:
    """Return the regime directory name used for outputs."""
    regime_name = (
        f"{args.train_regime}_curriculum"
        if args.hardness_curriculum
        else args.train_regime
    )
    suffixes: list[str] = []
    if args.train_hardness_miner != TFIDF_HARDNESS_MINER:
        suffixes.append(f"train-{miner_suffix(args.train_hardness_miner, args)}")
    if args.eval_hardness_miner != TFIDF_HARDNESS_MINER:
        suffixes.append(f"eval-{miner_suffix(args.eval_hardness_miner, args)}")
    if not suffixes:
        return regime_name
    return "__".join((regime_name, *suffixes))


def miner_suffix(miner: str, args: argparse.Namespace) -> str:
    """Return a compact miner label for output directories."""
    if miner == TFIDF_HARDNESS_MINER:
        return miner
    return f"{miner}-{sanitize_model_name(args.hardness_miner_model)}"


def sanitize_model_name(model_name: str) -> str:
    """Make a Hugging Face model name safe for output directories."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("_")


def print_result_summary(result: dict) -> None:
    print("\nContrastive experiment complete")
    print(f"  model: {result['model_name']}")
    print(f"  seed: {result['seed']}")
    print(f"  device: {result['device']}")
    print(f"  best_epoch: {result['best_epoch']}")
    print(f"  best_threshold: {result['best_threshold']:.6f}")
    print("  test metrics:")
    for test_name in sorted(result["test_results"], key=test_set_sort_key):
        overall = result["test_results"][test_name]["overall"]
        roc_auc = format_metric(overall.get("roc_auc"))
        eer = format_metric(overall.get("eer"))
        f1 = format_metric(overall.get("f1"))
        print(f"    {test_name}: ROC-AUC={roc_auc} EER={eer} F1={f1}")


def format_metric(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.6f}"


def test_set_sort_key(test_name: str) -> tuple[int, str]:
    if test_name in TEST_SET_ORDER:
        return (TEST_SET_ORDER.index(test_name), test_name)
    return (len(TEST_SET_ORDER), test_name)


if __name__ == "__main__":
    main()
