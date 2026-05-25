#!/usr/bin/env python
"""Run one profile-based transformer verification experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from identity_verification.training.transformer_runner import (  # noqa: E402
    TEST_SET_ORDER,
    TransformerExperimentConfig,
    run_transformer_experiment,
)


TRAIN_REGIMES = ("random_only", "mixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pan")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--train-regime",
        choices=TRAIN_REGIMES,
        required=True,
        help="random_only baseline or mixed hardness-aware training.",
    )
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=Path("data/processed/profile_examples/pan"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/experiments/profile_transformers"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-profile-chars", type=int, default=6000)
    parser.add_argument("--max-query-chars", type=int, default=3000)
    parser.add_argument("--max-query-chars-per-text", type=int, default=1000)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--hardness-curriculum",
        action="store_true",
        help=(
            "For mixed training, warm up on train_random_only before switching "
            "to train_mixed. Results are written under mixed_curriculum."
        ),
    )
    parser.add_argument(
        "--curriculum-random-epochs",
        type=int,
        default=1,
        help="Number of initial epochs to train on train_random_only when curriculum is enabled.",
    )
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--no-save-best-model", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_dir = args.examples_root / f"seed_{args.seed}"
    train_path = seed_dir / f"train_{args.train_regime}.jsonl"
    val_path = seed_dir / f"val_{args.train_regime}.jsonl"
    curriculum_train_paths = resolve_curriculum_train_paths(
        args=args,
        seed_dir=seed_dir,
        train_path=train_path,
    )
    test_paths = {
        test_name: seed_dir / f"{test_name}.jsonl"
        for test_name in TEST_SET_ORDER
    }

    validate_paths(
        train_path=train_path,
        val_path=val_path,
        test_paths=test_paths,
        curriculum_train_paths=curriculum_train_paths,
    )

    output_dir = (
        args.output_root
        / args.dataset
        / sanitize_model_name(args.model_name)
        / f"seed_{args.seed}"
        / output_regime_name(args)
    )
    config = TransformerExperimentConfig(
        model_name=args.model_name,
        output_dir=output_dir,
        train_path=train_path,
        val_path=val_path,
        curriculum_train_paths=curriculum_train_paths,
        test_paths=test_paths,
        seed=args.seed,
        max_length=args.max_length,
        max_profile_chars=args.max_profile_chars,
        max_query_chars=args.max_query_chars,
        max_query_chars_per_text=args.max_query_chars_per_text,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        fp16=not args.no_fp16,
        num_workers=args.num_workers,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
        max_train_steps=args.max_train_steps,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        save_best_model=not args.no_save_best_model,
        show_progress=not args.no_progress,
    )
    result = run_transformer_experiment(config)
    print_result_summary(result)


def validate_paths(
    *,
    train_path: Path,
    val_path: Path,
    test_paths: dict[str, Path],
    curriculum_train_paths: tuple[Path, ...] | None = None,
) -> None:
    missing_paths = [
        path
        for path in [
            train_path,
            val_path,
            *test_paths.values(),
            *(curriculum_train_paths or ()),
        ]
        if not path.is_file()
    ]
    if missing_paths:
        formatted = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required example files:\n{formatted}")


def resolve_curriculum_train_paths(
    *,
    args: argparse.Namespace,
    seed_dir: Path,
    train_path: Path,
) -> tuple[Path, ...] | None:
    """Build the epoch-level training schedule for hardness curriculum."""
    if not args.hardness_curriculum:
        return None
    if args.train_regime != "mixed":
        raise ValueError("--hardness-curriculum is only valid with --train-regime mixed")
    if args.curriculum_random_epochs <= 0:
        raise ValueError("--curriculum-random-epochs must be positive")

    random_path = seed_dir / "train_random_only.jsonl"
    schedule = [
        random_path if epoch <= args.curriculum_random_epochs else train_path
        for epoch in range(1, args.num_epochs + 1)
    ]
    return tuple(schedule)


def output_regime_name(args: argparse.Namespace) -> str:
    """Return the regime directory name used for outputs."""
    if args.hardness_curriculum:
        return f"{args.train_regime}_curriculum"
    return args.train_regime


def sanitize_model_name(model_name: str) -> str:
    """Make a Hugging Face model name safe for output directories."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("_")


def print_result_summary(result: dict) -> None:
    print("\nExperiment complete")
    print(f"  model: {result['model_name']}")
    print(f"  seed: {result['seed']}")
    print(f"  device: {result['device']}")
    print(f"  best_epoch: {result['best_epoch']}")
    print(f"  best_threshold: {result['best_threshold']:.6f}")
    print("  test ROC-AUC:")
    for test_name, metrics in result["test_results"].items():
        roc_auc = metrics["overall"].get("roc_auc")
        roc_text = "None" if roc_auc is None else f"{roc_auc:.6f}"
        print(f"    {test_name}: {roc_text}")


if __name__ == "__main__":
    main()
