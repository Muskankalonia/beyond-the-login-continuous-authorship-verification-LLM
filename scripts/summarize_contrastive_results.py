#!/usr/bin/env python
"""Summarize contrastive profile experiment result files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from identity_verification.evaluation.profile_metrics import summarize_metric_runs  # noqa: E402


OVERALL_METRICS = (
    "roc_auc",
    "pr_auc",
    "f1",
    "eer",
    "far_at_frr_1pct",
    "tar_at_far_1pct",
    "brier_score",
    "ece",
)
GAP_METRICS = ("roc_auc", "pr_auc", "f1", "eer_increase")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/experiments/contrastive_profile_transformers"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/summaries/contrastive_profile_transformers"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_paths = sorted(args.results_root.glob("*/*/seed_*/*/result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No result.json files found under {args.results_root}")

    rows = [
        row
        for path in result_paths
        for row in flatten_result(path)
    ]
    summary = build_summary(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_dir / "runs.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary["summary_rows"])
    write_json(args.output_dir / "summary.json", summary)

    print(f"Wrote {len(rows)} run rows")
    print(f"  {args.output_dir / 'runs.csv'}")
    print(f"  {args.output_dir / 'summary.csv'}")
    print(f"  {args.output_dir / 'summary.json'}")


def flatten_result(path: Path) -> dict[str, Any]:
    """Flatten one result.json into one row per test regime."""
    result = json.loads(path.read_text(encoding="utf-8"))
    dataset, model_name, seed_name, train_regime = path.parts[-5:-1]
    seed = int(seed_name.removeprefix("seed_"))
    rows: list[dict[str, Any]] = []
    base = {
        "dataset": dataset,
        "model_name": model_name,
        "seed": seed,
        "train_regime": train_regime,
        "best_epoch": result.get("best_epoch"),
        "best_threshold": result.get("best_threshold"),
        "device": result.get("device"),
        "result_path": str(path),
    }
    training_config = result.get("training_config", {})
    for key in (
        "val_regime",
        "num_epochs",
        "triplets_per_author",
        "train_batch_size",
        "eval_batch_size",
        "max_length",
        "max_chars_per_text",
    ):
        base[f"config_{key}"] = training_config.get(key)

    for test_name, metrics in result.get("test_results", {}).items():
        overall = metrics.get("overall", {})
        row = {**base, "test_regime": test_name}
        for metric_name in OVERALL_METRICS:
            row[metric_name] = overall.get(metric_name)
        for prefix_len, prefix_metrics in metrics.get("by_prefix", {}).items():
            row[f"auc_at_{prefix_len}"] = prefix_metrics.get("roc_auc")
            row[f"eer_at_{prefix_len}"] = prefix_metrics.get("eer")
        operating = metrics.get("profile_operating_metrics", {})
        row["imposter_catch_rate_at_max_prefix"] = operating.get(
            "imposter_catch_rate_at_max_prefix"
        )
        row["genuine_survival_rate_at_max_prefix"] = operating.get(
            "genuine_survival_rate_at_max_prefix"
        )
        row["median_messages_to_detect"] = operating.get("median_messages_to_detect")
        row["p90_messages_to_detect"] = operating.get("p90_messages_to_detect")
        rows.append(row)

    gap_row = {**base, "test_regime": "gap_random_minus_hard"}
    for metric_name in GAP_METRICS:
        gap_row[metric_name] = result.get("robustness_gaps", {}).get("overall", {}).get(
            metric_name
        )
    rows.append(gap_row)
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate mean/std rows grouped by dataset/model/regime/test."""
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["model_name"]),
            str(row["train_regime"]),
            str(row["test_regime"]),
        )
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, model_name, train_regime, test_regime), group_rows in sorted(groups.items()):
        metric_names = GAP_METRICS if test_regime == "gap_random_minus_hard" else OVERALL_METRICS
        metric_summary = summarize_metric_runs(group_rows, metric_names=metric_names)
        summary_row: dict[str, Any] = {
            "dataset": dataset,
            "model_name": model_name,
            "train_regime": train_regime,
            "test_regime": test_regime,
            "seed_count": len({row["seed"] for row in group_rows}),
        }
        for metric_name, stats in metric_summary.items():
            summary_row[f"{metric_name}_mean"] = stats["mean"]
            summary_row[f"{metric_name}_std"] = stats["std"]
            summary_row[f"{metric_name}_count"] = stats["count"]
        summary_rows.append(summary_row)

    return {"runs": rows, "summary_rows": summary_rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
