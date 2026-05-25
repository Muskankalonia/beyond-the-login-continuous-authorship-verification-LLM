"""Profile-verification metric helpers for transformer experiments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

import numpy as np

from identity_verification.metrics import evaluate_binary_scores


@dataclass(frozen=True)
class ProfilePrediction:
    """One scored profile/query example."""

    example_id: str
    label: int
    score: float
    prefix_len: int
    hardness: str
    metadata: dict[str, Any] | None = None


def evaluate_profile_predictions(
    predictions: Sequence[ProfilePrediction],
    *,
    threshold: float,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Compute overall, prefix, and early-detection metrics."""
    if not predictions:
        raise ValueError("No predictions provided")

    result: dict[str, Any] = {
        "overall": _evaluate_subset(predictions, threshold=threshold, ece_bins=ece_bins),
        "by_prefix": {},
        "profile_operating_metrics": profile_operating_metrics(
            predictions,
            threshold=threshold,
        ),
    }

    for prefix_len in sorted({prediction.prefix_len for prediction in predictions}):
        prefix_predictions = [
            prediction for prediction in predictions if prediction.prefix_len == prefix_len
        ]
        result["by_prefix"][str(prefix_len)] = _evaluate_subset(
            prefix_predictions,
            threshold=threshold,
            ece_bins=ece_bins,
        )

    return result


def profile_operating_metrics(
    predictions: Sequence[ProfilePrediction],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Compute profile-specific threshold metrics from prefix scores."""
    prefix_lengths = sorted({prediction.prefix_len for prediction in predictions})
    max_prefix = max(prefix_lengths)

    max_prefix_predictions = [
        prediction for prediction in predictions if prediction.prefix_len == max_prefix
    ]
    imposters_at_max = [
        prediction for prediction in max_prefix_predictions if prediction.label == 0
    ]
    genuines_at_max = [
        prediction for prediction in max_prefix_predictions if prediction.label == 1
    ]

    imposter_catch_rate = _safe_mean(
        [prediction.score < threshold for prediction in imposters_at_max]
    )
    genuine_survival_rate = _safe_mean(
        [prediction.score >= threshold for prediction in genuines_at_max]
    )

    detection_times = imposter_messages_to_detect(
        predictions,
        threshold=threshold,
    )
    detected_times = [
        detection_time
        for detection_time in detection_times
        if detection_time is not None
    ]

    return {
        "max_prefix_len": max_prefix,
        "imposter_catch_rate_at_max_prefix": imposter_catch_rate,
        "genuine_survival_rate_at_max_prefix": genuine_survival_rate,
        "median_messages_to_detect": _percentile(detected_times, 50),
        "p90_messages_to_detect": _percentile(detected_times, 90),
        "undetected_imposter_count": len(detection_times) - len(detected_times),
        "total_imposter_streams": len(detection_times),
    }


def imposter_messages_to_detect(
    predictions: Sequence[ProfilePrediction],
    *,
    threshold: float,
) -> list[int | None]:
    """Return coarse messages-to-detect values for imposter streams.

    With current examples this is based on prefixes 1 and 3, so possible values
    are 1, 3, or None when the imposter is not detected by prefix 3.
    """
    grouped = _group_prediction_streams(predictions)
    detection_times: list[int | None] = []
    for stream_predictions in grouped.values():
        labels = {prediction.label for prediction in stream_predictions}
        if labels != {0}:
            continue

        detection_time: int | None = None
        for prediction in sorted(stream_predictions, key=lambda item: item.prefix_len):
            if prediction.score < threshold:
                detection_time = prediction.prefix_len
                break
        detection_times.append(detection_time)

    return detection_times


def robustness_gap(
    random_metrics: dict[str, Any],
    hard_metrics: dict[str, Any],
    *,
    metric_name: str = "roc_auc",
) -> float | None:
    """Return random-test minus hard-test performance for one metric."""
    random_value = random_metrics.get(metric_name)
    hard_value = hard_metrics.get(metric_name)
    if random_value is None or hard_value is None:
        return None
    return float(random_value - hard_value)


def summarize_metric_runs(
    runs: Iterable[dict[str, Any]],
    *,
    metric_names: Sequence[str],
) -> dict[str, dict[str, float | int | None]]:
    """Aggregate flat metric dictionaries as mean +/- population std."""
    runs = list(runs)
    summary: dict[str, dict[str, float | int | None]] = {}
    for metric_name in metric_names:
        values = [
            float(run[metric_name])
            for run in runs
            if run.get(metric_name) is not None
        ]
        if not values:
            summary[metric_name] = {"count": 0, "mean": None, "std": None}
            continue
        summary[metric_name] = {
            "count": len(values),
            "mean": float(mean(values)),
            "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        }
    return summary


def _evaluate_subset(
    predictions: Sequence[ProfilePrediction],
    *,
    threshold: float,
    ece_bins: int,
) -> dict[str, Any]:
    return evaluate_binary_scores(
        [prediction.label for prediction in predictions],
        [prediction.score for prediction in predictions],
        threshold=threshold,
        ece_bins=ece_bins,
    )


def _group_prediction_streams(
    predictions: Sequence[ProfilePrediction],
) -> dict[str, list[ProfilePrediction]]:
    grouped: dict[str, list[ProfilePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[_stream_key(prediction)].append(prediction)
    return grouped


def _stream_key(prediction: ProfilePrediction) -> str:
    metadata = prediction.metadata or {}
    profile_author_id = metadata.get("profile_author_id")
    query_author_id = metadata.get("query_author_id")
    if profile_author_id is not None and query_author_id is not None:
        return f"{profile_author_id}|{query_author_id}|{prediction.label}"

    # Example IDs end with a hash, so remove only the prefix component when
    # author IDs are unavailable. This keeps @1/@3 examples grouped together
    # for the same stream in generated benchmark files.
    return prediction.example_id.replace("_p1_", "_pX_").replace("_p3_", "_pX_")


def _safe_mean(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _percentile(values: Sequence[int], percentile: float) -> float | None:
    if not values:
        return None
    if percentile < 0 or percentile > 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")

    # Message counts are discrete. Use nearest-rank/CDF semantics so
    # "P90 messages-to-detect" is an observed prefix length, not 2.8 messages.
    sorted_values = np.sort(np.asarray(values, dtype=float))
    rank = int(np.ceil((percentile / 100) * len(sorted_values)))
    index = min(max(rank - 1, 0), len(sorted_values) - 1)
    return float(sorted_values[index])
