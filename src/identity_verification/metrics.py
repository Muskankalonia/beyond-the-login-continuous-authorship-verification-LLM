"""Shared binary verification metrics.

All benchmark builders should eventually produce the same minimal evaluation
shape: a binary label and a model score where higher means "same author/user".
This module intentionally has no scikit-learn dependency so it can run in the
lightweight data/audit environment too.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

import numpy as np


DEFAULT_OPERATING_POINT = 0.01


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    f1: float
    precision: float
    recall: float


def evaluate_binary_scores(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    threshold: float = 0.5,
    probabilities: Iterable[float] | None = None,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Compute common binary verification metrics.

    Args:
        labels: Binary labels, where 1 means genuine/same and 0 means imposter/different.
        scores: Model scores where higher means more likely genuine/same.
        threshold: Decision threshold used for F1/precision/recall/accuracy.
        probabilities: Optional calibrated probabilities for Brier score and ECE. If omitted,
            scores are used for calibration metrics only when every score is already in [0, 1].
        ece_bins: Number of equal-width probability bins for expected calibration error.
    """
    label_array, score_array = _validate_labels_and_scores(labels, scores)
    positive_count = int(label_array.sum())
    negative_count = int(len(label_array) - positive_count)
    has_both_classes = positive_count > 0 and negative_count > 0

    threshold_metrics = _threshold_metrics(label_array, score_array, threshold)
    probability_array = _resolve_probability_array(score_array, probabilities)

    result: dict[str, Any] = {
        "n_examples": int(len(label_array)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "threshold": float(threshold),
        **threshold_metrics,
        "roc_auc": None,
        "pr_auc": None,
        "eer": None,
        "eer_threshold": None,
        "far_at_frr_1pct": None,
        "tar_at_far_1pct": None,
        "brier_score": None,
        "ece": None,
    }

    if has_both_classes:
        roc_points = _roc_points(label_array, score_array)
        result["roc_auc"] = _roc_auc_from_ranks(label_array, score_array)
        result["pr_auc"] = _average_precision(label_array, score_array)
        result["eer"], result["eer_threshold"] = _eer_from_roc_points(roc_points)
        result["far_at_frr_1pct"] = _far_at_frr(roc_points, DEFAULT_OPERATING_POINT)
        result["tar_at_far_1pct"] = _tar_at_far(roc_points, DEFAULT_OPERATING_POINT)

    if probability_array is not None:
        result["brier_score"] = _brier_score(label_array, probability_array)
        result["ece"] = _expected_calibration_error(label_array, probability_array, ece_bins)

    return result


def select_threshold_for_max_f1(
    labels: Iterable[int],
    scores: Iterable[float],
) -> ThresholdSelection:
    """Select the threshold with highest F1 on a validation set.

    Ties prefer the higher threshold, which is the more conservative operating point.
    """
    label_array, score_array = _validate_labels_and_scores(labels, scores)
    unique_scores = np.unique(score_array)
    candidates = [float(np.nextafter(unique_scores[-1], np.inf))]
    candidates.extend(float(score) for score in unique_scores[::-1])

    best: ThresholdSelection | None = None
    for candidate in candidates:
        metrics = _threshold_metrics(label_array, score_array, candidate)
        selection = ThresholdSelection(
            threshold=candidate,
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
        )
        if best is None or (selection.f1, selection.threshold) > (best.f1, best.threshold):
            best = selection

    if best is None:
        raise ValueError("Cannot select a threshold from an empty score set")
    return best


def _validate_labels_and_scores(
    labels: Iterable[int],
    scores: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(list(labels))
    score_array = np.asarray(list(scores), dtype=float)

    if label_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional")
    if len(label_array) == 0:
        raise ValueError("At least one label/score pair is required")
    if len(label_array) != len(score_array):
        raise ValueError(
            f"Labels and scores must have the same length: {len(label_array)} != {len(score_array)}"
        )
    if not np.all(np.isfinite(score_array)):
        raise ValueError("Scores must be finite numbers")

    try:
        label_array = label_array.astype(int)
    except ValueError as exc:
        raise ValueError("Labels must be binary integers 0/1") from exc

    unique_labels = set(label_array.tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1 values, got {sorted(unique_labels)}")

    return label_array, score_array


def _resolve_probability_array(
    score_array: np.ndarray,
    probabilities: Iterable[float] | None,
) -> np.ndarray | None:
    if probabilities is None:
        if np.all((0.0 <= score_array) & (score_array <= 1.0)):
            return score_array
        return None

    probability_array = np.asarray(list(probabilities), dtype=float)
    if probability_array.ndim != 1:
        raise ValueError("Probabilities must be one-dimensional")
    if len(probability_array) != len(score_array):
        raise ValueError(
            "Probabilities must have the same length as scores: "
            f"{len(probability_array)} != {len(score_array)}"
        )
    if not np.all(np.isfinite(probability_array)):
        raise ValueError("Probabilities must be finite numbers")
    if not np.all((0.0 <= probability_array) & (probability_array <= 1.0)):
        raise ValueError("Probabilities must be in [0, 1]")
    return probability_array


def _threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    if not isfinite(threshold):
        raise ValueError("Threshold must be finite")

    predictions = scores >= threshold
    positives = labels == 1
    negatives = labels == 0

    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & negatives))
    tn = int(np.sum(~predictions & negatives))
    fn = int(np.sum(~predictions & positives))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(tp + tn, len(labels))
    far = _safe_divide(fp, fp + tn)
    frr = _safe_divide(fn, fn + tp)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _roc_auc_from_ranks(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    ranks = _average_ranks(scores)
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )
    return float(auc)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(labels.sum())
    if positive_count == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    tp = 0
    fp = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0

    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1

        group_positive_count = int(np.sum(sorted_labels[start:end]))
        group_negative_count = (end - start) - group_positive_count
        tp += group_positive_count
        fp += group_negative_count

        recall = tp / positive_count
        precision = tp / (tp + fp)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end

    return float(average_precision)


def _roc_points(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    points = [{"threshold": float("inf"), "fpr": 0.0, "tpr": 0.0}]
    tp = 0
    fp = 0
    start = 0

    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1

        group_positive_count = int(np.sum(sorted_labels[start:end]))
        group_negative_count = (end - start) - group_positive_count
        tp += group_positive_count
        fp += group_negative_count
        points.append(
            {
                "threshold": float(sorted_scores[start]),
                "fpr": fp / negative_count,
                "tpr": tp / positive_count,
            }
        )
        start = end

    return points


def _eer_from_roc_points(points: list[dict[str, float]]) -> tuple[float, float]:
    best_eer = 1.0
    best_gap = float("inf")
    best_threshold = points[0]["threshold"]
    previous = points[0]

    for point in points:
        frr = 1.0 - point["tpr"]
        gap = point["fpr"] - frr
        candidate_eer = (point["fpr"] + frr) / 2
        if abs(gap) < best_gap:
            best_eer = candidate_eer
            best_gap = abs(gap)
            best_threshold = point["threshold"]

        previous_frr = 1.0 - previous["tpr"]
        previous_gap = previous["fpr"] - previous_frr
        if previous is not point and previous_gap * gap < 0:
            interpolation = previous_gap / (previous_gap - gap)
            interpolated_fpr = previous["fpr"] + interpolation * (point["fpr"] - previous["fpr"])
            return float(interpolated_fpr), float(point["threshold"])

        if gap == 0:
            return float(point["fpr"]), float(point["threshold"])
        previous = point

    return float(best_eer), float(best_threshold)


def _far_at_frr(points: list[dict[str, float]], target_frr: float) -> float:
    candidates = [point["fpr"] for point in points if 1.0 - point["tpr"] <= target_frr]
    if not candidates:
        return 1.0
    return float(min(candidates))


def _tar_at_far(points: list[dict[str, float]], target_far: float) -> float:
    candidates = [point["tpr"] for point in points if point["fpr"] <= target_far]
    if not candidates:
        return 0.0
    return float(max(candidates))


def _brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int,
) -> float:
    if n_bins <= 0:
        raise ValueError("ece_bins must be greater than zero")

    ece = 0.0
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            in_bin = (lower <= probabilities) & (probabilities <= upper)
        else:
            in_bin = (lower <= probabilities) & (probabilities < upper)

        bin_count = int(np.sum(in_bin))
        if bin_count == 0:
            continue

        bin_probability = float(np.mean(probabilities[in_bin]))
        bin_positive_rate = float(np.mean(labels[in_bin]))
        ece += bin_count / len(labels) * abs(bin_probability - bin_positive_rate)

    return float(ece)
