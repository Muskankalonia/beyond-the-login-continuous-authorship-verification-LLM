from __future__ import annotations

import pytest

from identity_verification.metrics import (
    evaluate_binary_scores,
    select_threshold_for_max_f1,
)


def test_evaluate_binary_scores_perfect_separation():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]

    metrics = evaluate_binary_scores(labels, scores, threshold=0.5)

    assert metrics["n_examples"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 2
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["far"] == pytest.approx(0.0)
    assert metrics["frr"] == pytest.approx(0.0)
    assert metrics["eer"] == pytest.approx(0.0)
    assert metrics["far_at_frr_1pct"] == pytest.approx(0.0)
    assert metrics["tar_at_far_1pct"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.025)
    assert metrics["ece"] == pytest.approx(0.15)


def test_evaluate_binary_scores_handles_tied_scores():
    labels = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]

    metrics = evaluate_binary_scores(labels, scores, threshold=0.5)

    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["pr_auc"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_select_threshold_for_max_f1_prefers_validation_threshold():
    labels = [0, 1, 1, 0]
    scores = [0.1, 0.4, 0.9, 0.8]

    selection = select_threshold_for_max_f1(labels, scores)

    assert selection.threshold == pytest.approx(0.4)
    assert selection.f1 == pytest.approx(0.8)
    assert selection.precision == pytest.approx(2 / 3)
    assert selection.recall == pytest.approx(1.0)


def test_unbounded_scores_skip_calibration_metrics_without_probabilities():
    labels = [0, 1, 1]
    scores = [-2.0, 0.5, 3.0]

    metrics = evaluate_binary_scores(labels, scores, threshold=0.0)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["brier_score"] is None
    assert metrics["ece"] is None


def test_probabilities_can_be_supplied_for_calibration_metrics():
    labels = [0, 1, 1]
    scores = [-2.0, 0.5, 3.0]
    probabilities = [0.1, 0.6, 0.9]

    metrics = evaluate_binary_scores(
        labels,
        scores,
        threshold=0.0,
        probabilities=probabilities,
    )

    assert metrics["brier_score"] == pytest.approx((0.01 + 0.16 + 0.01) / 3)
    assert metrics["ece"] is not None


def test_one_class_input_keeps_threshold_metrics_but_skips_ranking_metrics():
    metrics = evaluate_binary_scores([1, 1], [0.8, 0.9], threshold=0.5)

    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["eer"] is None
    assert metrics["tar_at_far_1pct"] is None
    assert metrics["far_at_frr_1pct"] is None


def test_invalid_labels_are_rejected():
    with pytest.raises(ValueError, match="binary"):
        evaluate_binary_scores([0, 2], [0.1, 0.9])

