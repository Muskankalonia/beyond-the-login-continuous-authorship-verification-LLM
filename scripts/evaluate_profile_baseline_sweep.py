#!/usr/bin/env python
"""Evaluate profile-verification baselines against the contrastive protocol.

This script intentionally mirrors the contrastive profile runner:
- thresholds are selected with ``select_threshold_for_max_f1`` on combined
  k=1+k=3 val_mixed predictions;
- TF-IDF hard/semi-hard imposters come from the existing TF-IDF hardness caches;
- transformer baselines use the same attention-mask mean pooling and L2
  normalization as the fine-tuned contrastive encoders.

The SBERT baseline uses all-mpnet-base-v2, while frozen MiniLM remains only in
the frozen-encoder block. ``max_length=256`` is deliberate for mpnet too: it
matches the contrastive cap rather than mpnet's native 384-token limit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.preprocessing import normalize as normalize_sparse
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from identity_verification.datasets.profile_examples import (  # noqa: E402
    MIXED_REGIME,
    AuthorHardnessCandidates,
    ProfileRecord,
    load_hardness_index,
    load_profile_records,
)
from identity_verification.evaluation.profile_metrics import (  # noqa: E402
    ProfilePrediction,
    evaluate_profile_predictions,
)
from identity_verification.metrics import select_threshold_for_max_f1  # noqa: E402
from identity_verification.models.encoders import TfidfEncoder  # noqa: E402
from identity_verification.schemas import HardnessLabel  # noqa: E402
from identity_verification.training.contrastive_profile_dataset import (  # noqa: E402
    negative_hardness_sequence,
    sample_negative_candidate,
)
from identity_verification.training.contrastive_profile_runner import (  # noqa: E402
    TEST_REGIMES,
    encode_profile_embeddings,
    encode_query_embeddings,
    load_contrastive_components,
    random_for_split,
    test_set_sort_key,
)
from identity_verification.training.transformer_runner import (  # noqa: E402
    resolve_device,
)


BASELINE_CHOICES = ("tfidf", "sbert", "frozen_encoder")
DATASET_CHOICES = ("pan", "wildchat")
DEFAULT_SEEDS = (13, 21, 42)
DEFAULT_SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_FROZEN_MODELS = (
    "distilbert-base-uncased",
    "bert-base-uncased",
    "sentence-transformers/all-MiniLM-L6-v2",
    "google/electra-base-discriminator",
)
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
OPERATING_METRICS = (
    "imposter_catch_rate_at_max_prefix",
    "genuine_survival_rate_at_max_prefix",
    "median_messages_to_detect",
    "p90_messages_to_detect",
)
TABLE_DATASET_LABELS = {
    "pan": "PAN-20",
    "wildchat": "WildChat",
}
TABLE_CONTRASTIVE_MODELS = (
    ("distilbert-base-uncased", "DistilBERT"),
    ("bert-base-uncased", "BERT"),
    ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM"),
    ("google/electra-base-discriminator", "ELECTRA"),
)


@dataclass(frozen=True)
class BaselineSpec:
    """One baseline/model combination to evaluate."""

    baseline: str
    model_name: str | None
    model_label: str
    table_label: str

    @property
    def output_model_name(self) -> str:
        return self.model_name or self.model_label


@dataclass(frozen=True)
class BaselinePredictionRow:
    """One scored profile/query prediction with raw and mapped scores."""

    split_name: str
    regime_name: str
    regime: str
    label: int
    prefix_len: int
    hardness: str
    profile_author_id: str
    query_author_id: str
    raw_score: float
    score: float
    similarity_score: float | None

    @property
    def example_id(self) -> str:
        return (
            f"{self.split_name}_{self.regime}_p{self.prefix_len}_"
            f"{self.profile_author_id}_{self.query_author_id}_{self.label}_{self.hardness}"
        )

    def to_profile_prediction(self) -> ProfilePrediction:
        return ProfilePrediction(
            example_id=self.example_id,
            label=self.label,
            score=self.score,
            prefix_len=self.prefix_len,
            hardness=self.hardness,
            metadata={
                "profile_author_id": self.profile_author_id,
                "query_author_id": self.query_author_id,
                "similarity_score": self.similarity_score,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CHOICES, default=list(DATASET_CHOICES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--baselines", nargs="+", choices=BASELINE_CHOICES, default=list(BASELINE_CHOICES))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_FROZEN_MODELS))
    parser.add_argument("--sbert-model", default=DEFAULT_SBERT_MODEL)
    parser.add_argument("--profiles-root-pan", type=Path, default=Path("data/processed/pan_profiles"))
    parser.add_argument(
        "--profiles-root-wildchat",
        type=Path,
        default=Path("data/processed/wildchat_profiles"),
    )
    parser.add_argument(
        "--hardness-cache-root",
        type=Path,
        default=Path("data/processed/hardness_indexes"),
    )
    parser.add_argument(
        "--contrastive-results-root",
        type=Path,
        default=Path("outputs/experiments/contrastive_profile_transformers"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/profile_baselines"),
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("outputs/summaries/profile_baselines"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--prefix-lengths", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-chars-per-text", type=int, default=2000)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-authors", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    specs = baseline_specs(args)

    for dataset in args.datasets:
        for seed in args.seeds:
            train_profiles, val_profiles, test_profiles = load_profile_splits(
                dataset,
                seed,
                args=args,
            )
            val_hardness = load_tfidf_hardness_index(args, dataset, seed, "val")
            test_hardness = load_tfidf_hardness_index(args, dataset, seed, "test")
            for spec in specs:
                run_or_skip_cell(
                    spec,
                    dataset=dataset,
                    seed=seed,
                    train_profiles=train_profiles,
                    val_profiles=val_profiles,
                    test_profiles=test_profiles,
                    val_hardness=val_hardness,
                    test_hardness=test_hardness,
                    args=args,
                )

    write_summaries(args)


def validate_args(args: argparse.Namespace) -> None:
    if any(prefix_len <= 0 for prefix_len in args.prefix_lengths):
        raise ValueError("--prefix-lengths must contain only positive integers")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")
    if args.max_chars_per_text <= 0:
        raise ValueError("--max-chars-per-text must be positive")
    if args.max_authors is not None and args.max_authors < 2:
        raise ValueError("--max-authors must be at least 2 when provided")


def baseline_specs(args: argparse.Namespace) -> list[BaselineSpec]:
    specs: list[BaselineSpec] = []
    if "tfidf" in args.baselines:
        specs.append(
            BaselineSpec(
                baseline="tfidf",
                model_name=None,
                model_label="tfidf",
                table_label="TF-IDF cosine",
            )
        )
    if "sbert" in args.baselines:
        specs.append(
            BaselineSpec(
                baseline="sbert",
                model_name=args.sbert_model,
                model_label=model_display_name(args.sbert_model),
                table_label="SBERT-mpnet",
            )
        )
    if "frozen_encoder" in args.baselines:
        specs.extend(
            BaselineSpec(
                baseline="frozen_encoder",
                model_name=model_name,
                model_label=model_display_name(model_name),
                table_label=f"Frozen {model_display_name(model_name)}",
            )
            for model_name in args.models
        )
    return specs


def run_or_skip_cell(
    spec: BaselineSpec,
    *,
    dataset: str,
    seed: int,
    train_profiles: Sequence[ProfileRecord],
    val_profiles: Sequence[ProfileRecord],
    test_profiles: Sequence[ProfileRecord],
    val_hardness: Mapping[str, AuthorHardnessCandidates],
    test_hardness: Mapping[str, AuthorHardnessCandidates],
    args: argparse.Namespace,
) -> None:
    output_dir = cell_output_dir(args, dataset=dataset, seed=seed, spec=spec)
    result_path = output_dir / "result.json"
    if not args.force and completed_result_matches(result_path, args=args, spec=spec):
        print(f"[baseline] skip completed {result_path}", flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[baseline] run dataset={dataset} seed={seed} "
        f"baseline={spec.baseline} model={spec.output_model_name}",
        flush=True,
    )
    result, predictions = run_baseline_cell(
        spec,
        dataset=dataset,
        seed=seed,
        train_profiles=train_profiles,
        val_profiles=val_profiles,
        test_profiles=test_profiles,
        val_hardness=val_hardness,
        test_hardness=test_hardness,
        args=args,
    )
    predictions_path = output_dir / "predictions.csv"
    write_predictions_csv(predictions_path, predictions)
    result["predictions_path"] = str(predictions_path)
    write_json(result_path, result)


def run_baseline_cell(
    spec: BaselineSpec,
    *,
    dataset: str,
    seed: int,
    train_profiles: Sequence[ProfileRecord],
    val_profiles: Sequence[ProfileRecord],
    test_profiles: Sequence[ProfileRecord],
    val_hardness: Mapping[str, AuthorHardnessCandidates],
    test_hardness: Mapping[str, AuthorHardnessCandidates],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[BaselinePredictionRow]]:
    if spec.baseline == "tfidf":
        encoder = fit_tfidf_encoder(train_profiles, args=args)
        val_profile_embeddings = encode_tfidf_profile_embeddings(encoder, val_profiles, args=args)
        val_query_embeddings = encode_tfidf_query_embeddings(encoder, val_profiles, args=args)
        test_profile_embeddings = encode_tfidf_profile_embeddings(encoder, test_profiles, args=args)
        test_query_embeddings = encode_tfidf_query_embeddings(encoder, test_profiles, args=args)
        device = "cpu"
        sanity = {
            "training_performed": False,
            "tfidf_fit_split": "train",
            "tfidf_min_df": 1,
            "tfidf_word_ngram_range": [1, 2],
            "tfidf_char_ngram_range": [3, 5],
            "tfidf_word_max_features": 200_000,
            "tfidf_char_max_features": 300_000,
        }
    else:
        components = load_frozen_components(spec, args=args)
        tokenizer = components["tokenizer"]
        model = components["model"]
        device_obj = components["device"]
        use_amp = device_obj.type == "cuda"
        with torch.inference_mode():
            val_profile_embeddings = encode_profile_embeddings(
                model,
                tokenizer=tokenizer,
                profiles=val_profiles,
                batch_size=args.eval_batch_size,
                max_length=args.max_length,
                max_chars_per_text=args.max_chars_per_text,
                device=device_obj,
                use_amp=use_amp,
                show_progress=not args.no_progress,
                description=f"{dataset}/{seed}/{spec.model_label} val profiles",
            )
            val_query_embeddings = encode_query_embeddings(
                model,
                tokenizer=tokenizer,
                profiles=val_profiles,
                prefix_lengths=args.prefix_lengths,
                batch_size=args.eval_batch_size,
                max_length=args.max_length,
                max_chars_per_text=args.max_chars_per_text,
                device=device_obj,
                use_amp=use_amp,
                show_progress=not args.no_progress,
                description=f"{dataset}/{seed}/{spec.model_label} val queries",
            )
            test_profile_embeddings = encode_profile_embeddings(
                model,
                tokenizer=tokenizer,
                profiles=test_profiles,
                batch_size=args.eval_batch_size,
                max_length=args.max_length,
                max_chars_per_text=args.max_chars_per_text,
                device=device_obj,
                use_amp=use_amp,
                show_progress=not args.no_progress,
                description=f"{dataset}/{seed}/{spec.model_label} test profiles",
            )
            test_query_embeddings = encode_query_embeddings(
                model,
                tokenizer=tokenizer,
                profiles=test_profiles,
                prefix_lengths=args.prefix_lengths,
                batch_size=args.eval_batch_size,
                max_length=args.max_length,
                max_chars_per_text=args.max_chars_per_text,
                device=device_obj,
                use_amp=use_amp,
                show_progress=not args.no_progress,
                description=f"{dataset}/{seed}/{spec.model_label} test queries",
            )
        device = str(device_obj)
        sanity = components["sanity"]
        del model
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

    val_predictions = score_baseline_predictions(
        profiles=val_profiles,
        profile_embeddings=val_profile_embeddings,
        query_embeddings=val_query_embeddings,
        regime=MIXED_REGIME,
        regime_name="val_mixed",
        seed=seed,
        split_name="val",
        prefix_lengths=args.prefix_lengths,
        hardness_index=val_hardness,
    )
    threshold = select_threshold_for_max_f1(
        [prediction.label for prediction in val_predictions],
        [prediction.score for prediction in val_predictions],
    ).threshold
    val_result = evaluate_profile_predictions(
        to_profile_predictions(val_predictions),
        threshold=threshold,
    )
    test_results: dict[str, Any] = {}
    test_predictions: list[BaselinePredictionRow] = []
    for test_name in sorted(TEST_REGIMES, key=test_set_sort_key):
        predictions = score_baseline_predictions(
            profiles=test_profiles,
            profile_embeddings=test_profile_embeddings,
            query_embeddings=test_query_embeddings,
            regime=TEST_REGIMES[test_name],
            regime_name=test_name,
            seed=seed,
            split_name=test_name,
            prefix_lengths=args.prefix_lengths,
            hardness_index=test_hardness,
        )
        test_predictions.extend(predictions)
        test_results[test_name] = evaluate_profile_predictions(
            to_profile_predictions(predictions),
            threshold=threshold,
        )

    all_predictions = [*val_predictions, *test_predictions]
    prefix_counts = prefix_prediction_counts(val_predictions)
    result = {
        "status": "done",
        "dataset": dataset,
        "seed": seed,
        "baseline": spec.baseline,
        "model_name": spec.model_name,
        "model_label": spec.model_label,
        "table_label": spec.table_label,
        "device": device,
        "threshold": float(threshold),
        "threshold_source": {
            "function": "identity_verification.metrics.select_threshold_for_max_f1",
            "split": "val",
            "regime": MIXED_REGIME,
            "regime_name": "val_mixed",
            "prefix_lengths": list(args.prefix_lengths),
            "n_examples": len(val_predictions),
            "by_prefix_counts": prefix_counts,
            "policy": "combined_prefixes",
        },
        "config": {
            "profiles_root": str(profiles_root_for_dataset(args, dataset)),
            "hardness_cache_root": str(args.hardness_cache_root),
            "sbert_model": args.sbert_model,
            "max_length": args.max_length,
            "max_chars_per_text": args.max_chars_per_text,
            "prefix_lengths": list(args.prefix_lengths),
            "eval_batch_size": args.eval_batch_size,
            "max_authors": args.max_authors,
            "score_mapping": "(raw_cosine + 1) / 2",
        },
        "sanity": sanity,
        "val_result": val_result,
        "test_results": test_results,
    }
    ranges = score_ranges(all_predictions)
    validate_score_ranges(ranges)
    result["score_ranges"] = ranges
    return result, all_predictions


def load_frozen_components(spec: BaselineSpec, *, args: argparse.Namespace) -> dict[str, Any]:
    if spec.model_name is None:
        raise ValueError("Frozen/SBERT baselines require a model name")
    components = load_contrastive_components(
        spec.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    device = resolve_device()
    model = components.model.to(device)
    model.eval()
    model.requires_grad_(False)
    if model.training:
        raise RuntimeError(f"Frozen model {spec.model_name} is still in training mode")
    any_requires_grad = any(parameter.requires_grad for parameter in model.parameters())
    if any_requires_grad:
        raise RuntimeError(f"Frozen model {spec.model_name} still has trainable parameters")
    return {
        "tokenizer": components.tokenizer,
        "model": model,
        "device": device,
        "sanity": {
            "training_performed": False,
            "optimizer_created": False,
            "model_eval_mode": not model.training,
            "any_requires_grad": any_requires_grad,
            "encoding_context": "torch.inference_mode",
            "pooling": "attention_mask_mean_pool_last_hidden_state_l2_normalized",
        },
    }


def score_baseline_predictions(
    *,
    profiles: Sequence[ProfileRecord],
    profile_embeddings: Mapping[str, Any],
    query_embeddings: Mapping[tuple[str, int], Any],
    regime: str,
    regime_name: str,
    seed: int,
    split_name: str,
    prefix_lengths: Sequence[int],
    hardness_index: Mapping[str, AuthorHardnessCandidates] | None,
) -> list[BaselinePredictionRow]:
    sorted_profiles = sorted(profiles, key=lambda profile: profile.author_id)
    rng = random_for_split(seed=seed, split_name=split_name, regime=regime)
    negative_hardnesses = negative_hardness_sequence(
        total=len(sorted_profiles),
        regime=regime,
        rng=rng,
    )
    author_ids = [profile.author_id for profile in sorted_profiles]
    rows: list[BaselinePredictionRow] = []

    for profile, negative_hardness in zip(sorted_profiles, negative_hardnesses, strict=True):
        profile_vector = profile_embeddings[profile.author_id]
        negative_candidate = sample_negative_candidate(
            profile.author_id,
            negative_hardness,
            valid_author_ids=author_ids,
            hardness_index=hardness_index,
            rng=rng,
        )
        for prefix_len in prefix_lengths:
            rows.append(
                make_prediction_row(
                    split_name=split_name,
                    regime_name=regime_name,
                    regime=regime,
                    profile_author_id=profile.author_id,
                    query_author_id=profile.author_id,
                    label=1,
                    prefix_len=prefix_len,
                    hardness=HardnessLabel.POSITIVE.value,
                    raw_score=cosine_from_normalized_vectors(
                        profile_vector,
                        query_embeddings[(profile.author_id, prefix_len)],
                    ),
                    similarity_score=None,
                )
            )
            rows.append(
                make_prediction_row(
                    split_name=split_name,
                    regime_name=regime_name,
                    regime=regime,
                    profile_author_id=profile.author_id,
                    query_author_id=negative_candidate.author_id,
                    label=0,
                    prefix_len=prefix_len,
                    hardness=negative_hardness.value,
                    raw_score=cosine_from_normalized_vectors(
                        profile_vector,
                        query_embeddings[(negative_candidate.author_id, prefix_len)],
                    ),
                    similarity_score=negative_candidate.similarity_score,
                )
            )
    return rows


def make_prediction_row(
    *,
    split_name: str,
    regime_name: str,
    regime: str,
    profile_author_id: str,
    query_author_id: str,
    label: int,
    prefix_len: int,
    hardness: str,
    raw_score: float,
    similarity_score: float | None,
) -> BaselinePredictionRow:
    mapped_score = (raw_score + 1.0) / 2.0
    return BaselinePredictionRow(
        split_name=split_name,
        regime_name=regime_name,
        regime=regime,
        label=label,
        prefix_len=prefix_len,
        hardness=hardness,
        profile_author_id=profile_author_id,
        query_author_id=query_author_id,
        raw_score=float(raw_score),
        score=float(mapped_score),
        similarity_score=similarity_score,
    )


def cosine_from_normalized_vectors(left: Any, right: Any) -> float:
    """Return cosine for vectors that should already be L2-normalized."""
    left_norm = vector_norm(left)
    right_norm = vector_norm(right)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    if abs(left_norm - 1.0) > 1e-3 or abs(right_norm - 1.0) > 1e-3:
        raise ValueError(
            "cosine_from_normalized_vectors expected L2-normalized inputs, "
            f"got norms {left_norm:.6f} and {right_norm:.6f}"
        )
    product = left.dot(right.T) if hasattr(left, "dot") else np.dot(left, right)
    if hasattr(product, "toarray"):
        value = float(product.toarray().ravel()[0])
    else:
        value = float(np.asarray(product).ravel()[0])
    if value < -1.0001 or value > 1.0001:
        raise ValueError(f"Cosine score outside expected range after normalization: {value}")
    return float(min(1.0, max(-1.0, value)))


def vector_norm(vector: Any) -> float:
    if hasattr(vector, "multiply"):
        return float(np.sqrt(vector.multiply(vector).sum()))
    return float(np.linalg.norm(vector))


def to_profile_predictions(
    predictions: Sequence[BaselinePredictionRow],
) -> list[ProfilePrediction]:
    return [prediction.to_profile_prediction() for prediction in predictions]


def fit_tfidf_encoder(
    train_profiles: Sequence[ProfileRecord],
    *,
    args: argparse.Namespace,
) -> TfidfEncoder:
    texts = [
        document
        for profile in train_profiles
        for document in (profile.profile_text, profile.query_text)
        if document.strip()
    ]
    encoder = TfidfEncoder(min_df=1)
    encoder.fit(texts)
    return encoder


def encode_tfidf_profile_embeddings(
    encoder: TfidfEncoder,
    profiles: Sequence[ProfileRecord],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    del args
    documents = [profile.profile_text for profile in profiles]
    validate_nonempty_documents(documents, context="TF-IDF profile documents")
    matrix = normalize_tfidf_rows(encoder.encode(documents).tocsr())
    return {
        profile.author_id: matrix.getrow(index)
        for index, profile in enumerate(profiles)
    }


def encode_tfidf_query_embeddings(
    encoder: TfidfEncoder,
    profiles: Sequence[ProfileRecord],
    *,
    args: argparse.Namespace,
) -> dict[tuple[str, int], Any]:
    keys: list[tuple[str, int]] = []
    documents: list[str] = []
    for profile in profiles:
        for prefix_len in args.prefix_lengths:
            if len(profile.query_texts) < prefix_len:
                raise ValueError(
                    f"Author {profile.author_id} has only {len(profile.query_texts)} "
                    f"query texts, cannot build prefix {prefix_len}"
                )
            keys.append((profile.author_id, prefix_len))
            documents.append("\n".join(profile.query_texts[:prefix_len]))
    validate_nonempty_documents(documents, context="TF-IDF query documents")
    matrix = normalize_tfidf_rows(encoder.encode(documents).tocsr())
    return {
        key: matrix.getrow(index)
        for index, key in enumerate(keys)
    }


def validate_nonempty_documents(documents: Sequence[str], *, context: str) -> None:
    empty_indexes = [
        index
        for index, document in enumerate(documents)
        if not document.strip()
    ]
    if empty_indexes:
        preview = ", ".join(str(index) for index in empty_indexes[:5])
        raise ValueError(
            f"{context} contains {len(empty_indexes)} empty documents "
            f"(first indexes: {preview})"
        )


def normalize_tfidf_rows(matrix: Any) -> Any:
    """Normalize FeatureUnion rows so sparse dot products are true cosines."""
    return normalize_sparse(matrix, norm="l2", axis=1, copy=False)


def load_profile_splits(
    dataset: str,
    seed: int,
    *,
    args: argparse.Namespace,
) -> tuple[list[ProfileRecord], list[ProfileRecord], list[ProfileRecord]]:
    root = profiles_root_for_dataset(args, dataset)
    seed_dir = root / f"seed_{seed}"
    train_profiles = limit_profiles(load_profile_records(seed_dir / "train.jsonl"), args=args)
    val_profiles = limit_profiles(load_profile_records(seed_dir / "val.jsonl"), args=args)
    test_profiles = limit_profiles(load_profile_records(seed_dir / "test.jsonl"), args=args)
    return train_profiles, val_profiles, test_profiles


def profiles_root_for_dataset(args: argparse.Namespace, dataset: str) -> Path:
    if dataset == "pan":
        return args.profiles_root_pan
    if dataset == "wildchat":
        return args.profiles_root_wildchat
    raise ValueError(f"Unsupported dataset: {dataset}")


def limit_profiles(
    profiles: Sequence[ProfileRecord],
    *,
    args: argparse.Namespace,
) -> list[ProfileRecord]:
    profiles = sorted(profiles, key=lambda profile: profile.author_id)
    if args.max_authors is None:
        return list(profiles)
    if len(profiles) < args.max_authors:
        raise ValueError(
            f"Requested {args.max_authors} authors, but split has only {len(profiles)}"
        )
    return list(profiles[: args.max_authors])


def load_tfidf_hardness_index(
    args: argparse.Namespace,
    dataset: str,
    seed: int,
    split_name: str,
) -> dict[str, AuthorHardnessCandidates]:
    path = args.hardness_cache_root / dataset / f"seed_{seed}" / f"{split_name}_tfidf.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing TF-IDF hardness cache: {path}")
    return load_hardness_index(path, include_random=False)


def completed_result_matches(
    path: Path,
    *,
    args: argparse.Namespace,
    spec: BaselineSpec,
) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (
        result.get("status") == "done"
        and result.get("baseline") == spec.baseline
        and result.get("model_name") == spec.model_name
        and result.get("config", {}).get("prefix_lengths") == list(args.prefix_lengths)
        and result.get("config", {}).get("sbert_model") == args.sbert_model
        and result.get("config", {}).get("max_length") == args.max_length
        and result.get("config", {}).get("max_chars_per_text") == args.max_chars_per_text
        and result.get("config", {}).get("max_authors") == args.max_authors
        and result.get("threshold_source", {}).get("policy") == "combined_prefixes"
    )


def write_summaries(args: argparse.Namespace) -> None:
    result_paths = sorted(args.output_dir.glob("*/*/*/seed_*/result.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    if not results:
        print("[baseline] no results found; skipping summaries", flush=True)
        return

    args.summary_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [row for result in results for row in flatten_result(result)]
    summary_rows = summarize_rows(run_rows)
    write_csv(args.summary_dir / "runs.csv", run_rows)
    write_csv(args.summary_dir / "summary.csv", summary_rows)
    write_csv(args.summary_dir / "per_regime_summary.csv", summary_rows)
    write_json(args.summary_dir / "summary.json", {"runs": run_rows, "summary_rows": summary_rows})
    write_json(args.summary_dir / "per_regime_summary.json", {"summary_rows": summary_rows})
    (args.summary_dir / "summary.md").write_text(render_summary_markdown(summary_rows), encoding="utf-8")
    (args.summary_dir / "table2.md").write_text(
        render_table2_markdown(summary_rows, args=args),
        encoding="utf-8",
    )
    (args.summary_dir / "table2_main_hard_tabular.tex").write_text(
        render_table2_tabular(summary_rows, args=args),
        encoding="utf-8",
    )
    print(f"[baseline] wrote summaries to {args.summary_dir}", flush=True)


def flatten_result(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "dataset": result["dataset"],
        "seed": result["seed"],
        "baseline": result["baseline"],
        "model_name": result.get("model_name"),
        "model_label": result["model_label"],
        "table_label": result["table_label"],
        "threshold": result["threshold"],
    }
    for test_name, metrics in result.get("test_results", {}).items():
        overall = metrics.get("overall", {})
        operating = metrics.get("profile_operating_metrics", {})
        row = {**base, "test_regime": test_name}
        for metric_name in OVERALL_METRICS:
            row[metric_name] = overall.get(metric_name)
        for metric_name in OPERATING_METRICS:
            row[metric_name] = operating.get(metric_name)
        for prefix_len, prefix_metrics in metrics.get("by_prefix", {}).items():
            row[f"roc_auc_at_{prefix_len}"] = prefix_metrics.get("roc_auc")
            row[f"eer_at_{prefix_len}"] = prefix_metrics.get("eer")
        rows.append(row)
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["baseline"]),
            str(row.get("model_name")),
            str(row["table_label"]),
            str(row["test_regime"]),
        )
        groups.setdefault(key, []).append(row)

    metric_names = (
        *OVERALL_METRICS,
        *OPERATING_METRICS,
        "roc_auc_at_1",
        "roc_auc_at_3",
        "eer_at_1",
        "eer_at_3",
    )
    summary_rows: list[dict[str, Any]] = []
    for (dataset, baseline, model_name, table_label, test_regime), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "dataset": dataset,
            "baseline": baseline,
            "model_name": None if model_name == "None" else model_name,
            "table_label": table_label,
            "test_regime": test_regime,
            "seed_count": len({row["seed"] for row in group}),
        }
        for metric_name in metric_names:
            stats = summarize_metric(group, metric_name)
            summary[f"{metric_name}_mean"] = stats["mean"]
            summary[f"{metric_name}_std"] = stats["std"]
            summary[f"{metric_name}_count"] = stats["count"]
        summary_rows.append(summary)
    return summary_rows


def summarize_metric(
    rows: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> dict[str, float | int | None]:
    values = [
        float(row[metric_name])
        for row in rows
        if row.get(metric_name) is not None
    ]
    if not values:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "std": float(stdev(values)) if len(values) > 1 else 0.0,
    }


def render_summary_markdown(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Dataset | Baseline | Test | Seeds | ROC-AUC | TAR@FAR1% | catch@3 | EER |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {label} | {test} | {seeds} | {roc} | {tar} | {catch} | {eer} |".format(
                dataset=row["dataset"],
                label=row["table_label"],
                test=row["test_regime"],
                seeds=row["seed_count"],
                roc=format_mean_std(row, "roc_auc"),
                tar=format_mean_std(row, "tar_at_far_1pct"),
                catch=format_mean_std(row, "imposter_catch_rate_at_max_prefix"),
                eer=format_mean_std(row, "eer"),
            )
        )
    return "\n".join(lines) + "\n"


def render_table2_markdown(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
) -> str:
    rows = table2_rows(summary_rows, args=args)
    lines = [
        "| Dataset | Row | ROC-AUC | TAR@FAR1% | catch@3 | EER |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['label']} | {row['roc_auc']} | "
            f"{row['tar_at_far_1pct']} | {row['catch3']} | {row['eer']} |"
        )
    return "\n".join(lines) + "\n"


def render_table2_tabular(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
) -> str:
    baseline_rows = table2_rows(summary_rows, args=args, reference_only=True)
    contrastive_rows = contrastive_table_rows(args)
    lines = [
        r"\begin{tabular}{l l rr rr rr rr}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{\textbf{ROC-AUC}} & \multicolumn{2}{c}{\textbf{TAR@FAR=1\%}} & \multicolumn{2}{c}{\textbf{catch@3}} & \multicolumn{2}{c}{\textbf{EER}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"\textbf{Dataset} & \textbf{Encoder} & RO & Mix & RO & Mix & RO & Mix & RO & Mix \\",
        r"\midrule",
        r"\multicolumn{10}{l}{\textit{Reference baselines}} \\",
    ]
    for row in baseline_rows:
        lines.append(
            "{dataset} & {label} & \\multicolumn{{2}}{{c}}{{{roc}}} "
            "& \\multicolumn{{2}}{{c}}{{{tar}}} & \\multicolumn{{2}}{{c}}{{{catch}}} "
            "& \\multicolumn{{2}}{{c}}{{{eer}}} \\\\".format(
                dataset=row["dataset"],
                label=row["label"],
                roc=row["roc_auc"],
                tar=row["tar_at_far_1pct"],
                catch=row["catch3"],
                eer=row["eer"],
            )
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{10}{l}{\textit{Contrastive fine-tuning}} \\",
        ]
    )
    for row in contrastive_rows:
        lines.append(
            "{dataset} & {label} & {roc_ro} & {roc_mix} & {tar_ro} & {tar_mix} "
            "& {catch_ro} & {catch_mix} & {eer_ro} & {eer_mix} \\\\".format(**row)
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def table2_rows(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    args: argparse.Namespace,
    reference_only: bool = False,
) -> list[dict[str, str]]:
    del args, reference_only
    hard_rows = [
        row
        for row in summary_rows
        if row["test_regime"] == "test_hard"
    ]
    ordered: list[dict[str, str]] = []
    for dataset in DATASET_CHOICES:
        for label in (
            "TF-IDF cosine",
            "SBERT-mpnet",
            "Frozen DistilBERT",
            "Frozen BERT",
            "Frozen MiniLM",
            "Frozen ELECTRA",
        ):
            row = find_summary_row(hard_rows, dataset=dataset, table_label=label)
            if row is None:
                continue
            ordered.append(
                {
                    "dataset": TABLE_DATASET_LABELS[dataset],
                    "label": label,
                    "roc_auc": format_percent(row.get("roc_auc_mean")),
                    "tar_at_far_1pct": format_percent(row.get("tar_at_far_1pct_mean")),
                    "catch3": format_percent(row.get("imposter_catch_rate_at_max_prefix_mean")),
                    "eer": format_percent(row.get("eer_mean")),
                }
            )
    return ordered


def contrastive_table_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in DATASET_CHOICES:
        for model_name, label in TABLE_CONTRASTIVE_MODELS:
            random_summary = summarize_contrastive_results(
                args.contrastive_results_root,
                dataset=dataset,
                model_name=model_name,
                train_regime="random_only",
            )
            mixed_summary = summarize_contrastive_results(
                args.contrastive_results_root,
                dataset=dataset,
                model_name=model_name,
                train_regime="mixed",
            )
            if random_summary is None or mixed_summary is None:
                continue
            rows.append(
                {
                    "dataset": TABLE_DATASET_LABELS[dataset],
                    "label": label,
                    "roc_ro": format_percent(random_summary.get("roc_auc")),
                    "roc_mix": format_percent(mixed_summary.get("roc_auc")),
                    "tar_ro": format_percent(random_summary.get("tar_at_far_1pct")),
                    "tar_mix": format_percent(mixed_summary.get("tar_at_far_1pct")),
                    "catch_ro": format_percent(
                        random_summary.get("imposter_catch_rate_at_max_prefix")
                    ),
                    "catch_mix": format_percent(
                        mixed_summary.get("imposter_catch_rate_at_max_prefix")
                    ),
                    "eer_ro": format_percent(random_summary.get("eer")),
                    "eer_mix": format_percent(mixed_summary.get("eer")),
                }
            )
    return rows


def summarize_contrastive_results(
    root: Path,
    *,
    dataset: str,
    model_name: str,
    train_regime: str,
) -> dict[str, float | None] | None:
    values: dict[str, list[float]] = {
        "roc_auc": [],
        "tar_at_far_1pct": [],
        "imposter_catch_rate_at_max_prefix": [],
        "eer": [],
    }
    for path in sorted(
        (root / dataset / sanitize_model_name(model_name)).glob(f"seed_*/{train_regime}/result.json")
    ):
        result = json.loads(path.read_text(encoding="utf-8"))
        hard_metrics = result.get("test_results", {}).get("test_hard", {})
        overall = hard_metrics.get("overall", {})
        operating = hard_metrics.get("profile_operating_metrics", {})
        for key in ("roc_auc", "tar_at_far_1pct", "eer"):
            if overall.get(key) is not None:
                values[key].append(float(overall[key]))
        catch_value = operating.get("imposter_catch_rate_at_max_prefix")
        if catch_value is not None:
            values["imposter_catch_rate_at_max_prefix"].append(float(catch_value))
    if not any(values.values()):
        return None
    return {
        key: float(mean(metric_values)) if metric_values else None
        for key, metric_values in values.items()
    }


def find_summary_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    table_label: str,
) -> Mapping[str, Any] | None:
    for row in rows:
        if row["dataset"] == dataset and row["table_label"] == table_label:
            return row
    return None


def prefix_prediction_counts(
    predictions: Sequence[BaselinePredictionRow],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for prediction in predictions:
        key = str(prediction.prefix_len)
        counts[key] = counts.get(key, 0) + 1
    return counts


def score_ranges(
    predictions: Sequence[BaselinePredictionRow],
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[BaselinePredictionRow]] = {}
    for prediction in predictions:
        groups.setdefault(prediction.regime_name, []).append(prediction)
    return {
        group_name: {
            "raw_min": float(min(item.raw_score for item in group)),
            "raw_max": float(max(item.raw_score for item in group)),
            "mapped_min": float(min(item.score for item in group)),
            "mapped_max": float(max(item.score for item in group)),
        }
        for group_name, group in sorted(groups.items())
    }


def validate_score_ranges(ranges: Mapping[str, Mapping[str, float]]) -> None:
    for group_name, values in ranges.items():
        mapped_min = values["mapped_min"]
        mapped_max = values["mapped_max"]
        if mapped_min < -1e-6 or mapped_max > 1.0 + 1e-6:
            raise ValueError(
                f"Mapped scores for {group_name} are outside [0, 1]: "
                f"min={mapped_min}, max={mapped_max}"
            )


def cell_output_dir(
    args: argparse.Namespace,
    *,
    dataset: str,
    seed: int,
    spec: BaselineSpec,
) -> Path:
    return (
        args.output_dir
        / dataset
        / spec.baseline
        / sanitize_model_name(spec.output_model_name)
        / f"seed_{seed}"
    )


def model_display_name(model_name: str) -> str:
    mapping = {
        "distilbert-base-uncased": "DistilBERT",
        "bert-base-uncased": "BERT",
        "sentence-transformers/all-MiniLM-L6-v2": "MiniLM",
        "google/electra-base-discriminator": "ELECTRA",
        "sentence-transformers/all-mpnet-base-v2": "MPNet",
    }
    return mapping.get(model_name, model_name)


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("_")


def format_mean_std(row: Mapping[str, Any], metric_name: str) -> str:
    mean_value = row.get(f"{metric_name}_mean")
    std_value = row.get(f"{metric_name}_std")
    if mean_value is None:
        return "---"
    if std_value is None:
        return f"{100 * float(mean_value):.2f}"
    return f"{100 * float(mean_value):.2f}±{100 * float(std_value):.2f}"


def format_percent(value: Any) -> str:
    if value is None:
        return "---"
    return f"{100 * float(value):.1f}"


def write_predictions_csv(
    path: Path,
    predictions: Sequence[BaselinePredictionRow],
) -> None:
    rows = [
        {
            "example_id": prediction.example_id,
            "split_name": prediction.split_name,
            "regime_name": prediction.regime_name,
            "regime": prediction.regime,
            "label": prediction.label,
            "prefix_len": prediction.prefix_len,
            "hardness": prediction.hardness,
            "profile_author_id": prediction.profile_author_id,
            "query_author_id": prediction.query_author_id,
            "raw_score": prediction.raw_score,
            "score": prediction.score,
            "similarity_score": prediction.similarity_score,
        }
        for prediction in predictions
    ]
    write_csv(path, rows)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
