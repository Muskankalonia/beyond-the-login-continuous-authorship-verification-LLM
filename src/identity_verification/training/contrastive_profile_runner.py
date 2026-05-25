"""Bi-encoder contrastive training for profile verification."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn.functional import triplet_margin_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from identity_verification.datasets.profile_examples import (
    DEFAULT_SBERT_HARDNESS_MODEL,
    HARDNESS_MINERS,
    MIXED_REGIME,
    RANDOM_ONLY_REGIME,
    SBERT_HARDNESS_MINER,
    TFIDF_HARDNESS_MINER,
    AuthorHardnessCandidates,
    ProfileRecord,
    build_hardness_index,
    load_hardness_index,
    load_profile_records,
    save_hardness_index,
)
from identity_verification.evaluation.profile_metrics import (
    ProfilePrediction,
    evaluate_profile_predictions,
)
from identity_verification.metrics import select_threshold_for_max_f1
from identity_verification.schemas import HardnessLabel
from identity_verification.seed import set_seed
from identity_verification.training.contrastive_profile_dataset import (
    ContrastiveTripletDataset,
    TripletBatch,
    TripletCollator,
    negative_hardness_sequence,
    sample_contrastive_triplets,
    sample_negative_candidate,
)
from identity_verification.training.profile_dataset import truncate_text
from identity_verification.training.transformer_runner import (
    TEST_SET_ORDER,
    compute_robustness_gaps,
    resolve_device,
    write_json,
)


TEST_REGIMES = {
    "test_random": HardnessLabel.RANDOM.value,
    "test_semi_hard": HardnessLabel.SEMI_HARD.value,
    "test_hard": HardnessLabel.HARD.value,
    "test_mixed": MIXED_REGIME,
}


@dataclass(frozen=True)
class ContrastiveProfileConfig:
    """Configuration for one contrastive profile experiment."""

    model_name: str
    output_dir: Path
    train_profiles_path: Path
    val_profiles_path: Path
    test_profiles_path: Path
    train_regime: str
    seed: int = 13
    val_regime: str | None = None
    curriculum_regimes: tuple[str, ...] | None = None
    triplets_per_author: int = 16
    text_source: str = "all"
    max_length: int = 256
    max_chars_per_text: int = 2000
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    triplet_margin: float = 0.3
    fp16: bool = True
    num_workers: int = 0
    max_train_authors: int | None = None
    max_val_authors: int | None = None
    max_test_authors: int | None = None
    cache_dir: Path | None = None
    hardness_cache_dir: Path | None = None
    train_hardness_miner: str = TFIDF_HARDNESS_MINER
    eval_hardness_miner: str = TFIDF_HARDNESS_MINER
    hardness_miner_model: str = DEFAULT_SBERT_HARDNESS_MODEL
    hardness_mining_batch_size: int = 64
    local_files_only: bool = False
    save_best_model: bool = True
    show_progress: bool = True
    prefix_lengths: tuple[int, ...] = (1, 3)
    test_regimes: Mapping[str, str] = field(default_factory=lambda: TEST_REGIMES)


@dataclass(frozen=True)
class ContrastiveComponents:
    """Loaded tokenizer and encoder backbone."""

    tokenizer: Any
    model: torch.nn.Module


def run_contrastive_profile_experiment(config: ContrastiveProfileConfig) -> dict[str, Any]:
    """Train a contrastive encoder and evaluate profile verification metrics."""
    validate_config(config)
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    log_status(
        config.show_progress,
        f"starting contrastive run model={config.model_name} seed={config.seed} "
        f"train_regime={config.train_regime}",
    )
    log_status(config.show_progress, "loading profile splits")
    train_profiles = load_profile_records(config.train_profiles_path)
    val_profiles = load_profile_records(config.val_profiles_path)
    test_profiles = load_profile_records(config.test_profiles_path)
    train_profiles = limit_profile_records(
        train_profiles,
        max_authors=config.max_train_authors,
        split_name="train",
    )
    val_profiles = limit_profile_records(
        val_profiles,
        max_authors=config.max_val_authors,
        split_name="val",
    )
    test_profiles = limit_profile_records(
        test_profiles,
        max_authors=config.max_test_authors,
        split_name="test",
    )
    log_status(
        config.show_progress,
        f"profile counts train={len(train_profiles)} val={len(val_profiles)} "
        f"test={len(test_profiles)}",
    )

    train_regimes = training_regimes(config)
    train_hardness_index = maybe_build_hardness_index(
        train_profiles,
        regimes=train_regimes,
        cache_path=hardness_cache_path(
            config,
            split_name="train",
            miner=config.train_hardness_miner,
            max_authors=config.max_train_authors,
        ),
        miner=config.train_hardness_miner,
        config=config,
        show_progress=config.show_progress,
    )
    val_regime = config.val_regime or MIXED_REGIME
    val_hardness_index = maybe_build_hardness_index(
        val_profiles,
        regimes=(val_regime,),
        cache_path=hardness_cache_path(
            config,
            split_name="val",
            miner=config.eval_hardness_miner,
            max_authors=config.max_val_authors,
        ),
        miner=config.eval_hardness_miner,
        config=config,
        show_progress=config.show_progress,
    )

    log_status(config.show_progress, "loading encoder model and tokenizer")
    components = load_contrastive_components(
        config.model_name,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )
    tokenizer = components.tokenizer
    model = components.model
    device = resolve_device()
    model.to(device)
    log_status(config.show_progress, f"using device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_result: dict[str, Any] | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, config.num_epochs + 1):
        epoch_seed = seed_for_epoch(config.seed, epoch)
        epoch_regime = train_regime_for_epoch(config, epoch)
        log_status(
            config.show_progress,
            f"epoch {epoch}/{config.num_epochs}: sampling triplets regime={epoch_regime}",
        )
        triplets = sample_contrastive_triplets(
            train_profiles,
            regime=epoch_regime,
            triplets_per_author=config.triplets_per_author,
            random_seed=epoch_seed,
            text_source=config.text_source,
            hardness_index=train_hardness_index,
        )
        log_status(config.show_progress, f"epoch {epoch}: sampled {len(triplets)} triplets")
        train_loader = make_triplet_dataloader(
            triplets,
            tokenizer=tokenizer,
            batch_size=config.train_batch_size,
            max_length=config.max_length,
            max_chars_per_text=config.max_chars_per_text,
            shuffle=True,
            seed=epoch_seed,
            num_workers=config.num_workers,
        )
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            triplet_margin=config.triplet_margin,
            start_step=global_step,
            show_progress=config.show_progress,
            description=f"train epoch {epoch}",
        )

        log_status(config.show_progress, f"epoch {epoch}: validating regime={val_regime}")
        val_predictions = predict_profile_records(
            model,
            tokenizer=tokenizer,
            profiles=val_profiles,
            regime=val_regime,
            seed=config.seed,
            split_name="val",
            prefix_lengths=config.prefix_lengths,
            hardness_index=val_hardness_index,
            batch_size=config.eval_batch_size,
            max_length=config.max_length,
            max_chars_per_text=config.max_chars_per_text,
            device=device,
            use_amp=use_amp,
            show_progress=config.show_progress,
        )
        threshold = select_threshold_for_max_f1(
            [prediction.label for prediction in val_predictions],
            [prediction.score for prediction in val_predictions],
        ).threshold
        val_metrics = evaluate_profile_predictions(val_predictions, threshold=threshold)
        epoch_result = {
            "epoch": epoch,
            "global_step": global_step,
            "train_regime": epoch_regime,
            "triplet_count": len(triplets),
            "train_loss": train_loss,
            "threshold": threshold,
            "val_regime": val_regime,
            "val_metrics": val_metrics,
        }
        history.append(epoch_result)
        log_status(
            config.show_progress,
            f"epoch {epoch}: train_loss={train_loss:.6f} "
            f"val_roc_auc={format_metric_for_log(val_metrics['overall'].get('roc_auc'))}",
        )

        if best_result is None or selection_roc_auc(epoch_result) > selection_roc_auc(best_result):
            best_result = epoch_result
            best_epoch = epoch
            if config.save_best_model:
                save_contrastive_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=config.output_dir / "best_model",
                )

    if best_result is None:
        raise RuntimeError("Training finished without a validation result")

    if config.save_best_model:
        components = load_contrastive_components(
            str(config.output_dir / "best_model"),
            local_files_only=True,
        )
        tokenizer = components.tokenizer
        model = components.model.to(device)

    log_status(config.show_progress, "evaluating test regimes")
    test_results = evaluate_contrastive_test_sets(
        model,
        tokenizer=tokenizer,
        test_profiles=test_profiles,
        test_regimes=config.test_regimes,
        threshold=float(best_result["threshold"]),
        config=config,
        device=device,
        use_amp=use_amp,
    )

    result = {
        "model_name": config.model_name,
        "seed": config.seed,
        "device": str(device),
        "train_profiles_path": str(config.train_profiles_path),
        "val_profiles_path": str(config.val_profiles_path),
        "test_profiles_path": str(config.test_profiles_path),
        "training_config": {
            "train_regime": config.train_regime,
            "val_regime": val_regime,
            "curriculum_regimes": list(config.curriculum_regimes)
            if config.curriculum_regimes is not None
            else None,
            "triplets_per_author": config.triplets_per_author,
            "text_source": config.text_source,
            "max_length": config.max_length,
            "max_chars_per_text": config.max_chars_per_text,
            "train_batch_size": config.train_batch_size,
            "eval_batch_size": config.eval_batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "num_epochs": config.num_epochs,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "triplet_margin": config.triplet_margin,
            "fp16": config.fp16,
            "max_train_authors": config.max_train_authors,
            "max_val_authors": config.max_val_authors,
            "max_test_authors": config.max_test_authors,
            "hardness_cache_dir": str(config.hardness_cache_dir)
            if config.hardness_cache_dir is not None
            else None,
            "train_hardness_miner": config.train_hardness_miner,
            "eval_hardness_miner": config.eval_hardness_miner,
            "hardness_miner_model": config.hardness_miner_model,
            "hardness_mining_batch_size": config.hardness_mining_batch_size,
        },
        "best_epoch": best_epoch,
        "best_threshold": float(best_result["threshold"]),
        "history": history,
        "test_results": test_results,
        "robustness_gaps": compute_robustness_gaps(test_results),
    }
    write_json(config.output_dir / "result.json", result)
    return result


def validate_config(config: ContrastiveProfileConfig) -> None:
    """Validate high-level experiment settings before expensive work."""
    if config.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if config.triplets_per_author <= 0:
        raise ValueError("triplets_per_author must be positive")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.triplet_margin < 0:
        raise ValueError("triplet_margin must be non-negative")
    if config.train_hardness_miner not in HARDNESS_MINERS:
        raise ValueError(
            f"Unsupported train_hardness_miner={config.train_hardness_miner!r}. "
            f"Valid miners: {HARDNESS_MINERS}"
        )
    if config.eval_hardness_miner not in HARDNESS_MINERS:
        raise ValueError(
            f"Unsupported eval_hardness_miner={config.eval_hardness_miner!r}. "
            f"Valid miners: {HARDNESS_MINERS}"
        )
    if config.hardness_mining_batch_size <= 0:
        raise ValueError("hardness_mining_batch_size must be positive")
    for field_name, value in (
        ("max_train_authors", config.max_train_authors),
        ("max_val_authors", config.max_val_authors),
        ("max_test_authors", config.max_test_authors),
    ):
        if value is not None and value < 2:
            raise ValueError(f"{field_name} must be at least 2 when provided")
    if config.curriculum_regimes is not None and not config.curriculum_regimes:
        raise ValueError("curriculum_regimes cannot be empty when provided")
    for path in (
        config.train_profiles_path,
        config.val_profiles_path,
        config.test_profiles_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Profile split file does not exist: {path}")


def log_status(enabled: bool, message: str) -> None:
    """Print a flushed progress message for long-running experiment stages."""
    if enabled:
        print(f"[contrastive] {message}", flush=True)


def format_metric_for_log(value: float | None) -> str:
    """Format optional metrics for concise progress logs."""
    return "None" if value is None else f"{value:.6f}"


def limit_profile_records(
    profiles: Sequence[ProfileRecord],
    *,
    max_authors: int | None,
    split_name: str,
) -> list[ProfileRecord]:
    """Limit profile records for smoke tests while preserving split file order."""
    if max_authors is None:
        return list(profiles)
    if len(profiles) < max_authors:
        raise ValueError(
            f"Requested {max_authors} {split_name} authors, but only "
            f"{len(profiles)} are available"
        )
    return list(profiles[:max_authors])


def load_contrastive_components(
    model_name: str,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> ContrastiveComponents:
    """Load a tokenizer and encoder backbone for contrastive training."""
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir_str,
            local_files_only=local_files_only,
            use_fast=True,
        )
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir_str,
            local_files_only=local_files_only,
        )
    except OSError as exc:
        mode = "offline/local-only" if local_files_only else "online download"
        raise RuntimeError(
            f"Could not load contrastive model '{model_name}' in {mode} mode."
        ) from exc
    return ContrastiveComponents(tokenizer=tokenizer, model=model)


def maybe_build_hardness_index(
    profiles: Sequence[ProfileRecord],
    *,
    regimes: Sequence[str],
    cache_path: Path | None = None,
    miner: str = TFIDF_HARDNESS_MINER,
    config: ContrastiveProfileConfig | None = None,
    show_progress: bool = True,
) -> dict[str, AuthorHardnessCandidates] | None:
    """Build hardness candidates only when at least one regime needs them."""
    if all(regime in {RANDOM_ONLY_REGIME, HardnessLabel.RANDOM.value} for regime in regimes):
        log_status(show_progress, f"hardness index not needed for regimes={tuple(regimes)}")
        return None
    if cache_path is not None and cache_path.is_file():
        cached_index = load_hardness_index(cache_path, include_random=False)
        current_author_ids = {profile.author_id for profile in profiles}
        cached_author_ids = set(cached_index)
        if cached_author_ids == current_author_ids:
            log_status(show_progress, f"loading {miner} hardness index cache: {cache_path}")
            return cached_index
        log_status(
            show_progress,
            f"cache author set mismatch, rebuilding hardness index: {cache_path} "
            f"(current={len(current_author_ids)} cached={len(cached_author_ids)})",
        )

    log_status(
        show_progress,
        f"building {miner} hardness index for regimes={tuple(regimes)}",
    )
    hardness_index = build_hardness_index(
        profiles,
        miner=miner,
        miner_model=config.hardness_miner_model if config is not None else None,
        cache_dir=config.cache_dir if config is not None else None,
        local_files_only=config.local_files_only if config is not None else False,
        max_length=config.max_length if config is not None else 256,
        max_chars_per_text=config.max_chars_per_text if config is not None else 2000,
        batch_size=hardness_mining_batch_size(miner=miner, config=config),
    )
    if cache_path is not None:
        log_status(show_progress, f"saving {miner} hardness index cache: {cache_path}")
        save_hardness_index(cache_path, hardness_index)
    return hardness_index


def hardness_cache_path(
    config: ContrastiveProfileConfig,
    *,
    split_name: str,
    miner: str,
    max_authors: int | None,
) -> Path | None:
    """Return the cache path for a full split, or None for uncached smoke subsets."""
    # Subsetted smoke runs have different candidate rankings, so they must not
    # read or write full-split caches.
    if config.hardness_cache_dir is None or max_authors is not None:
        return None
    if miner == TFIDF_HARDNESS_MINER:
        filename = f"{split_name}_tfidf.json"
    elif miner == SBERT_HARDNESS_MINER:
        filename = (
            f"{split_name}_sbert_{sanitize_cache_component(config.hardness_miner_model)}"
            f"_len{config.max_length}_chars{config.max_chars_per_text}.jsonl"
        )
    else:
        raise ValueError(f"Unsupported hardness miner for cache path: {miner}")
    return config.hardness_cache_dir / f"seed_{config.seed}" / filename


def sanitize_cache_component(value: str) -> str:
    """Return a stable filesystem-safe component for miner cache names."""
    return "".join(character if character.isalnum() else "__" for character in value)


def hardness_mining_batch_size(
    *,
    miner: str,
    config: ContrastiveProfileConfig | None,
) -> int:
    """Return miner-specific batch size while preserving TF-IDF defaults."""
    if config is None:
        return 256
    if miner == SBERT_HARDNESS_MINER:
        return config.hardness_mining_batch_size
    return 256


def training_regimes(config: ContrastiveProfileConfig) -> tuple[str, ...]:
    """Return every negative regime the training loop may use."""
    if config.curriculum_regimes is None:
        return (config.train_regime,)
    return tuple(dict.fromkeys((config.train_regime, *config.curriculum_regimes)))


def train_regime_for_epoch(config: ContrastiveProfileConfig, epoch: int) -> str:
    """Resolve the contrastive negative regime for one epoch."""
    if epoch <= 0:
        raise ValueError(f"epoch must be positive, got {epoch}")
    if config.curriculum_regimes is None:
        return config.train_regime
    schedule_index = min(epoch - 1, len(config.curriculum_regimes) - 1)
    return config.curriculum_regimes[schedule_index]


def seed_for_epoch(seed: int, epoch: int) -> int:
    """Return a stable per-epoch seed with low cross-seed collision risk."""
    if epoch <= 0:
        raise ValueError(f"epoch must be positive, got {epoch}")
    return seed * 1_000_003 + epoch


def make_triplet_dataloader(
    triplets: Sequence,
    *,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    max_chars_per_text: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader[TripletBatch]:
    """Create a dataloader for sampled triplets."""
    dataset = ContrastiveTripletDataset(triplets)
    collator = TripletCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        max_chars_per_text=max_chars_per_text,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[TripletBatch],
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    triplet_margin: float,
    start_step: int,
    show_progress: bool,
    description: str,
) -> tuple[float, int]:
    """Train for one epoch with triplet margin loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    global_step = start_step
    iterator = make_progress_bar(dataloader, desc=description, show_progress=show_progress)

    for batch_index, batch in enumerate(iterator, start=1):
        batch = move_triplet_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            anchor = encode_tokenized_batch(model, batch.anchor)
            positive = encode_tokenized_batch(model, batch.positive)
            negative = encode_tokenized_batch(model, batch.negative)
            loss = triplet_margin_loss(
                anchor,
                positive,
                negative,
                margin=triplet_margin,
                p=2,
            )
            scaled_loss = loss / gradient_accumulation_steps

        if torch.isfinite(loss):
            losses.append(float(loss.detach().float().cpu()))
        scaler.scale(scaled_loss).backward()

        should_step = batch_index % gradient_accumulation_steps == 0
        is_last_batch = batch_index == len(dataloader)
        if should_step or is_last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if show_progress:
            loss_text = f"{losses[-1]:.4f}" if losses else "nan"
            iterator.set_postfix(loss=loss_text, step=global_step)

    average_loss = float(sum(losses) / len(losses)) if losses else 0.0
    return average_loss, global_step


@torch.no_grad()
def predict_profile_records(
    model: torch.nn.Module,
    *,
    tokenizer: Any,
    profiles: Sequence[ProfileRecord],
    regime: str,
    seed: int,
    split_name: str,
    prefix_lengths: Sequence[int],
    hardness_index: Mapping[str, AuthorHardnessCandidates] | None,
    batch_size: int,
    max_length: int,
    max_chars_per_text: int,
    device: torch.device,
    use_amp: bool,
    show_progress: bool,
) -> list[ProfilePrediction]:
    """Score profile/query verification examples with cosine similarities."""
    sorted_profiles = sorted(profiles, key=lambda profile: profile.author_id)
    profile_embeddings = encode_profile_embeddings(
        model,
        tokenizer=tokenizer,
        profiles=sorted_profiles,
        batch_size=batch_size,
        max_length=max_length,
        max_chars_per_text=max_chars_per_text,
        device=device,
        use_amp=use_amp,
        show_progress=show_progress,
        description=f"{split_name} profile embeddings",
    )
    query_embeddings = encode_query_embeddings(
        model,
        tokenizer=tokenizer,
        profiles=sorted_profiles,
        prefix_lengths=prefix_lengths,
        batch_size=batch_size,
        max_length=max_length,
        max_chars_per_text=max_chars_per_text,
        device=device,
        use_amp=use_amp,
        show_progress=show_progress,
        description=f"{split_name} query embeddings",
    )

    return score_profile_predictions(
        profiles=sorted_profiles,
        profile_embeddings=profile_embeddings,
        query_embeddings=query_embeddings,
        regime=regime,
        seed=seed,
        split_name=split_name,
        prefix_lengths=prefix_lengths,
        hardness_index=hardness_index,
    )


def score_profile_predictions(
    *,
    profiles: Sequence[ProfileRecord],
    profile_embeddings: Mapping[str, np.ndarray],
    query_embeddings: Mapping[tuple[str, int], np.ndarray],
    regime: str,
    seed: int,
    split_name: str,
    prefix_lengths: Sequence[int],
    hardness_index: Mapping[str, AuthorHardnessCandidates] | None,
) -> list[ProfilePrediction]:
    """Score profile/query pairs from precomputed profile and query embeddings."""
    sorted_profiles = sorted(profiles, key=lambda profile: profile.author_id)
    predictions: list[ProfilePrediction] = []
    rng = random_for_split(seed=seed, split_name=split_name, regime=regime)
    negative_hardnesses = negative_hardness_sequence(
        total=len(sorted_profiles),
        regime=regime,
        rng=rng,
    )
    author_ids = [profile.author_id for profile in sorted_profiles]

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
            positive_score = cosine_score(
                profile_vector,
                query_embeddings[(profile.author_id, prefix_len)],
            )
            predictions.append(
                make_prediction(
                    split_name=split_name,
                    regime=regime,
                    profile_author_id=profile.author_id,
                    query_author_id=profile.author_id,
                    prefix_len=prefix_len,
                    label=1,
                    hardness=HardnessLabel.POSITIVE.value,
                    score=positive_score,
                    similarity_score=None,
                )
            )

            negative_score = cosine_score(
                profile_vector,
                query_embeddings[(negative_candidate.author_id, prefix_len)],
            )
            predictions.append(
                make_prediction(
                    split_name=split_name,
                    regime=regime,
                    profile_author_id=profile.author_id,
                    query_author_id=negative_candidate.author_id,
                    prefix_len=prefix_len,
                    label=0,
                    hardness=negative_hardness.value,
                    score=negative_score,
                    similarity_score=negative_candidate.similarity_score,
                )
            )

    return predictions


@torch.no_grad()
def encode_profile_embeddings(
    model: torch.nn.Module,
    *,
    tokenizer: Any,
    profiles: Sequence[ProfileRecord],
    batch_size: int,
    max_length: int,
    max_chars_per_text: int,
    device: torch.device,
    use_amp: bool,
    show_progress: bool,
    description: str,
) -> dict[str, np.ndarray]:
    """Encode each profile text separately and average by author."""
    grouped = {profile.author_id: profile.profile_texts for profile in profiles}
    return encode_text_groups(
        model,
        tokenizer=tokenizer,
        grouped_texts=grouped,
        batch_size=batch_size,
        max_length=max_length,
        max_chars_per_text=max_chars_per_text,
        device=device,
        use_amp=use_amp,
        show_progress=show_progress,
        description=description,
    )


@torch.no_grad()
def encode_query_embeddings(
    model: torch.nn.Module,
    *,
    tokenizer: Any,
    profiles: Sequence[ProfileRecord],
    prefix_lengths: Sequence[int],
    batch_size: int,
    max_length: int,
    max_chars_per_text: int,
    device: torch.device,
    use_amp: bool,
    show_progress: bool,
    description: str,
) -> dict[tuple[str, int], np.ndarray]:
    """Encode query prefixes and average their text embeddings."""
    grouped: dict[tuple[str, int], Sequence[str]] = {}
    for profile in profiles:
        for prefix_len in prefix_lengths:
            if len(profile.query_texts) < prefix_len:
                raise ValueError(
                    f"Author {profile.author_id} has only {len(profile.query_texts)} "
                    f"query texts, cannot build prefix {prefix_len}"
                )
            grouped[(profile.author_id, prefix_len)] = profile.query_texts[:prefix_len]
    return encode_text_groups(
        model,
        tokenizer=tokenizer,
        grouped_texts=grouped,
        batch_size=batch_size,
        max_length=max_length,
        max_chars_per_text=max_chars_per_text,
        device=device,
        use_amp=use_amp,
        show_progress=show_progress,
        description=description,
    )


@torch.no_grad()
def encode_text_groups(
    model: torch.nn.Module,
    *,
    tokenizer: Any,
    grouped_texts: Mapping[Any, Sequence[str]],
    batch_size: int,
    max_length: int,
    max_chars_per_text: int,
    device: torch.device,
    use_amp: bool,
    show_progress: bool,
    description: str,
) -> dict[Any, np.ndarray]:
    """Encode individual texts and mean-pool them into one vector per group."""
    model.eval()
    sums: dict[Any, np.ndarray] = {}
    counts: dict[Any, int] = {}
    batch_texts: list[str] = []
    batch_keys: list[Any] = []
    total_texts = sum(len(texts) for texts in grouped_texts.values())
    iterator = make_progress_bar(total=total_texts, desc=description, show_progress=show_progress)

    def flush_batch() -> None:
        nonlocal batch_texts, batch_keys
        if not batch_texts:
            return
        tokenized = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in dict(tokenized).items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            embeddings = encode_tokenized_batch(model, tokenized)
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
        for key, embedding in zip(batch_keys, embeddings_np, strict=True):
            if key not in sums:
                sums[key] = np.zeros_like(embedding)
                counts[key] = 0
            sums[key] += embedding
            counts[key] += 1
        iterator.update(len(batch_texts))
        batch_texts = []
        batch_keys = []

    try:
        for key, texts in grouped_texts.items():
            for text in texts:
                clean = " ".join((text or "").split())
                if not clean:
                    iterator.update(1)
                    continue
                batch_texts.append(truncate_text(clean, max_chars_per_text, keep="head"))
                batch_keys.append(key)
                if len(batch_texts) >= batch_size:
                    flush_batch()
        flush_batch()
    finally:
        iterator.close()

    embeddings_by_key: dict[Any, np.ndarray] = {}
    for key in grouped_texts:
        if counts.get(key, 0) == 0:
            raise ValueError(f"No non-empty texts available for embedding group {key!r}")
        vector = sums[key] / float(counts[key])
        embeddings_by_key[key] = normalize_numpy_vector(vector)
    return embeddings_by_key


def encode_tokenized_batch(
    model: torch.nn.Module,
    tokenized_batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Mean-pool a tokenized batch into L2-normalized embeddings."""
    outputs = model(**tokenized_batch)
    hidden_states = outputs.last_hidden_state
    attention_mask = tokenized_batch["attention_mask"]
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def make_progress_bar(*args: Any, desc: str, show_progress: bool, **kwargs: Any) -> tqdm:
    """Create a tqdm progress bar that stays readable in terminals and logs."""
    return tqdm(
        *args,
        desc=desc,
        disable=not show_progress or not sys.stderr.isatty(),
        dynamic_ncols=True,
        mininterval=5.0,
        miniters=25,
        leave=True,
        **kwargs,
    )


def evaluate_contrastive_test_sets(
    model: torch.nn.Module,
    *,
    tokenizer: Any,
    test_profiles: Sequence[ProfileRecord],
    test_regimes: Mapping[str, str],
    threshold: float,
    config: ContrastiveProfileConfig,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    """Evaluate all configured test regimes."""
    results: dict[str, Any] = {}
    sorted_test_profiles = sorted(test_profiles, key=lambda profile: profile.author_id)
    log_status(config.show_progress, "encoding test profile embeddings")
    profile_embeddings = encode_profile_embeddings(
        model,
        tokenizer=tokenizer,
        profiles=sorted_test_profiles,
        batch_size=config.eval_batch_size,
        max_length=config.max_length,
        max_chars_per_text=config.max_chars_per_text,
        device=device,
        use_amp=use_amp,
        show_progress=config.show_progress,
        description="test profile embeddings",
    )
    log_status(config.show_progress, "encoding test query embeddings")
    query_embeddings = encode_query_embeddings(
        model,
        tokenizer=tokenizer,
        profiles=sorted_test_profiles,
        prefix_lengths=config.prefix_lengths,
        batch_size=config.eval_batch_size,
        max_length=config.max_length,
        max_chars_per_text=config.max_chars_per_text,
        device=device,
        use_amp=use_amp,
        show_progress=config.show_progress,
        description="test query embeddings",
    )
    required_hardness = {
        regime
        for regime in test_regimes.values()
        if regime not in {RANDOM_ONLY_REGIME, HardnessLabel.RANDOM.value}
    }
    test_hardness_index = maybe_build_hardness_index(
        sorted_test_profiles,
        regimes=tuple(required_hardness),
        cache_path=hardness_cache_path(
            config,
            split_name="test",
            miner=config.eval_hardness_miner,
            max_authors=config.max_test_authors,
        ),
        miner=config.eval_hardness_miner,
        config=config,
        show_progress=config.show_progress,
    )
    for test_name in sorted(test_regimes, key=test_set_sort_key):
        log_status(config.show_progress, f"scoring {test_name} regime={test_regimes[test_name]}")
        predictions = score_profile_predictions(
            profiles=sorted_test_profiles,
            profile_embeddings=profile_embeddings,
            query_embeddings=query_embeddings,
            regime=test_regimes[test_name],
            seed=config.seed,
            split_name=test_name,
            prefix_lengths=config.prefix_lengths,
            hardness_index=test_hardness_index,
        )
        results[test_name] = evaluate_profile_predictions(predictions, threshold=threshold)
    return results


def make_prediction(
    *,
    split_name: str,
    regime: str,
    profile_author_id: str,
    query_author_id: str,
    prefix_len: int,
    label: int,
    hardness: str,
    score: float,
    similarity_score: float | None,
) -> ProfilePrediction:
    """Create one profile prediction with stable metadata."""
    example_id = (
        f"{split_name}_{regime}_p{prefix_len}_{profile_author_id}_"
        f"{query_author_id}_{label}_{hardness}"
    )
    return ProfilePrediction(
        example_id=example_id,
        label=label,
        score=score,
        prefix_len=prefix_len,
        hardness=hardness,
        metadata={
            "profile_author_id": profile_author_id,
            "query_author_id": query_author_id,
            "similarity_score": similarity_score,
        },
    )


def cosine_score(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity mapped from [-1, 1] into [0, 1]."""
    cosine = float(np.dot(left, right))
    return float((cosine + 1.0) / 2.0)


def normalize_numpy_vector(vector: np.ndarray) -> np.ndarray:
    """Return a safe L2-normalized numpy vector."""
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def move_triplet_batch_to_device(batch: TripletBatch, device: torch.device) -> TripletBatch:
    """Move token tensors in a triplet batch to the selected device."""
    return TripletBatch(
        anchor={key: value.to(device) for key, value in batch.anchor.items()},
        positive={key: value.to(device) for key, value in batch.positive.items()},
        negative={key: value.to(device) for key, value in batch.negative.items()},
        negative_hardness=batch.negative_hardness,
        metadata=batch.metadata,
    )


def save_contrastive_checkpoint(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: str | Path,
) -> None:
    """Save encoder/tokenizer in Hugging Face format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def selection_roc_auc(epoch_result: Mapping[str, Any]) -> float:
    """Validation ROC-AUC used for checkpoint selection."""
    value = epoch_result["val_metrics"]["overall"].get("roc_auc")
    return float(value) if value is not None else -1.0


def random_for_split(*, seed: int, split_name: str, regime: str) -> random.Random:
    """Create a deterministic RNG for one split/regime."""
    derived = f"{seed}:{split_name}:{regime}"
    return random.Random(derived)


def test_set_sort_key(test_name: str) -> tuple[int, str]:
    """Sort test outputs in the same order as classifier experiments."""
    if test_name in TEST_SET_ORDER:
        return (TEST_SET_ORDER.index(test_name), test_name)
    return (len(TEST_SET_ORDER), test_name)
