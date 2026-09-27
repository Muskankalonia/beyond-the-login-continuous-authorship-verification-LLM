"""Generic transformer training and evaluation for profile verification."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from identity_verification.evaluation.profile_metrics import (
    ProfilePrediction,
    evaluate_profile_predictions,
    robustness_gap,
)
from identity_verification.metrics import select_threshold_for_max_f1
from identity_verification.seed import set_seed
from identity_verification.training.profile_dataset import (
    JsonlProfilePairDataset,
    ProfilePairBatch,
    ProfilePairCollator,
)


TEST_SET_ORDER = ("test_random", "test_semi_hard", "test_hard", "test_mixed")


@dataclass(frozen=True)
class TransformerExperimentConfig:
    """Configuration for one model/seed/training-regime experiment."""

    model_name: str
    output_dir: Path
    train_path: Path
    val_path: Path
    curriculum_train_paths: tuple[Path, ...] | None = None
    test_paths: Mapping[str, Path] = field(default_factory=dict)
    seed: int = 13
    max_length: int = 512
    max_profile_chars: int = 6000
    max_query_chars: int = 3000
    max_query_chars_per_text: int = 1000
    train_batch_size: int = 4
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    num_workers: int = 0
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    max_train_steps: int | None = None
    cache_dir: Path | None = None
    local_files_only: bool = False
    save_best_model: bool = True
    show_progress: bool = True


@dataclass(frozen=True)
class TransformerComponents:
    """Loaded tokenizer and sequence-classification model."""

    tokenizer: Any
    model: torch.nn.Module


def load_transformer_components(
    model_name: str,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> TransformerComponents:
    """Load a Hugging Face tokenizer/model, downloading if needed.

    When ``local_files_only`` is false, Hugging Face will fetch missing model
    files from the Hub and cache them locally. When true, this function only uses
    already cached files and fails clearly if they are absent.
    """
    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir_str,
            local_files_only=local_files_only,
            use_fast=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir_str,
            local_files_only=local_files_only,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
    except OSError as exc:
        mode = "offline/local-only" if local_files_only else "online download"
        raise RuntimeError(
            f"Could not load transformer model '{model_name}' in {mode} mode. "
            "If this is the first run, enable internet access or pre-download the model."
        ) from exc

    return TransformerComponents(tokenizer=tokenizer, model=model)


def run_transformer_experiment(config: TransformerExperimentConfig) -> dict[str, Any]:
    """Train a verifier and evaluate it on the configured test sets."""
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if config.curriculum_train_paths is not None and not config.curriculum_train_paths:
        raise ValueError("curriculum_train_paths cannot be empty when provided")

    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    components = load_transformer_components(
        config.model_name,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )
    tokenizer = components.tokenizer
    model = components.model

    device = resolve_device()
    model.to(device)

    val_loader = make_profile_dataloader(
        config.val_path,
        tokenizer=tokenizer,
        batch_size=config.eval_batch_size,
        max_length=config.max_length,
        max_profile_chars=config.max_profile_chars,
        max_query_chars=config.max_query_chars,
        max_query_chars_per_text=config.max_query_chars_per_text,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
        max_examples=config.max_eval_examples,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_result: dict[str, Any] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    global_step = 0
    train_loaders: dict[Path, DataLoader[ProfilePairBatch]] = {}

    for epoch in range(1, config.num_epochs + 1):
        epoch_train_path = train_path_for_epoch(config, epoch)
        train_loader = get_or_create_train_loader(
            train_loaders,
            path=epoch_train_path,
            tokenizer=tokenizer,
            config=config,
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
            max_train_steps=config.max_train_steps,
            start_step=global_step,
            show_progress=config.show_progress,
            description=f"train epoch {epoch}",
        )

        val_predictions = predict_profile_scores(
            model,
            val_loader,
            device=device,
            use_amp=use_amp,
            show_progress=config.show_progress,
            description=f"val epoch {epoch}",
        )
        threshold = select_threshold_for_max_f1(
            [prediction.label for prediction in val_predictions],
            [prediction.score for prediction in val_predictions],
        ).threshold
        val_metrics = evaluate_profile_predictions(
            val_predictions,
            threshold=threshold,
        )
        epoch_result = {
            "epoch": epoch,
            "global_step": global_step,
            "train_path": str(epoch_train_path),
            "train_loss": train_loss,
            "threshold": threshold,
            "val_metrics": val_metrics,
        }
        history.append(epoch_result)

        if best_result is None or _selection_roc_auc(epoch_result) > _selection_roc_auc(best_result):
            best_result = epoch_result
            best_epoch = epoch
            if config.save_best_model:
                save_transformer_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=config.output_dir / "best_model",
                )
            else:
                best_state_dict = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.state_dict().items()
                }

        if config.max_train_steps is not None and global_step >= config.max_train_steps:
            break

    if best_result is None:
        raise RuntimeError("Training finished without producing a validation result")

    if config.save_best_model:
        components = load_transformer_components(
            str(config.output_dir / "best_model"),
            local_files_only=True,
        )
        tokenizer = components.tokenizer
        model = components.model.to(device)
    elif best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)

    test_results = evaluate_test_sets(
        model,
        tokenizer=tokenizer,
        test_paths=config.test_paths,
        threshold=float(best_result["threshold"]),
        config=config,
        device=device,
        use_amp=use_amp,
    )

    result = {
        "model_name": config.model_name,
        "seed": config.seed,
        "device": str(device),
        "train_path": str(config.train_path),
        "val_path": str(config.val_path),
        "test_paths": {name: str(path) for name, path in config.test_paths.items()},
        "training_config": {
            "max_length": config.max_length,
            "max_profile_chars": config.max_profile_chars,
            "max_query_chars": config.max_query_chars,
            "max_query_chars_per_text": config.max_query_chars_per_text,
            "curriculum_train_paths": (
                [str(path) for path in config.curriculum_train_paths]
                if config.curriculum_train_paths is not None
                else None
            ),
            "train_batch_size": config.train_batch_size,
            "eval_batch_size": config.eval_batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "num_epochs": config.num_epochs,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "fp16": config.fp16,
            "max_train_examples": config.max_train_examples,
            "max_eval_examples": config.max_eval_examples,
            "max_train_steps": config.max_train_steps,
        },
        "best_epoch": best_epoch,
        "best_threshold": float(best_result["threshold"]),
        "history": history,
        "test_results": test_results,
        "robustness_gaps": compute_robustness_gaps(test_results),
    }
    write_json(config.output_dir / "result.json", result)
    return result


def train_path_for_epoch(config: TransformerExperimentConfig, epoch: int) -> Path:
    """Resolve the training JSONL path for one epoch."""
    if epoch <= 0:
        raise ValueError(f"epoch must be positive, got {epoch}")
    if config.curriculum_train_paths is None:
        return config.train_path

    schedule_index = min(epoch - 1, len(config.curriculum_train_paths) - 1)
    return config.curriculum_train_paths[schedule_index]


def get_or_create_train_loader(
    train_loaders: dict[Path, DataLoader[ProfilePairBatch]],
    *,
    path: Path,
    tokenizer: Any,
    config: TransformerExperimentConfig,
) -> DataLoader[ProfilePairBatch]:
    """Create a train dataloader once per path and reuse it across epochs."""
    if path not in train_loaders:
        train_loaders[path] = make_profile_dataloader(
            path,
            tokenizer=tokenizer,
            batch_size=config.train_batch_size,
            max_length=config.max_length,
            max_profile_chars=config.max_profile_chars,
            max_query_chars=config.max_query_chars,
            max_query_chars_per_text=config.max_query_chars_per_text,
            shuffle=True,
            seed=config.seed,
            num_workers=config.num_workers,
            max_examples=config.max_train_examples,
        )
    return train_loaders[path]


def make_profile_dataloader(
    path: str | Path,
    *,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    max_profile_chars: int = 6000,
    max_query_chars: int = 3000,
    max_query_chars_per_text: int = 1000,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    max_examples: int | None = None,
) -> DataLoader[ProfilePairBatch]:
    """Create a dataloader for generated profile examples."""
    dataset = JsonlProfilePairDataset(path, max_examples=max_examples)
    collator = ProfilePairCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        max_profile_chars=max_profile_chars,
        max_query_chars=max_query_chars,
        max_query_chars_per_text=max_query_chars_per_text,
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
    dataloader: DataLoader[ProfilePairBatch],
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    max_train_steps: int | None,
    start_step: int,
    show_progress: bool,
    description: str,
) -> tuple[float, int]:
    """Train for one epoch and return average non-accumulated loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    global_step = start_step

    iterator = tqdm(dataloader, desc=description, disable=not show_progress)
    for batch_index, batch in enumerate(iterator, start=1):
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(**batch.inputs, labels=batch.labels)
            loss = outputs.loss
            scaled_loss = loss / gradient_accumulation_steps

        losses.append(float(loss.detach().cpu()))
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

            if max_train_steps is not None and global_step >= max_train_steps:
                break

        if show_progress:
            iterator.set_postfix(loss=f"{losses[-1]:.4f}", step=global_step)

    average_loss = float(sum(losses) / len(losses)) if losses else 0.0
    return average_loss, global_step


@torch.no_grad()
def predict_profile_scores(
    model: torch.nn.Module,
    dataloader: DataLoader[ProfilePairBatch],
    *,
    device: torch.device,
    use_amp: bool,
    show_progress: bool,
    description: str,
) -> list[ProfilePrediction]:
    """Run model inference and return positive-class probabilities."""
    model.eval()
    predictions: list[ProfilePrediction] = []
    iterator = tqdm(dataloader, desc=description, disable=not show_progress)

    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(**batch.inputs)
            scores = torch.softmax(outputs.logits, dim=-1)[:, 1]

        scores_cpu = scores.detach().cpu().tolist()
        labels_cpu = batch.labels.detach().cpu().tolist()
        prefix_lens_cpu = batch.prefix_lens.detach().cpu().tolist()
        for index, score in enumerate(scores_cpu):
            predictions.append(
                ProfilePrediction(
                    example_id=batch.example_ids[index],
                    label=int(labels_cpu[index]),
                    score=float(score),
                    prefix_len=int(prefix_lens_cpu[index]),
                    hardness=batch.hardness[index],
                    metadata=_compact_metadata(batch.metadata[index]),
                )
            )

    return predictions


def evaluate_test_sets(
    model: torch.nn.Module,
    *,
    tokenizer: Any,
    test_paths: Mapping[str, Path],
    threshold: float,
    config: TransformerExperimentConfig,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    """Evaluate a trained model on every requested test set."""
    results: dict[str, Any] = {}
    for test_name in sorted(test_paths, key=_test_set_sort_key):
        dataloader = make_profile_dataloader(
            test_paths[test_name],
            tokenizer=tokenizer,
            batch_size=config.eval_batch_size,
            max_length=config.max_length,
            max_profile_chars=config.max_profile_chars,
            max_query_chars=config.max_query_chars,
            max_query_chars_per_text=config.max_query_chars_per_text,
            shuffle=False,
            seed=config.seed,
            num_workers=config.num_workers,
            max_examples=config.max_eval_examples,
        )
        predictions = predict_profile_scores(
            model,
            dataloader,
            device=device,
            use_amp=use_amp,
            show_progress=config.show_progress,
            description=test_name,
        )
        results[test_name] = evaluate_profile_predictions(
            predictions,
            threshold=threshold,
        )
    return results


def compute_robustness_gaps(test_results: Mapping[str, Any]) -> dict[str, Any]:
    """Compute random-vs-hard gaps for available overall and prefix metrics."""
    if "test_random" not in test_results or "test_hard" not in test_results:
        return {}

    gaps: dict[str, Any] = {
        "overall": {
            metric_name: robustness_gap(
                test_results["test_random"]["overall"],
                test_results["test_hard"]["overall"],
                metric_name=metric_name,
            )
            for metric_name in ("roc_auc", "pr_auc", "f1")
        },
        "by_prefix": {},
    }
    gaps["overall"]["eer_increase"] = _metric_increase(
        test_results["test_random"]["overall"],
        test_results["test_hard"]["overall"],
        metric_name="eer",
    )

    random_prefixes = test_results["test_random"].get("by_prefix", {})
    hard_prefixes = test_results["test_hard"].get("by_prefix", {})
    for prefix_len in sorted(set(random_prefixes) & set(hard_prefixes), key=int):
        gaps["by_prefix"][prefix_len] = {
            metric_name: robustness_gap(
                random_prefixes[prefix_len],
                hard_prefixes[prefix_len],
                metric_name=metric_name,
            )
            for metric_name in ("roc_auc", "pr_auc", "f1")
        }
        gaps["by_prefix"][prefix_len]["eer_increase"] = _metric_increase(
            random_prefixes[prefix_len],
            hard_prefixes[prefix_len],
            metric_name="eer",
        )

    return gaps


def save_transformer_checkpoint(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: str | Path,
) -> None:
    """Save model/tokenizer in Hugging Face format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def move_batch_to_device(batch: ProfilePairBatch, device: torch.device) -> ProfilePairBatch:
    """Move tensors in a profile pair batch to the selected device."""
    return ProfilePairBatch(
        inputs={key: value.to(device) for key, value in batch.inputs.items()},
        labels=batch.labels.to(device),
        example_ids=batch.example_ids,
        hardness=batch.hardness,
        prefix_lens=batch.prefix_lens.to(device),
        metadata=batch.metadata,
    )


def resolve_device() -> torch.device:
    """Use CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compact_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keep_fields = (
        "dataset",
        "seed",
        "split",
        "regime",
        "profile_author_id",
        "query_author_id",
        "similarity_score",
    )
    return {key: metadata[key] for key in keep_fields if key in metadata}


def _selection_roc_auc(epoch_result: Mapping[str, Any]) -> float:
    value = epoch_result["val_metrics"]["overall"].get("roc_auc")
    return float(value) if value is not None else -1.0


def _metric_increase(
    random_metrics: Mapping[str, Any],
    hard_metrics: Mapping[str, Any],
    *,
    metric_name: str,
) -> float | None:
    random_value = random_metrics.get(metric_name)
    hard_value = hard_metrics.get(metric_name)
    if random_value is None or hard_value is None:
        return None
    return float(hard_value - random_value)


def _test_set_sort_key(test_name: str) -> tuple[int, str]:
    if test_name in TEST_SET_ORDER:
        return (TEST_SET_ORDER.index(test_name), test_name)
    return (len(TEST_SET_ORDER), test_name)
