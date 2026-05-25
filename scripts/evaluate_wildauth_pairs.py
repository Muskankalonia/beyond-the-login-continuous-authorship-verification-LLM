#!/usr/bin/env python
"""Evaluate trained WildChat contrastive encoders on WildAuth pairs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from identity_verification.metrics import (  # noqa: E402
    evaluate_binary_scores,
    select_threshold_for_max_f1,
)
from identity_verification.training.profile_dataset import truncate_text  # noqa: E402


TIER_ORDER = ("easy", "medium", "hard")
DEFAULT_MODELS = ("distilbert-base-uncased", "bert-base-uncased")
DEFAULT_REGIMES = ("random_only", "mixed")
DEFAULT_SEEDS = (13, 21, 42)
PAPER_BASELINES = {
    "LUAR": {"easy": 95.81, "medium": 91.49, "hard": 87.58},
    "UMA(PISC)": {"easy": 98.23, "medium": 94.92, "hard": 92.05},
}


@dataclass(frozen=True)
class WildAuthPair:
    pair_id: str
    dialogue_a_id: str
    dialogue_b_id: str
    split: str
    tier: str
    label: int
    turns_a: tuple[str, ...]
    turns_b: tuple[str, ...]
    source_path: str


@dataclass(frozen=True)
class ScoredPair:
    pair_id: str
    split: str
    tier: str
    label: int
    score: float
    turns_a: int
    turns_b: int
    source_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wildauth-root", type=Path, default=Path("data/raw/wildauth"))
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("outputs/experiments/contrastive_profile_transformers/wildchat"),
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluations/wildauth"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-chars-per-text", type=int, default=2000)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--max-pairs-per-tier",
        type=int,
        default=None,
        help="Optional balanced smoke-test cap per split/tier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs, skipped = load_wildauth_pairs(args.wildauth_root)
    pairs = apply_pair_cap(pairs, args.max_pairs_per_tier)
    print_dataset_report(pairs, skipped)

    all_summary_rows: list[dict[str, Any]] = []
    for model_name in args.models:
        for seed in args.seeds:
            for regime in args.regimes:
                try:
                    run_rows = evaluate_one_checkpoint(
                        args,
                        model_name=model_name,
                        seed=seed,
                        regime=regime,
                        pairs=pairs,
                    )
                except FileNotFoundError as exc:
                    print(
                        f"[wildauth] SKIP model={model_name} seed={seed} "
                        f"regime={regime}: {exc}"
                    )
                    continue
                all_summary_rows.extend(run_rows)

    write_aggregate_outputs(args.output_dir, all_summary_rows)


def load_wildauth_pairs(root: Path) -> tuple[list[WildAuthPair], list[dict[str, Any]]]:
    if not root.exists():
        raise FileNotFoundError(
            f"WildAuth root does not exist: {root}. "
            "Clone/download https://github.com/wwx0015/WildAuth and pass --wildauth-root."
        )

    scan_root = root / "data" if (root / "data").is_dir() else root
    raw_records: list[tuple[dict[str, Any], dict[str, str], Path]] = []
    for path in sorted(scan_root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in {".json", ".jsonl"}:
            raw_records.extend(load_json_records(path))
        elif path.suffix.lower() in {".csv", ".tsv"}:
            raw_records.extend(load_delimited_records(path))
        elif path.suffix.lower() in {".pkl", ".pickle"} and path.stem.endswith("_pairs"):
            raw_records.extend(load_pickle_records(path))

    if not raw_records:
        raise ValueError(f"No JSON/JSONL/CSV/TSV/PKL WildAuth records found under {root}")

    pairs: list[WildAuthPair] = []
    skipped: list[dict[str, Any]] = []
    for index, (record, context, path) in enumerate(raw_records):
        pair_id = str(first_present(record, ("pair_id", "id", "sample_id", "uid")) or index)
        try:
            split = normalize_split(first_present(record, ("split", "partition", "set")) or context.get("split"))
            tier = normalize_tier(
                first_present(record, ("tier", "difficulty", "level", "benchmark", "subset"))
                or context.get("tier")
            )
            label = parse_label(first_present(record, ("label", "y", "target", "same", "is_same", "match")))
            dialogue_a, dialogue_b = extract_dialogue_sides(record)
            turns_a = tuple(extract_user_turns(dialogue_a))
            turns_b = tuple(extract_user_turns(dialogue_b))
            if not turns_a or not turns_b:
                raise ValueError(
                    f"zero user turns after filtering: turns_a={len(turns_a)} turns_b={len(turns_b)}"
                )
            pairs.append(
                WildAuthPair(
                    pair_id=pair_id,
                    dialogue_a_id=str(
                        first_present(record, ("dialogue_a_id", "dialog_a_id", "d_h_id", "left_id"))
                        or f"{pair_id}:a"
                    ),
                    dialogue_b_id=str(
                        first_present(record, ("dialogue_b_id", "dialog_b_id", "d_k_id", "right_id"))
                        or f"{pair_id}:b"
                    ),
                    split=split,
                    tier=tier,
                    label=label,
                    turns_a=turns_a,
                    turns_b=turns_b,
                    source_path=str(path),
                )
            )
        except ValueError as exc:
            skipped.append({"pair_id": pair_id, "source_path": str(path), "reason": str(exc)})

    if not pairs:
        reasons = "\n".join(f"- {item['source_path']}: {item['reason']}" for item in skipped[:20])
        raise ValueError(f"No usable WildAuth pairs were parsed. First skipped records:\n{reasons}")
    return pairs, skipped


def load_json_records(path: Path) -> list[tuple[dict[str, Any], dict[str, str], Path]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if isinstance(value, dict):
                    context = context_from_path(path)
                    context["line"] = str(line_number)
                    rows.append((value, context, path))
        return rows

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return collect_json_records(value, context_from_path(path), path)


def collect_json_records(
    value: Any,
    context: dict[str, str],
    path: Path,
) -> list[tuple[dict[str, Any], dict[str, str], Path]]:
    rows: list[tuple[dict[str, Any], dict[str, str], Path]] = []
    if isinstance(value, list):
        for item in value:
            rows.extend(collect_json_records(item, context, path))
        return rows
    if not isinstance(value, dict):
        return rows

    if looks_like_pair_record(value):
        rows.append((value, dict(context), path))
        return rows

    for key, item in value.items():
        child_context = dict(context)
        lowered = str(key).lower()
        if lowered in TIER_ORDER:
            child_context["tier"] = lowered
        if lowered in {"train", "test", "dev", "val", "valid", "validation"}:
            child_context["split"] = normalize_split(lowered)
        rows.extend(collect_json_records(item, child_context, path))
    return rows


def load_delimited_records(path: Path) -> list[tuple[dict[str, Any], dict[str, str], Path]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [(dict(row), context_from_path(path), path) for row in csv.DictReader(handle, delimiter=delimiter)]


def load_pickle_records(path: Path) -> list[tuple[dict[str, Any], dict[str, str], Path]]:
    import pandas as pd

    value = pd.read_pickle(path)
    context = context_from_path(path)
    if not hasattr(value, "iterrows"):
        raise ValueError(f"Expected a pandas DataFrame in {path}, found {type(value).__name__}")

    rows: list[tuple[dict[str, Any], dict[str, str], Path]] = []
    for index, row in value.iterrows():
        record = row.to_dict()
        if "content_h" in record and "content_k" in record:
            record = dict(record)
            record["pair_id"] = (
                f"{path.stem}:{index}:"
                f"{record.get('d_h_id', 'left')}-{record.get('d_k_id', 'right')}"
            )
            record["dialogue_a"] = alternating_dialogue_turns(record["content_h"])
            record["dialogue_b"] = alternating_dialogue_turns(record["content_k"])
            record["split"] = context.get("split")
            record["tier"] = context.get("tier")
        rows.append((record, dict(context), path))
    return rows


def alternating_dialogue_turns(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return [{"role": "user", "content": normalize_text(str(items))}]
    turns: list[dict[str, str]] = []
    for index, item in enumerate(items):
        text = normalize_text(str(item))
        if not text:
            continue
        role = "user" if index % 2 == 0 else "assistant"
        turns.append({"role": role, "content": text})
    return turns


def context_from_path(path: Path) -> dict[str, str]:
    context: dict[str, str] = {}
    parts = [part.lower() for part in path.parts]
    stem = path.stem.lower()
    if "hard1" in stem:
        context["tier"] = "easy"
    elif "hard2" in stem:
        context["tier"] = "medium"
    elif "hard3" in stem:
        context["tier"] = "hard"
    for tier in TIER_ORDER:
        if tier in parts or tier in stem.split("_"):
            context["tier"] = tier
    for split in ("train", "test", "dev", "val", "valid", "validation"):
        if split in parts or split in stem.split("_"):
            context["split"] = normalize_split(split)
    return context


def looks_like_pair_record(record: Mapping[str, Any]) -> bool:
    keys = {str(key).lower() for key in record}
    has_label = bool(keys & {"label", "y", "target", "same", "is_same", "match"})
    side_patterns = (
        ("dialogue_a", "dialogue_b"),
        ("dialog_a", "dialog_b"),
        ("conversation_a", "conversation_b"),
        ("session_a", "session_b"),
        ("text_a", "text_b"),
        ("content_h", "content_k"),
        ("d_a", "d_b"),
        ("da", "db"),
    )
    has_sides = any(left in keys and right in keys for left, right in side_patterns)
    has_sides = has_sides or ("a" in keys and "b" in keys)
    return has_label and has_sides


def extract_dialogue_sides(record: Mapping[str, Any]) -> tuple[Any, Any]:
    candidates = (
        ("dialogue_a", "dialogue_b"),
        ("dialog_a", "dialog_b"),
        ("conversation_a", "conversation_b"),
        ("session_a", "session_b"),
        ("text_a", "text_b"),
        ("content_h", "content_k"),
        ("D_A", "D_B"),
        ("d_a", "d_b"),
        ("DA", "DB"),
        ("da", "db"),
        ("a", "b"),
    )
    for left, right in candidates:
        if left in record and right in record:
            return parse_maybe_json(record[left]), parse_maybe_json(record[right])
    raise ValueError("missing dialogue side fields")


def extract_user_turns(dialogue: Any) -> list[str]:
    dialogue = parse_maybe_json(dialogue)
    if isinstance(dialogue, str):
        clean = normalize_text(dialogue)
        return [clean] if clean else []
    if isinstance(dialogue, list):
        return extract_user_turns_from_list(dialogue)
    if isinstance(dialogue, dict):
        for key in ("user_turns", "user_messages", "user_texts", "prompts", "queries"):
            if key in dialogue:
                return extract_user_turns(dialogue[key])
        for key in ("turns", "messages", "dialogue", "conversation", "session"):
            if key in dialogue:
                return extract_user_turns(dialogue[key])
        if is_user_turn(dialogue):
            text = text_from_turn(dialogue)
            return [text] if text else []
    return []


def extract_user_turns_from_list(items: Sequence[Any]) -> list[str]:
    turns: list[str] = []
    for item in items:
        item = parse_maybe_json(item)
        if isinstance(item, str):
            clean = normalize_text(item)
            if clean:
                turns.append(clean)
            continue
        if not isinstance(item, dict):
            continue
        role = role_from_turn(item)
        if role is None or is_user_role(role):
            text = text_from_turn(item)
            if text:
                turns.append(text)
    return turns


def is_user_turn(turn: Mapping[str, Any]) -> bool:
    role = role_from_turn(turn)
    return role is None or is_user_role(role)


def role_from_turn(turn: Mapping[str, Any]) -> str | None:
    role = first_present(turn, ("role", "speaker", "from", "author", "type"))
    if role is None:
        return None
    return str(role).strip().lower()


def is_user_role(role: str) -> bool:
    return role in {"user", "human", "client", "prompter", "questioner"}


def text_from_turn(turn: Mapping[str, Any]) -> str:
    value = first_present(
        turn,
        ("content", "text", "message", "value", "prompt", "input", "query", "utterance"),
    )
    if value is None:
        return ""
    return normalize_text(str(value))


def parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def first_present(record: Mapping[str, Any], names: Iterable[str]) -> Any | None:
    lowered = {str(key).lower(): key for key in record}
    for name in names:
        if name in record:
            value = record[name]
        elif name.lower() in lowered:
            value = record[lowered[name.lower()]]
        else:
            continue
        if value is not None and value != "":
            return value
    return None


def normalize_split(value: Any) -> str:
    if value is None:
        raise ValueError("missing split")
    text = str(value).strip().lower()
    if text in {"train", "training"}:
        return "train"
    if text in {"test", "testing"}:
        return "test"
    if text in {"dev", "val", "valid", "validation"}:
        return "validation"
    raise ValueError(f"unknown split {value!r}")


def normalize_tier(value: Any) -> str:
    if value is None:
        raise ValueError("missing tier")
    text = str(value).strip().lower()
    if text in {"hard1", "level1"}:
        return "easy"
    if text in {"hard2", "level2"}:
        return "medium"
    if text in {"hard3", "level3"}:
        return "hard"
    if "easy" in text or "random" in text:
        return "easy"
    if "medium" in text or "topic" in text:
        return "medium"
    if "hard" in text or "cross" in text:
        return "hard"
    raise ValueError(f"unknown tier {value!r}")


def parse_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if int(value) in (0, 1):
            return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "same", "positive", "pos", "genuine", "match"}:
        return 1
    if text in {"0", "false", "different", "negative", "neg", "imposter", "nonmatch", "non-match"}:
        return 0
    raise ValueError(f"unknown label {value!r}")


def apply_pair_cap(
    pairs: Sequence[WildAuthPair],
    max_pairs_per_tier: int | None,
) -> list[WildAuthPair]:
    if max_pairs_per_tier is None:
        return list(pairs)
    capped: list[WildAuthPair] = []
    counts: dict[tuple[str, str, int], int] = {}
    per_label_cap = max(1, max_pairs_per_tier // 2)
    for pair in pairs:
        key = (pair.split, pair.tier, pair.label)
        count = counts.get(key, 0)
        if count >= per_label_cap:
            continue
        capped.append(pair)
        counts[key] = count + 1
    return capped


def print_dataset_report(pairs: Sequence[WildAuthPair], skipped: Sequence[Mapping[str, Any]]) -> None:
    print(f"[wildauth] usable_pairs={len(pairs)} skipped_pairs={len(skipped)}")
    for split in sorted({pair.split for pair in pairs}):
        for tier in TIER_ORDER:
            subset = [pair for pair in pairs if pair.split == split and pair.tier == tier]
            if not subset:
                continue
            labels = {pair.label for pair in subset}
            turns = [len(pair.turns_a) for pair in subset] + [len(pair.turns_b) for pair in subset]
            print(
                f"[wildauth] split={split} tier={tier} n={len(subset)} "
                f"labels={sorted(labels)} user_turns_min/median/max="
                f"{min(turns)}/{statistics.median(turns):.1f}/{max(turns)}"
            )


def evaluate_one_checkpoint(
    args: argparse.Namespace,
    *,
    model_name: str,
    seed: int,
    regime: str,
    pairs: Sequence[WildAuthPair],
) -> list[dict[str, Any]]:
    checkpoint_dir = args.model_root / model_name / f"seed_{seed}" / regime / "best_model"
    result_path = args.model_root / model_name / f"seed_{seed}" / regime / "result.json"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing original result.json: {result_path}")

    wildchat_threshold = load_wildchat_threshold(result_path)
    output_dir = args.output_dir / model_name / f"seed_{seed}" / regime
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(checkpoint_dir, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    print(f"[wildauth] scoring model={model_name} seed={seed} regime={regime} device={device}")
    scored = score_pairs(
        pairs,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
        max_chars_per_text=args.max_chars_per_text,
        device=device,
        show_progress=True,
    )
    write_predictions(output_dir / "predictions.csv", scored)

    thresholds = compute_thresholds(scored, wildchat_threshold=wildchat_threshold)
    metrics = compute_metrics(scored, thresholds)
    write_json(output_dir / "thresholds.json", thresholds)
    write_json(
        output_dir / "metrics.json",
        {
            "model": model_name,
            "seed": seed,
            "regime": regime,
            "checkpoint_dir": str(checkpoint_dir),
            "wildchat_result_path": str(result_path),
            "n_pairs": len(scored),
            "thresholds": thresholds,
            "metrics": metrics,
        },
    )
    return flatten_metric_rows(model_name, seed, regime, metrics)


def load_wildchat_threshold(result_path: Path) -> float:
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    threshold = result.get("best_threshold")
    if threshold is None:
        raise ValueError(f"Missing best_threshold in {result_path}")
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Expected best_threshold in [0, 1] mapped-cosine scale, got {threshold} in {result_path}"
        )
    return threshold


@torch.no_grad()
def score_pairs(
    pairs: Sequence[WildAuthPair],
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    max_chars_per_text: int,
    device: torch.device,
    show_progress: bool,
) -> list[ScoredPair]:
    dialogue_embeddings: dict[str, np.ndarray] = {}
    dialogue_turns: dict[str, tuple[str, ...]] = {}
    for pair in pairs:
        dialogue_turns.setdefault(pair.dialogue_a_id, pair.turns_a)
        dialogue_turns.setdefault(pair.dialogue_b_id, pair.turns_b)

    batch_texts: list[str] = []
    batch_keys: list[str] = []
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    total_turns = sum(len(turns) for turns in dialogue_turns.values())
    progress = tqdm(total=total_turns, desc="wildauth turns", disable=not show_progress)

    def flush() -> None:
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
        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            embeddings = encode_tokenized_batch(model, tokenized)
        for key, embedding in zip(batch_keys, embeddings.detach().cpu().numpy(), strict=True):
            embedding = embedding.astype(np.float32, copy=False)
            if key not in sums:
                sums[key] = np.zeros_like(embedding)
                counts[key] = 0
            sums[key] += embedding
            counts[key] += 1
        progress.update(len(batch_texts))
        batch_texts = []
        batch_keys = []

    try:
        for key, turns in dialogue_turns.items():
            for turn in turns:
                clean = normalize_text(turn)
                if not clean:
                    progress.update(1)
                    continue
                batch_texts.append(truncate_text(clean, max_chars_per_text, keep="head"))
                batch_keys.append(key)
                if len(batch_texts) >= batch_size:
                    flush()
        flush()
    finally:
        progress.close()

    unusable_keys: set[str] = set()
    for key in dialogue_turns:
        if counts.get(key, 0) == 0:
            unusable_keys.add(key)
            continue
        dialogue_embeddings[key] = normalize_numpy_vector(sums[key] / float(counts[key]))

    if unusable_keys:
        dropped_pair_ids = {
            pair.pair_id
            for pair in pairs
            if pair.dialogue_a_id in unusable_keys or pair.dialogue_b_id in unusable_keys
        }
        print(
            f"[wildauth] WARN: dropping {len(dropped_pair_ids)} pairs with empty encoded turn content"
        )

    scored: list[ScoredPair] = []
    for pair in pairs:
        if pair.dialogue_a_id in unusable_keys or pair.dialogue_b_id in unusable_keys:
            continue
        emb_a = dialogue_embeddings[pair.dialogue_a_id]
        emb_b = dialogue_embeddings[pair.dialogue_b_id]
        score = cosine_score(emb_a, emb_b)
        scored.append(
            ScoredPair(
                pair_id=pair.pair_id,
                split=pair.split,
                tier=pair.tier,
                label=pair.label,
                score=score,
                turns_a=len(pair.turns_a),
                turns_b=len(pair.turns_b),
                source_path=pair.source_path,
            )
        )
    return scored


def encode_tokenized_batch(
    model: torch.nn.Module,
    tokenized_batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(**tokenized_batch)
    hidden_states = outputs.last_hidden_state
    attention_mask = tokenized_batch["attention_mask"]
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def normalize_numpy_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize zero vector")
    return (vector / norm).astype(np.float32, copy=False)


def cosine_score(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.dot(left, right))
    return float((cosine + 1.0) / 2.0)


def compute_thresholds(
    scored: Sequence[ScoredPair],
    *,
    wildchat_threshold: float,
) -> dict[str, Any]:
    thresholds: dict[str, Any] = {
        "wildchat_val": wildchat_threshold,
        "wildauth_train_by_tier": {},
    }
    for tier in TIER_ORDER:
        train = [pair for pair in scored if pair.split == "train" and pair.tier == tier]
        if has_both_labels(train):
            selected = select_threshold_for_max_f1(
                [pair.label for pair in train],
                [pair.score for pair in train],
            )
            thresholds["wildauth_train_by_tier"][tier] = {
                "threshold": selected.threshold,
                "f1": selected.f1,
                "precision": selected.precision,
                "recall": selected.recall,
                "n_train": len(train),
            }
        else:
            thresholds["wildauth_train_by_tier"][tier] = None
    return thresholds


def compute_metrics(
    scored: Sequence[ScoredPair],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    test_pairs = [pair for pair in scored if pair.split == "test"]
    for tier in TIER_ORDER:
        tier_pairs = [pair for pair in test_pairs if pair.tier == tier]
        if not tier_pairs:
            continue
        wildchat_metrics = evaluate_binary_scores(
            [pair.label for pair in tier_pairs],
            [pair.score for pair in tier_pairs],
            threshold=float(thresholds["wildchat_val"]),
        )
        tier_threshold = thresholds["wildauth_train_by_tier"].get(tier)
        wildauth_threshold = (
            float(tier_threshold["threshold"])
            if isinstance(tier_threshold, dict)
            else float(thresholds["wildchat_val"])
        )
        wildauth_metrics = evaluate_binary_scores(
            [pair.label for pair in tier_pairs],
            [pair.score for pair in tier_pairs],
            threshold=wildauth_threshold,
        )
        metrics[tier] = {
            "n_test": len(tier_pairs),
            "label_counts": {
                "positive": sum(pair.label == 1 for pair in tier_pairs),
                "negative": sum(pair.label == 0 for pair in tier_pairs),
            },
            "wildchat_val_threshold": wildchat_metrics,
            "wildauth_train_threshold": wildauth_metrics,
        }
    return metrics


def has_both_labels(pairs: Sequence[ScoredPair]) -> bool:
    return {pair.label for pair in pairs} == {0, 1}


def flatten_metric_rows(
    model_name: str,
    seed: int,
    regime: str,
    metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tier, tier_metrics in metrics.items():
        for threshold_source in ("wildchat_val_threshold", "wildauth_train_threshold"):
            flat = {
                "model": model_name,
                "seed": seed,
                "regime": regime,
                "tier": tier,
                "threshold_source": threshold_source,
                "n_test": tier_metrics["n_test"],
                "positive_count": tier_metrics["label_counts"]["positive"],
                "negative_count": tier_metrics["label_counts"]["negative"],
            }
            flat.update(tier_metrics[threshold_source])
            rows.append(flat)
    return rows


def write_predictions(path: Path, scored: Sequence[ScoredPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pair_id",
                "split",
                "tier",
                "label",
                "score",
                "turns_a",
                "turns_b",
                "source_path",
            ),
        )
        writer.writeheader()
        for pair in scored:
            writer.writerow(asdict(pair))


def write_aggregate_outputs(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(output_dir / "summary.csv", rows)
    write_summary_markdown(output_dir / "summary.md", rows)
    write_delta_markdown(output_dir / "delta_summary.md", rows)


def write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_markdown(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    table_rows: list[tuple[str, str, str, str]] = [
        ("LUAR", "95.81", "91.49", "87.58"),
        ("UMA(PISC)", "98.23", "94.92", "92.05"),
    ]
    for model_name in DEFAULT_MODELS:
        for regime in DEFAULT_REGIMES:
            label = f"{display_model_name(model_name)} {regime}"
            values = []
            for tier in TIER_ORDER:
                values.append(format_metric_summary(rows, model_name, regime, tier, "roc_auc"))
            table_rows.append((label, *values))

    with path.open("w", encoding="utf-8") as handle:
        handle.write("| Method | Easy AUC | Medium AUC | Hard AUC |\n")
        handle.write("|---|---:|---:|---:|\n")
        for row in table_rows:
            handle.write("| " + " | ".join(row) + " |\n")


def write_delta_markdown(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| Model | Easy Delta AUC mixed-random | Medium Delta AUC | Hard Delta AUC |\n")
        handle.write("|---|---:|---:|---:|\n")
        for model_name in DEFAULT_MODELS:
            values = [format_delta_summary(rows, model_name, tier) for tier in TIER_ORDER]
            handle.write("| " + " | ".join((display_model_name(model_name), *values)) + " |\n")


def format_metric_summary(
    rows: Sequence[dict[str, Any]],
    model_name: str,
    regime: str,
    tier: str,
    metric: str,
) -> str:
    values = [
        float(row[metric]) * 100.0
        for row in rows
        if row.get("model") == model_name
        and row.get("regime") == regime
        and row.get("tier") == tier
        and row.get("threshold_source") == "wildchat_val_threshold"
        and row.get(metric) is not None
    ]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} +/- {statistics.stdev(values):.2f}"


def format_delta_summary(rows: Sequence[dict[str, Any]], model_name: str, tier: str) -> str:
    random_by_seed = {
        int(row["seed"]): float(row["roc_auc"]) * 100.0
        for row in rows
        if row.get("model") == model_name
        and row.get("regime") == "random_only"
        and row.get("tier") == tier
        and row.get("threshold_source") == "wildchat_val_threshold"
        and row.get("roc_auc") is not None
    }
    mixed_by_seed = {
        int(row["seed"]): float(row["roc_auc"]) * 100.0
        for row in rows
        if row.get("model") == model_name
        and row.get("regime") == "mixed"
        and row.get("tier") == tier
        and row.get("threshold_source") == "wildchat_val_threshold"
        and row.get("roc_auc") is not None
    }
    seeds = sorted(set(random_by_seed) & set(mixed_by_seed))
    deltas = [mixed_by_seed[seed] - random_by_seed[seed] for seed in seeds]
    if not deltas:
        return "-"
    if len(deltas) == 1:
        return f"{deltas[0]:+.2f}"
    return f"{statistics.mean(deltas):+.2f} +/- {statistics.stdev(deltas):.2f}"


def display_model_name(model_name: str) -> str:
    if model_name == "distilbert-base-uncased":
        return "DistilBERT"
    if model_name == "bert-base-uncased":
        return "BERT-base"
    return model_name


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
