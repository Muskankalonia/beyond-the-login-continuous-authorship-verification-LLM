"""Dataset utilities for profile/query transformer training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch
from torch.utils.data import Dataset


REQUIRED_EXAMPLE_FIELDS = {
    "example_id",
    "profile_text",
    "query_text",
    "label",
    "hardness",
    "prefix_len",
}


class PairTokenizer(Protocol):
    """Minimal tokenizer protocol used by the profile pair collator."""

    def __call__(
        self,
        text: Sequence[str],
        text_pair: Sequence[str],
        *,
        padding: str | bool,
        truncation: bool | str,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        """Tokenize paired profile/query texts."""


@dataclass(frozen=True)
class ProfileExample:
    """One raw-text profile/query verification example."""

    example_id: str
    profile_text: str
    query_texts: tuple[str, ...]
    query_text: str
    label: int
    hardness: str
    prefix_len: int
    metadata: dict[str, Any]


class JsonlProfilePairDataset(Dataset[ProfileExample]):
    """Memory-light dataset for generated raw-text profile examples.

    The dataset stores byte offsets instead of parsed rows. This keeps training
    memory stable even when the JSONL file itself is several GB.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_examples: int | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Profile example file does not exist: {self.path}")
        if max_examples is not None and max_examples <= 0:
            raise ValueError("max_examples must be positive when provided")

        self.offsets = _build_jsonl_offsets(self.path, max_examples=max_examples)
        if not self.offsets:
            raise ValueError(f"Profile example file contains no examples: {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> ProfileExample:
        if index < 0:
            index += len(self.offsets)
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)

        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offsets[index])
            row = json.loads(handle.readline())
        return profile_example_from_row(row, source_path=self.path, row_index=index)


@dataclass(frozen=True)
class ProfilePairBatch:
    """Tokenized model inputs plus labels and metadata."""

    inputs: dict[str, torch.Tensor]
    labels: torch.Tensor
    example_ids: list[str]
    hardness: list[str]
    prefix_lens: torch.Tensor
    metadata: list[dict[str, Any]]


@dataclass(frozen=True)
class ProfilePairCollator:
    """Collate raw profile examples into tokenizer-ready transformer batches."""

    tokenizer: PairTokenizer
    max_length: int = 512
    max_profile_chars: int = 6000
    max_query_chars: int = 3000
    max_query_chars_per_text: int = 1000
    padding: str | bool = "longest"
    truncation: bool | str = True

    def __call__(self, examples: Sequence[ProfileExample]) -> ProfilePairBatch:
        if not examples:
            raise ValueError("Cannot collate an empty batch")

        profile_texts = [
            truncate_text(example.profile_text, self.max_profile_chars, keep="tail")
            for example in examples
        ]
        query_texts = [
            truncate_query_texts(
                example.query_texts,
                max_total_chars=self.max_query_chars,
                max_chars_per_text=self.max_query_chars_per_text,
            )
            for example in examples
        ]
        tokenized = self.tokenizer(
            profile_texts,
            query_texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([example.label for example in examples], dtype=torch.long)
        prefix_lens = torch.tensor(
            [example.prefix_len for example in examples],
            dtype=torch.long,
        )

        return ProfilePairBatch(
            inputs=dict(tokenized),
            labels=labels,
            example_ids=[example.example_id for example in examples],
            hardness=[example.hardness for example in examples],
            prefix_lens=prefix_lens,
            metadata=[example.metadata for example in examples],
        )


def truncate_text(text: str, max_chars: int | None, *, keep: str) -> str:
    """Bound raw text before tokenization to avoid tokenizer memory spikes."""
    if max_chars is None:
        return text
    if max_chars <= 0:
        raise ValueError("max_chars must be positive when provided")
    if len(text) <= max_chars:
        return text
    if keep == "head":
        return text[:max_chars]
    if keep == "tail":
        return text[-max_chars:]
    raise ValueError(f"Unsupported keep strategy: {keep}")


def truncate_query_texts(
    query_texts: Sequence[str],
    *,
    max_total_chars: int | None,
    max_chars_per_text: int | None,
) -> str:
    """Truncate query messages per text, then enforce a total query budget."""
    if not query_texts:
        return ""
    shortened_texts = [
        truncate_text(text, max_chars_per_text, keep="head")
        for text in query_texts
    ]
    joined = "\n".join(shortened_texts)
    return truncate_text(joined, max_total_chars, keep="head")


def profile_example_from_row(
    row: dict[str, Any],
    *,
    source_path: str | Path,
    row_index: int,
) -> ProfileExample:
    """Validate and convert one generated JSONL row."""
    missing_fields = REQUIRED_EXAMPLE_FIELDS - set(row)
    if missing_fields:
        raise ValueError(
            f"Missing required fields at {source_path}:{row_index}: "
            f"{sorted(missing_fields)}"
        )

    label = int(row["label"])
    if label not in {0, 1}:
        raise ValueError(f"Expected binary label at {source_path}:{row_index}, got {label}")

    profile_text = str(row["profile_text"])
    query_text = str(row["query_text"])
    query_texts = tuple(str(text) for text in row.get("query_texts", [query_text]))
    if not profile_text:
        raise ValueError(f"Empty profile_text at {source_path}:{row_index}")
    if not query_text:
        raise ValueError(f"Empty query_text at {source_path}:{row_index}")

    reserved_metadata_fields = {
        "example_id",
        "profile_text",
        "query_texts",
        "query_text",
        "label",
        "hardness",
        "prefix_len",
    }
    metadata = {
        key: value
        for key, value in row.items()
        if key not in reserved_metadata_fields
    }

    return ProfileExample(
        example_id=str(row["example_id"]),
        profile_text=profile_text,
        query_texts=query_texts,
        query_text=query_text,
        label=label,
        hardness=str(row["hardness"]),
        prefix_len=int(row["prefix_len"]),
        metadata=metadata,
    )


def _build_jsonl_offsets(
    path: Path,
    *,
    max_examples: int | None,
) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
                if max_examples is not None and len(offsets) >= max_examples:
                    break
    return offsets
