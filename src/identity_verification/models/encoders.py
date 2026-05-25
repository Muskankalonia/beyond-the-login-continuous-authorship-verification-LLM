"""Dataset-agnostic text encoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


class TextEncoder(Protocol):
    """Minimal encoder protocol shared by benchmarks and miners."""

    name: str

    def fit(self, texts: Sequence[str]) -> TextEncoder:
        """Fit encoder state from training texts."""

    def encode(self, texts: Sequence[str]) -> Any:
        """Encode texts into vector representations."""


@dataclass
class TfidfEncoder:
    """Word + character n-gram TF-IDF encoder.

    This is the shared TF-IDF component we can use both as a baseline verifier and as the
    primary hard-negative miner for PAN-derived and WildChat benchmarks.
    """

    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    word_max_features: int | None = 200_000
    char_max_features: int | None = 300_000
    min_df: int | float = 2
    max_df: int | float = 1.0
    lowercase: bool = True
    sublinear_tf: bool = True
    norm: str = "l2"
    name: str = "tfidf_word_char"

    def __post_init__(self) -> None:
        self.vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=self.word_ngram_range,
                        max_features=self.word_max_features,
                        min_df=self.min_df,
                        max_df=self.max_df,
                        lowercase=self.lowercase,
                        sublinear_tf=self.sublinear_tf,
                        norm=self.norm,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=self.char_ngram_range,
                        max_features=self.char_max_features,
                        min_df=self.min_df,
                        max_df=self.max_df,
                        lowercase=self.lowercase,
                        sublinear_tf=self.sublinear_tf,
                        norm=self.norm,
                    ),
                ),
            ]
        )
        self.is_fit = False

    def fit(self, texts: Sequence[str]) -> TfidfEncoder:
        if not texts:
            raise ValueError("Cannot fit TfidfEncoder on zero documents")
        self.vectorizer.fit(list(texts))
        self.is_fit = True
        return self

    def fit_iter(
        self,
        texts: Iterable[str],
        *,
        max_documents: int | None = None,
    ) -> TfidfEncoder:
        collected_texts: list[str] = []
        for text in texts:
            if max_documents is not None and len(collected_texts) >= max_documents:
                break
            collected_texts.append(text)
        return self.fit(collected_texts)

    def encode(self, texts: Sequence[str]) -> Any:
        if not self.is_fit:
            raise ValueError("TfidfEncoder must be fit before encode")
        return self.vectorizer.transform(list(texts))
