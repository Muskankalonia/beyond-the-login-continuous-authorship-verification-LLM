from __future__ import annotations

import pytest

from identity_verification.models import TfidfEncoder, score_text_pairs


def test_tfidf_scores_identical_pair_above_unrelated_pair():
    encoder = TfidfEncoder(min_df=1, word_max_features=None, char_max_features=None)
    encoder.fit(
        [
            "the author writes with a very steady rhythm",
            "the author writes with a very steady rhythm",
            "quantum circuits and linear algebra are unrelated here",
        ]
    )

    scores = score_text_pairs(
        encoder,
        [
            "the author writes with a very steady rhythm",
            "the author writes with a very steady rhythm",
        ],
        [
            "the author writes with a very steady rhythm",
            "quantum circuits and linear algebra are unrelated here",
        ],
    )

    assert scores[0] == pytest.approx(1.0)
    assert scores[0] > scores[1]


def test_score_text_pairs_rejects_mismatched_lengths():
    encoder = TfidfEncoder(min_df=1)
    encoder.fit(["a small document", "another small document"])

    with pytest.raises(ValueError, match="same length"):
        score_text_pairs(encoder, ["a small document"], ["one", "two"])
