from __future__ import annotations

import json

from identity_verification.datasets.pan import audit_pan_dataset


def test_pan_audit_counts_authors_and_texts(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    truth_path = tmp_path / "truth.jsonl"

    pairs = [
        {"id": "p1", "pair": ["a one", "a two"]},
        {"id": "p2", "pair": ["a three", "b one"]},
        {"id": "p3", "pair": ["b two", "c one"]},
        {"id": "p4", "pair": ["c two", "c three"]},
    ]
    truth = [
        {"id": "p1", "same": True, "authors": ["a", "a"]},
        {"id": "p2", "same": False, "authors": ["a", "b"]},
        {"id": "p3", "same": False, "authors": ["b", "c"]},
        {"id": "p4", "same": True, "authors": ["c", "c"]},
    ]

    pairs_path.write_text("\n".join(json.dumps(row) for row in pairs), encoding="utf-8")
    truth_path.write_text("\n".join(json.dumps(row) for row in truth), encoding="utf-8")

    audit = audit_pan_dataset(pairs_path, truth_path)

    assert audit["positive_pairs"] == 2
    assert audit["negative_pairs"] == 2
    assert audit["unique_authors"] == 3
    assert audit["unique_texts"] == 8
    assert audit["authors_with_min_texts"]["ge_2"] == 3
    assert audit["authors_with_min_texts"]["ge_3"] == 2
    assert audit["authors_with_min_texts"]["ge_10"] == 0

