from __future__ import annotations

import json

import pytest

from identity_verification.datasets.wildchat import audit_wildchat_dataset


def test_wildchat_audit_counts_users_and_english_subset(tmp_path):
    data_path = tmp_path / "wildchat.jsonl"
    rows = [
        {
            "hashed_ip": "u1",
            "language": "English",
            "timestamp": "2024-01-01T00:00:00",
            "conversation": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "help me code"},
            ],
        },
        {
            "hashed_ip": "u1",
            "language": "en",
            "conversation": [{"role": "user", "content": "another question"}],
        },
        {
            "hashed_ip": "u2",
            "language": "Hindi",
            "conversation": [{"role": "user", "content": "namaste"}],
        },
    ]
    data_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    audit = audit_wildchat_dataset(data_path)

    assert audit["row_count"] == 3
    assert audit["default_user_field"] == "hashed_ip"
    assert audit["all_users"]["unique_users"] == 2
    assert audit["all_users"]["message_counts_per_user"]["max"] == 3
    assert audit["all_users"]["message_counts_per_user"]["p90"] == 3
    assert audit["english_users"]["unique_users"] == 1
    assert audit["english_users"]["message_counts_per_user"]["max"] == 3
    assert audit["detected_timestamp_fields"]["timestamp"] == 1


def test_wildchat_audit_discovers_uppercase_supported_suffix(tmp_path):
    data_path = tmp_path / "wildchat.JSONL"
    row = {
        "hashed_ip": "u1",
        "language": "English",
        "conversation": [{"role": "user", "content": "hello"}],
    }
    data_path.write_text(json.dumps(row), encoding="utf-8")

    audit = audit_wildchat_dataset(tmp_path)

    assert audit["file_count"] == 1
    assert audit["row_count"] == 1
    assert audit["all_users"]["unique_users"] == 1


def test_wildchat_audit_can_group_by_message_level_user_field(tmp_path):
    data_path = tmp_path / "wildchat.jsonl"
    rows = [
        {
            "language": "English",
            "conversation": [
                {"role": "user", "content": "hello", "hashed_ip": "u1"},
                {"role": "assistant", "content": "hi"},
            ],
        },
        {
            "language": "English",
            "conversation": [{"role": "user", "content": "missing grouping field"}],
        },
    ]
    data_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    audit = audit_wildchat_dataset(data_path)

    assert audit["missing_user_id_rows"] == 1
    assert audit["all_users"]["unique_users"] == 1
    assert audit["all_users"]["message_counts_per_user"]["max"] == 1


def test_wildchat_audit_rejects_directory_without_supported_files(tmp_path):
    (tmp_path / "notes.txt").write_text("not a dataset", encoding="utf-8")

    with pytest.raises(ValueError, match="No supported input files found"):
        audit_wildchat_dataset(tmp_path)
