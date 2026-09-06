from __future__ import annotations

import copy

import pytest

from scripts.split_shared_val_test import split_metadata


def _metadata() -> dict:
    paths = [f"images/frame_{index:04d}.png" for index in range(8)]
    return {
        "frames": [{"file_path": path} for path in paths],
        "train_filenames": paths[:4],
        "val_filenames": paths[4:],
        "test_filenames": paths[4:],
    }


def test_split_preserves_train_and_records_recoverable_shared_pool() -> None:
    original = _metadata()
    payload, changed = split_metadata(copy.deepcopy(original))
    assert changed
    assert payload["train_filenames"] == original["train_filenames"]
    assert payload["val_filenames"] == original["val_filenames"][::2]
    assert payload["test_filenames"] == original["val_filenames"][1::2]
    assert not set(payload["val_filenames"]).intersection(payload["test_filenames"])
    assert payload["aegir_evaluation_split"]["original_shared_heldout_filenames"] == original["val_filenames"]


def test_split_is_idempotent_and_detects_later_metadata_drift() -> None:
    payload, _ = split_metadata(_metadata())
    verified, changed = split_metadata(copy.deepcopy(payload))
    assert not changed
    assert verified == payload
    payload["test_filenames"] = payload["val_filenames"]
    with pytest.raises(ValueError, match="recorded split"):
        split_metadata(payload)


def test_split_rejects_training_leakage() -> None:
    payload = _metadata()
    payload["train_filenames"].append(payload["val_filenames"][0])
    with pytest.raises(ValueError, match="overlap"):
        split_metadata(payload)
