"""Split duplicated val/test metadata without changing the training frames."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

SPLIT_FORMAT = "aegir-heldout-split-v1"


def _normalized(value: str) -> str:
    normalized = Path(value.replace("\\", "/")).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def split_metadata(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    frames = payload.get("frames", [])
    frame_paths = {_normalized(frame["file_path"]) for frame in frames}
    if len(frame_paths) != len(frames):
        raise ValueError("frame file paths must be unique")

    recorded = payload.get("aegir_evaluation_split")
    if recorded is not None:
        if recorded.get("format") != SPLIT_FORMAT:
            raise ValueError("unsupported aegir_evaluation_split format")
        expected_val = recorded.get("val_filenames", [])
        expected_test = recorded.get("test_filenames", [])
        if payload.get("val_filenames") != expected_val:
            raise ValueError("current val_filenames differ from the recorded split")
        if payload.get("test_filenames") != expected_test:
            raise ValueError("current test_filenames differ from the recorded split")
        return payload, False

    train = [_normalized(path) for path in payload.get("train_filenames", [])]
    val = [_normalized(path) for path in payload.get("val_filenames", [])]
    test = [_normalized(path) for path in payload.get("test_filenames", [])]
    if not train or not val or not test:
        raise ValueError("train_filenames, val_filenames, and test_filenames are required")
    if val != test:
        raise ValueError("refusing to split metadata whose val/test lists are not identical")
    if len(val) < 2:
        raise ValueError("shared held-out pool must contain at least two frames")
    if len(train) != len(set(train)) or len(val) != len(set(val)):
        raise ValueError("split filename lists must not contain duplicates")
    unknown = (set(train) | set(val)).difference(frame_paths)
    if unknown:
        raise ValueError(f"split metadata references unknown frame {min(unknown)!r}")
    overlap = set(train).intersection(val)
    if overlap:
        raise ValueError(f"training and held-out pools overlap at {min(overlap)!r}")
    if set(train) | set(val) != frame_paths:
        missing = frame_paths.difference(set(train) | set(val))
        raise ValueError(f"split metadata leaves frame unassigned: {min(missing)!r}")

    # Interleave the already ordered held-out pool. This preserves train identity
    # and gives both subsets coverage across the original view sequence.
    new_val = val[::2]
    new_test = val[1::2]
    if not new_val or not new_test:
        raise ValueError("independent val and test splits must both be non-empty")
    payload["train_filenames"] = train
    payload["val_filenames"] = new_val
    payload["test_filenames"] = new_test
    payload["aegir_evaluation_split"] = {
        "format": SPLIT_FORMAT,
        "policy": "preserve_train_and_alternate_original_shared_heldout",
        "original_shared_heldout_filenames": val,
        "val_filenames": new_val,
        "test_filenames": new_test,
        "train_frames": len(train),
        "val_frames": len(new_val),
        "test_frames": len(new_test),
    }
    return payload, True


def update_file(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload, changed = split_metadata(payload)
    if not changed:
        return False

    original = path.stat()
    descriptor, temporary_name = tempfile.mkstemp(prefix=".transforms-split-", dir=path.parent)
    try:
        os.fchmod(descriptor, stat.S_IMODE(original.st_mode))
        try:
            os.fchown(descriptor, original.st_uid, original.st_gid)
        except PermissionError:
            pass
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="transforms.json file(s)")
    args = parser.parse_args()
    changed = 0
    for requested in args.paths:
        path = requested.resolve()
        if not path.is_file() or path.name != "transforms.json":
            raise FileNotFoundError(path)
        was_changed = update_file(path)
        changed += int(was_changed)
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{'updated' if was_changed else 'verified'}\t{path.parent.name}\t"
            f"train={len(payload['train_filenames'])}\t"
            f"val={len(payload['val_filenames'])}\t"
            f"test={len(payload['test_filenames'])}"
        )
    print(f"files={len(args.paths)} changed={changed} verified={len(args.paths) - changed}")


if __name__ == "__main__":
    main()
