from __future__ import annotations

import numpy as np

from scripts.prepare_rgb_pose_mask_view import (
    EMITTER_EXEMPT_VALUE,
    encode_training_mask,
)


def test_training_mask_preserves_invalid_emitter_and_valid_classes() -> None:
    valid = np.array([[False, True, True], [True, False, True]])
    emitter = np.array([[False, True, False], [False, True, False]])

    encoded = encode_training_mask(
        valid,
        emitter,
        encode_emitter_exemption=True,
    )

    np.testing.assert_array_equal(
        encoded,
        np.array(
            [
                [0, EMITTER_EXEMPT_VALUE, 255],
                [255, 0, 255],
            ],
            dtype=np.uint8,
        ),
    )


def test_binary_training_mask_keeps_legacy_emitter_exclusion() -> None:
    valid = np.array([[True, True]])
    emitter = np.array([[True, False]])

    encoded = encode_training_mask(
        valid,
        emitter,
        encode_emitter_exemption=False,
    )

    np.testing.assert_array_equal(encoded, np.array([[0, 255]], dtype=np.uint8))
