#!/usr/bin/env python3
"""Idempotently add AEGIR's emitter-exclusion supervision to native 2DGS."""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "AEGIR_EMITTER_EXCLUSION_MASK"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one native 2DGS anchor, found {count}: {old[:72]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch native 2DGS so mask_path excludes emitter pixels from every loss."
    )
    parser.add_argument("--two-dgs-root", type=Path, required=True)
    args = parser.parse_args()
    train_path = args.two_dgs_root.resolve() / "train.py"
    text = train_path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"emitter exclusion already installed: {train_path}")
        return

    text = replace_once(
        text,
        """        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = viewpoint_cam.gt_depth.cuda()
        h, w = gt_depth.shape[-2:]
""",
        """        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = viewpoint_cam.gt_depth.cuda()
        # AEGIR_EMITTER_EXCLUSION_MASK: prepared datasets encode the valid
        # non-emitter region in depth_confidence/mask_path. Composite invalid
        # pixels with GT so they contribute zero RGB gradient and therefore do
        # not seed emitter-region densification.
        confidence_map = viewpoint_cam.depth_confidence.cuda()
        supervision_mask = (confidence_map > 0.5).to(image.dtype)
        masked_image = image * supervision_mask + gt_image * (1.0 - supervision_mask)
        h, w = gt_depth.shape[-2:]
""",
    )
    text = replace_once(
        text,
        """        ssim_loss = ssim(image, gt_image)
        Ll1 = l1_loss(image, gt_image)
""",
        """        ssim_loss = ssim(masked_image, gt_image)
        Ll1 = l1_loss(masked_image, gt_image)
""",
    )
    text = replace_once(
        text,
        """        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
""",
        """        valid_regularizer = supervision_mask > 0.5
        if int(valid_regularizer.sum()) == 0:
            normal_loss = normal_error.sum() * 0.0
            dist_loss = rend_dist.sum() * 0.0
        else:
            normal_loss = lambda_normal * normal_error[valid_regularizer].mean()
            dist_loss = lambda_dist * rend_dist[valid_regularizer].mean()
""",
    )
    text = replace_once(
        text,
        """        confidence_map = viewpoint_cam.depth_confidence.cuda()
        normal_diff = mean_angular_error(surf_normal, gt_normal)
""",
        """        normal_diff = mean_angular_error(surf_normal, gt_normal)
""",
    )
    text = text.replace(
        "depth_mask = gt_depth > 0.1",
        "depth_mask = (gt_depth > 0.1) & (supervision_mask > 0.5)",
    )
    text = replace_once(
        text,
        """                normal_l1 = (
                    l1_loss(
                        surf_normal[~normal_dilated_edges],
                        gt_normal[~normal_dilated_edges],
""",
        """                valid_normal = (~normal_dilated_edges) & (supervision_mask > 0.5).expand_as(gt_normal)
                normal_l1 = (
                    l1_loss(
                        surf_normal[valid_normal],
                        gt_normal[valid_normal],
""",
    )
    text = replace_once(
        text,
        "normal_confidence = (normal_confidence > 0).squeeze(0)",
        "normal_confidence = (normal_confidence > 0).squeeze(0) & (supervision_mask[0] > 0.5)",
    )
    train_path.write_text(text, encoding="utf-8")
    print(f"installed emitter exclusion: {train_path}")


if __name__ == "__main__":
    main()
