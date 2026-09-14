#!/usr/bin/env python3
"""Idempotently add AEGIR's emitter-exclusion supervision to native 2DGS."""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "AEGIR_EMITTER_EXCLUSION_MASK"
OPACITY_MARKER = "AEGIR_INVALID_OPACITY_LOSS"
CLASS_MASK_MARKER = "AEGIR_INVALID_OPACITY_CLASS_MASK"
PROTECTION_RADIUS_MARKER = "AEGIR_INVALID_OPACITY_PROTECTION_RADIUS"


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one native 2DGS anchor, found {count}: {old[:72]!r}")
    return text.replace(old, new, 1)


def add_invalid_opacity_loss(text: str) -> str:
    if OPACITY_MARKER in text:
        return text
    text = replace_once(
        text,
        """    depth_loss_type,
    normal_supervision,
):
""",
        """    depth_loss_type,
    normal_supervision,
    lambda_invalid_opacity,
):
""",
    )
    text = replace_once(
        text,
        """        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)

        # regularization
""",
        """        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss)

        # AEGIR_INVALID_OPACITY_LOSS: the supervision mask previously removed
        # invalid pixels from every loss but did not discourage Gaussian
        # footprints from covering them. Penalize accumulated opacity one pixel
        # inside the invalid region so valid silhouettes keep an antialiased edge.
        valid_dilated = torch.nn.functional.max_pool2d(
            supervision_mask.unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0)
        invalid_core = valid_dilated < 0.5
        rend_alpha = render_pkg["rend_alpha"]
        if int(invalid_core.sum()) == 0:
            invalid_opacity_loss = rend_alpha.sum() * 0.0
        else:
            invalid_opacity_loss = (
                float(lambda_invalid_opacity) * rend_alpha[invalid_core].mean()
            )

        # regularization
""",
    )
    text = replace_once(
        text,
        "total_loss = loss + dist_loss + normal_loss + normal_l1 + Ldepth",
        "total_loss = loss + dist_loss + normal_loss + normal_l1 + Ldepth + invalid_opacity_loss",
    )
    text = replace_once(
        text,
        """    parser.add_argument("--normal_supervision", action="store_true", default=None)
    args = parser.parse_args(sys.argv[1:])
""",
        """    parser.add_argument("--normal_supervision", action="store_true", default=None)
    parser.add_argument(
        "--lambda_invalid_opacity",
        type=float,
        default=0.0,
        help="Penalize rendered alpha in class-aware, protected invalid space.",
    )
    args = parser.parse_args(sys.argv[1:])
    if args.lambda_invalid_opacity < 0.0:
        parser.error("--lambda_invalid_opacity must be non-negative")
""",
    )
    text = replace_once(
        text,
        """        args.depth_loss_type,
        args.normal_supervision,
    )
""",
        """        args.depth_loss_type,
        args.normal_supervision,
        args.lambda_invalid_opacity,
    )
""",
    )
    return text


def upgrade_invalid_opacity_class_mask(text: str) -> str:
    if CLASS_MASK_MARKER in text:
        return text
    return replace_once(
        text,
        """        # AEGIR_INVALID_OPACITY_LOSS: the supervision mask previously removed
        # invalid pixels from every loss but did not discourage Gaussian
        # footprints from covering them. Penalize accumulated opacity one pixel
        # inside the invalid region so valid silhouettes keep an antialiased edge.
        valid_dilated = torch.nn.functional.max_pool2d(
            supervision_mask.unsqueeze(0), kernel_size=3, stride=1, padding=1
        ).squeeze(0)
        invalid_core = valid_dilated < 0.5
""",
        """        # AEGIR_INVALID_OPACITY_LOSS: discourage Gaussian coverage only
        # in true invalid-depth space. A prepared mask value of 64 identifies
        # emitter pixels that are excluded from surface supervision but must not
        # be treated as empty. Protect a five-pixel band around both supervised
        # surfaces and exempt emitters so thin structures such as blinds retain
        # enough room for their finite Gaussian footprints.
        # AEGIR_INVALID_OPACITY_CLASS_MASK
        exempt_or_valid = (confidence_map >= 0.125).to(image.dtype)
        protected = torch.nn.functional.max_pool2d(
            exempt_or_valid.unsqueeze(0), kernel_size=11, stride=1, padding=5
        ).squeeze(0)
        invalid_core = (confidence_map < 0.125) & (protected < 0.5)
""",
    )


def add_invalid_opacity_protection_radius(text: str) -> str:
    text = text.replace(
        'help="Penalize rendered alpha in the one-pixel-eroded invalid supervision mask.",',
        'help="Penalize rendered alpha in class-aware, protected invalid space.",',
    )
    if PROTECTION_RADIUS_MARKER in text:
        return text
    text = replace_once(
        text,
        """    lambda_invalid_opacity,
):
""",
        """    lambda_invalid_opacity,
    invalid_opacity_protection_radius,
):
""",
    )
    text = replace_once(
        text,
        """        # AEGIR_INVALID_OPACITY_CLASS_MASK
        exempt_or_valid = (confidence_map >= 0.125).to(image.dtype)
        protected = torch.nn.functional.max_pool2d(
            exempt_or_valid.unsqueeze(0), kernel_size=11, stride=1, padding=5
        ).squeeze(0)
        invalid_core = (confidence_map < 0.125) & (protected < 0.5)
""",
        """        # AEGIR_INVALID_OPACITY_CLASS_MASK
        # AEGIR_INVALID_OPACITY_PROTECTION_RADIUS
        exempt_or_valid = (confidence_map >= 0.125).to(image.dtype)
        protection_radius = int(invalid_opacity_protection_radius)
        if protection_radius > 0:
            protection_kernel = 2 * protection_radius + 1
            protected = torch.nn.functional.max_pool2d(
                exempt_or_valid.unsqueeze(0),
                kernel_size=protection_kernel,
                stride=1,
                padding=protection_radius,
            ).squeeze(0)
        else:
            protected = exempt_or_valid
        invalid_core = (confidence_map < 0.125) & (protected < 0.5)
""",
    )
    text = replace_once(
        text,
        """    args = parser.parse_args(sys.argv[1:])
    if args.lambda_invalid_opacity < 0.0:
        parser.error("--lambda_invalid_opacity must be non-negative")
""",
        """    parser.add_argument(
        "--invalid_opacity_protection_radius",
        type=int,
        default=5,
        help="Do not penalize invalid alpha within this many pixels of valid/exempt pixels.",
    )
    args = parser.parse_args(sys.argv[1:])
    if args.lambda_invalid_opacity < 0.0:
        parser.error("--lambda_invalid_opacity must be non-negative")
    if args.invalid_opacity_protection_radius < 0:
        parser.error("--invalid_opacity_protection_radius must be non-negative")
""",
    )
    text = replace_once(
        text,
        """        args.normal_supervision,
        args.lambda_invalid_opacity,
    )
""",
        """        args.normal_supervision,
        args.lambda_invalid_opacity,
        args.invalid_opacity_protection_radius,
    )
""",
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch native 2DGS so mask_path excludes emitter pixels from every loss."
    )
    parser.add_argument("--two-dgs-root", type=Path, required=True)
    args = parser.parse_args()
    train_path = args.two_dgs_root.resolve() / "train.py"
    text = train_path.read_text(encoding="utf-8")
    if MARKER in text:
        updated = add_invalid_opacity_protection_radius(
            upgrade_invalid_opacity_class_mask(add_invalid_opacity_loss(text))
        )
        if updated == text:
            print(
                "emitter exclusion and configurable class-aware invalid-opacity loss already "
                f"installed: {train_path}"
            )
        else:
            train_path.write_text(updated, encoding="utf-8")
            print(f"installed class-aware invalid-opacity loss upgrade: {train_path}")
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
    text = add_invalid_opacity_protection_radius(
        upgrade_invalid_opacity_class_mask(add_invalid_opacity_loss(text))
    )
    train_path.write_text(text, encoding="utf-8")
    print(f"installed emitter exclusion and class-aware invalid-opacity loss: {train_path}")


if __name__ == "__main__":
    main()
