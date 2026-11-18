import torch

from aegir.denoising import cross_bilateral_denoise, edge_aware_vector_median_denoise


def test_vector_median_removes_isolated_hdr_firefly():
    color = torch.ones((5, 5, 3))
    color[2, 2] = torch.tensor([100.0, 50.0, 10.0])
    depth = torch.ones((5, 5))
    normals = torch.zeros((5, 5, 3))
    normals[..., 2] = 1.0
    filtered = edge_aware_vector_median_denoise(color, depth, normals)
    assert torch.equal(filtered[2, 2], torch.ones(3))


def test_vector_median_respects_depth_and_normal_edge():
    color = torch.zeros((5, 5, 3))
    color[:, 3:] = 10.0
    depth = torch.ones((5, 5))
    depth[:, 3:] = 4.0
    normals = torch.zeros((5, 5, 3))
    normals[..., 2] = 1.0
    normals[:, 3:, 2] = -1.0
    filtered = edge_aware_vector_median_denoise(color, depth, normals)
    assert torch.equal(filtered[:, :3], color[:, :3])
    assert torch.equal(filtered[:, 3:], color[:, 3:])


def _flat_geometry(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    depth = torch.ones((height, width, 1))
    normals = torch.zeros((height, width, 3))
    normals[..., 2] = 1.0
    return depth, normals


def test_cross_bilateral_preserves_constant_radiance() -> None:
    color = torch.full((7, 9, 3), 2.5)
    depth, normals = _flat_geometry(7, 9)

    filtered = cross_bilateral_denoise(color, depth, normals)

    torch.testing.assert_close(filtered, color)


def test_cross_bilateral_reduces_impulse_on_flat_surface() -> None:
    color = torch.zeros((7, 7, 3))
    color[3, 3] = 10.0
    depth, normals = _flat_geometry(7, 7)

    filtered = cross_bilateral_denoise(color, depth, normals)

    assert 0.0 < float(filtered[3, 3, 0]) < 10.0
    assert float(filtered[3, 2, 0]) > 0.0


def test_cross_bilateral_does_not_blur_across_normal_boundary() -> None:
    color = torch.zeros((7, 8, 3))
    color[:, 4:] = 4.0
    depth, normals = _flat_geometry(7, 8)
    normals[:, 4:, 2] = -1.0

    filtered = cross_bilateral_denoise(color, depth, normals)

    torch.testing.assert_close(filtered[:, :4], color[:, :4])
    torch.testing.assert_close(filtered[:, 4:], color[:, 4:])


def test_cross_bilateral_keeps_invalid_pixels_and_excludes_them() -> None:
    color = torch.ones((5, 5, 3))
    color[2, 2] = 100.0
    depth, normals = _flat_geometry(5, 5)
    valid = torch.ones((5, 5), dtype=torch.bool)
    valid[2, 2] = False

    filtered = cross_bilateral_denoise(color, depth, normals, valid)

    torch.testing.assert_close(filtered[2, 2], color[2, 2])
    torch.testing.assert_close(filtered[2, 1], torch.ones(3))
